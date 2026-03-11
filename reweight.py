#!/usr/bin/env python3
"""
reweight.py

Auto-run version:
- If model exists -> run prediction/reweighting
- If model missing -> train then predict
You can still pass flags:
    --train    : force training
    --predict  : force prediction
    --csv PATH : path to gt_2011.csv (optional)
    --graph    : graph file (default delhi_noida.graphml)
    --paths_dir: directory with path CSVs (default allkpaths)
"""

import os
import glob
import json
import time
import argparse
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import requests
from tqdm import tqdm

# ML imports
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import joblib

# Graph libs
import osmnx as ox
import networkx as nx

# Settings & filenames
MODEL_FILE = "model.pt"
SCALER_FILE = "scaler.pkl"
META_FILE = "meta.json"        # saves feature order + column means
CACHE_FILE = "env_cache.json"
WEIGHTED_DIR = "allkpaths_weighted"
SUMMARY_CSV = "paths_summary.csv"
BEST_HTML = "best_path.html"

OPEN_METEO_FORECAST = "https://api.open-meteo.com/v1/forecast"
OPEN_METEO_ELEV = "https://api.open-meteo.com/v1/elevation"

ox.settings.use_cache = True
ox.settings.log_console = False

DEFAULTS = {
    "graph_file": "delhi_noida.graphml",
    "paths_dir": "allkpaths",
    "out_dir": WEIGHTED_DIR,
    "kround": 4,
    "sleep_api": 0.18,
    "batch_size": 64,
    "epochs": 200,
    "lr": 1e-3,
    "hidden": [64, 32],
    "test_size": 0.2,
    "random_state": 42
}

# -------------------------
# utilities: dataset utils
# -------------------------

def find_gt_csv():
    # look for gt_2011.csv first, else loosely match
    candidates = glob.glob("gt_2011*.csv") + glob.glob("*gt_2011*.csv")
    if candidates:
        return candidates[0]
    # fallback: any csv that seems to have CO, AT, AP, AH
    for f in glob.glob("*.csv"):
        try:
            df = pd.read_csv(f, nrows=3)
            cols = [c.lower() for c in df.columns]
            if any('co' in c for c in cols) and (any('temperature' in c for c in cols) or any('at' == c for c in cols)):
                return f
        except Exception:
            continue
    return None

def detect_feature_columns(df):
    """
    Map the dataset columns to expected turbine features:
    AT, AP, AH, AFDP, GTEP, TIT, TAT, TEY, CDP, CO (target)
    Returns dict mapping feature name -> column name in df.
    """
    mapping = {}
    lower = {c.lower(): c for c in df.columns}
    # heuristics
    for c in df.columns:
        lc = c.lower()
        if 'ambient temperature' in lc or lc == 'at' or lc.startswith('at'):
            mapping['AT'] = c
        if 'ambient pressure' in lc or lc == 'ap' or lc.startswith('ap'):
            mapping['AP'] = c
        if 'ambient humidity' in lc or lc == 'ah' or lc.startswith('ah'):
            mapping['AH'] = c
        if 'air filter' in lc or 'afdp' in lc:
            mapping['AFDP'] = c
        if 'gas turbine exhaust pressure' in lc or 'gtep' in lc:
            mapping['GTEP'] = c
        if 'turbine inlet temperature' in lc or 'tit' in lc:
            mapping['TIT'] = c
        if 'turbine after temperature' in lc or 'tat' in lc:
            mapping['TAT'] = c
        if 'turbine energy yield' in lc or 'tey' in lc:
            mapping['TEY'] = c
        if 'compressor discharge pressure' in lc or 'cdp' in lc:
            mapping['CDP'] = c
        if lc == 'co' or 'carbon monoxide' in lc:
            mapping['CO'] = c
        if 'nox' in lc:
            mapping['NOX'] = c
    # fallback matches if missing
    for key in ['AT','AP','AH','AFDP','GTEP','TIT','TAT','TEY','CDP','CO']:
        if key not in mapping:
            for lc, orig in lower.items():
                if key=='AT' and ('temperature' in lc and 'ambient' in lc):
                    mapping['AT'] = orig; break
                if key=='AP' and ('pressure' in lc and 'ambient' in lc):
                    mapping['AP'] = orig; break
                if key=='AH' and ('humidity' in lc and 'ambient' in lc):
                    mapping['AH'] = orig; break
                if key=='AFDP' and 'air filter' in lc:
                    mapping['AFDP'] = orig; break
                if key=='GTEP' and 'exhaust pressure' in lc:
                    mapping['GTEP'] = orig; break
                if key=='TIT' and 'turbine inlet' in lc:
                    mapping['TIT'] = orig; break
                if key=='TAT' and 'turbine after' in lc:
                    mapping['TAT'] = orig; break
                if key=='TEY' and 'energy yield' in lc:
                    mapping['TEY'] = orig; break
                if key=='CDP' and 'compressor discharge' in lc:
                    mapping['CDP'] = orig; break
                if key=='CO' and ('carbon monoxide' in lc or lc=='co'):
                    mapping['CO'] = orig; break
    return mapping

# -------------------------
# PyTorch model + dataset
# -------------------------
class TurbineDataset(torch.utils.data.Dataset):
    def __init__(self, X, y):
        self.X = X.astype('float32')
        self.y = y.astype('float32').reshape(-1,1)
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class MLP(nn.Module):
    def __init__(self, n_in, hidden):
        super().__init__()
        layers = []
        prev = n_in
        for h in hidden:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)
    def forward(self, x):
        return self.net(x)

# -------------------------
# training pipeline
# -------------------------
def prepare_training_data(csv_path):
    df = pd.read_csv(csv_path)
    mapping = detect_feature_columns(df)
    required = ['AT','AP','AH','AFDP','GTEP','TIT','TAT','TEY','CDP','CO']
    if not all(k in mapping for k in required):
        raise ValueError(f"Could not detect all required columns. Found mapping: {mapping}. CSV columns: {list(df.columns)}")
    feature_cols = [mapping[k] for k in ['AT','AP','AH','AFDP','GTEP','TIT','TAT','TEY','CDP']]
    X = df[feature_cols].values.astype(float)
    y = df[mapping['CO']].values.astype(float)
    # remove NaNs
    mask = ~np.isnan(X).any(axis=1) & ~np.isnan(y)
    X = X[mask]
    y = y[mask]
    # compute column means for input features (used later for filling unknown turbine-only features)
    col_means = np.mean(X, axis=0)
    feature_names = ['AT','AP','AH','AFDP','GTEP','TIT','TAT','TEY','CDP']
    return X, y, feature_names, col_means, mapping

def train_model(csv_path, epochs=DEFAULTS['epochs'], lr=DEFAULTS['lr'], hidden=DEFAULTS['hidden'], batch_size=DEFAULTS['batch_size'], test_size=DEFAULTS['test_size'], random_state=DEFAULTS['random_state']):
    print("Preparing training data from:", csv_path)
    X, y, feature_names, col_means, mapping = prepare_training_data(csv_path)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=test_size, random_state=random_state)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)

    train_ds = TurbineDataset(X_train_s, y_train)
    val_ds = TurbineDataset(X_val_s, y_val)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MLP(n_in=X.shape[1], hidden=hidden).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    best_val = float('inf')
    print("Training MLP (device=%s) ..." % device)
    for epoch in range(1, epochs+1):
        model.train()
        train_losses = []
        for xb, yb in train_loader:
            xb = xb.to(device); yb = yb.to(device)
            opt.zero_grad()
            out = model(xb)
            loss = loss_fn(out, yb)
            loss.backward()
            opt.step()
            train_losses.append(loss.item())
        # validation
        model.eval()
        val_losses = []
        y_trues = []
        y_preds = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device); yb = yb.to(device)
                out = model(xb)
                val_losses.append(loss_fn(out,yb).item())
                y_trues.append(yb.cpu().numpy())
                y_preds.append(out.cpu().numpy())
        mean_train = float(np.mean(train_losses)) if train_losses else 0.0
        mean_val = float(np.mean(val_losses)) if val_losses else 0.0
        if y_trues:
            y_true = np.vstack(y_trues)
            y_pred = np.vstack(y_preds)
            mse = mean_squared_error(y_true, y_pred)
            r2 = r2_score(y_true, y_pred)
        else:
            mse = None; r2 = None
        # build safe strings for optional metrics
        mse_str = f"{mse:.6f}" if mse is not None else "NA"
        r2_str = f"{r2:.4f}" if r2 is not None else "NA"

        if epoch % 10 == 0 or epoch == 1 or epoch == epochs:
            print(f"Epoch {epoch:03d}/{epochs}  train_loss={mean_train:.6f}  val_loss={mean_val:.6f}  val_mse={mse_str}  r2={r2_str}")
        # checkpoint
        if mean_val < best_val:
            best_val = mean_val
            torch.save(model.state_dict(), MODEL_FILE)
            joblib.dump(scaler, SCALER_FILE)
            meta = {"feature_names": feature_names, "col_means": col_means.tolist(), "mapping": mapping}
            with open(META_FILE, "w") as f:
                json.dump(meta, f, indent=2)
    print("Training finished. Best model saved to", MODEL_FILE)
    # load best model into cpu for later use
    model_cpu = MLP(n_in=X.shape[1], hidden=hidden)
    model_cpu.load_state_dict(torch.load(MODEL_FILE, map_location='cpu'))
    model_cpu.eval()
    scaler = joblib.load(SCALER_FILE)
    return model_cpu, scaler, feature_names, np.array(col_means)

# -------------------------
# environment API (Open-Meteo)
# -------------------------
def load_cache(fname=CACHE_FILE):
    if os.path.exists(fname):
        try:
            with open(fname, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            return {}
    return {}

def save_cache(cache, fname=CACHE_FILE):
    with open(fname, 'w', encoding='utf-8') as f:
        json.dump(cache, f, indent=2)

def round_coord(lat, lon, kround=DEFAULTS['kround']):
    return (round(float(lat), kround), round(float(lon), kround))

def query_open_meteo(lat, lon, cache, kround=DEFAULTS['kround'], sleep=DEFAULTS['sleep_api']):
    """
    Query Open-Meteo for current temperature (C), relative humidity (%), surface_pressure (hPa).
    Returns dict with keys: AT, AH, AP, elevation
    Caches results keyed by rounded lat,lon.
    """
    rlat, rlon = round_coord(lat, lon, kround)
    key = f"{rlat},{rlon}"
    if key in cache:
        return cache[key]
    # default fallback
    features = {"AT": 25.0, "AH": 50.0, "AP": 1013.25, "elevation": 0.0}
    params = {
        "latitude": lat, "longitude": lon,
        "current_weather": "true",
        "hourly": "relativehumidity_2m,surface_pressure",
        "timezone": "UTC"
    }
    try:
        resp = requests.get(OPEN_METEO_FORECAST, params=params, timeout=15)
        resp.raise_for_status()
        j = resp.json()
        cw = j.get("current_weather", {})
        if cw and "temperature" in cw:
            features["AT"] = float(cw["temperature"])
        # hourly arrays pick nearest hour
        hourly = j.get("hourly", {})
        times = hourly.get("time", [])
        now_utc = datetime.utcnow()
        hour_str = now_utc.strftime("%Y-%m-%dT%H:00")
        idx = 0
        if hour_str in times:
            idx = times.index(hour_str)
        else:
            idx = min(len(times)-1, 0) if len(times)>0 else 0
        rh = hourly.get("relativehumidity_2m", [])
        sp = hourly.get("surface_pressure", [])
        if len(rh) > idx:
            features["AH"] = float(rh[idx])
        if len(sp) > idx:
            features["AP"] = float(sp[idx])
    except Exception as e:
        print(f"⚠️ Open-Meteo error at {lat:.5f},{lon:.5f}: {e}")
    # elevation
    try:
        r2 = requests.get(OPEN_METEO_ELEV, params={"latitude": lat, "longitude": lon}, timeout=10)
        r2.raise_for_status()
        j2 = r2.json()
        if "elevation" in j2:
            features["elevation"] = float(j2["elevation"])
        elif "data" in j2 and isinstance(j2["data"], list) and len(j2["data"])>0 and "elevation" in j2["data"][0]:
            features["elevation"] = float(j2["data"][0]["elevation"])
    except Exception:
        pass
    # cache and return
    cache[key] = features
    time.sleep(sleep)
    return features

# -------------------------
# Reweighting pipeline
# -------------------------
def reweight_all_paths(graph_file=DEFAULTS['graph_file'], paths_dir=DEFAULTS['paths_dir'], out_dir=DEFAULTS['out_dir']):
    # load model/scaler/meta
    if not (os.path.exists(MODEL_FILE) and os.path.exists(SCALER_FILE) and os.path.exists(META_FILE)):
        raise FileNotFoundError("Model/scaler/meta not found. Run with --train first.")
    meta = json.load(open(META_FILE, 'r'))
    feature_names = meta['feature_names']  # ['AT','AP','AH','AFDP','GTEP','TIT','TAT','TEY','CDP']
    col_means = np.array(meta['col_means'], dtype=float)
    # load model
    model = MLP(n_in=len(feature_names), hidden=DEFAULTS['hidden'])
    model.load_state_dict(torch.load(MODEL_FILE, map_location='cpu'))
    model.eval()
    scaler = joblib.load(SCALER_FILE)
    # load graph
    if not os.path.exists(graph_file):
        raise FileNotFoundError(f"Graph file not found: {graph_file}")
    G = ox.load_graphml(graph_file)
    path_files = sorted([f for f in os.listdir(paths_dir) if f.endswith('.csv')])
    if len(path_files) == 0:
        raise FileNotFoundError(f"No path CSVs found in {paths_dir}")
    os.makedirs(out_dir, exist_ok=True)
    cache = load_cache()
    summary = []
    for pf in tqdm(path_files, desc="Reweighting paths"):
        df = pd.read_csv(os.path.join(paths_dir, pf))
        # ensure necessary columns
        for c in ['mid_lat','mid_lon','AT','AP','AH','pred_CO','edge_emission']:
            if c not in df.columns:
                df[c] = pd.NA
        for idx, row in df.iterrows():
            u = int(row['from']); v = int(row['to'])
            length_m = float(row.get('length_m', 0.0))
            length_km = length_m / 1000.0
            # compute midpoint coords
            if u in G.nodes and v in G.nodes:
                y1, x1 = G.nodes[u]['y'], G.nodes[u]['x']
                y2, x2 = G.nodes[v]['y'], G.nodes[v]['x']
                mid_lat = (y1 + y2) / 2.0
                mid_lon = (x1 + x2) / 2.0
                env = query_open_meteo(mid_lat, mid_lon, cache)
                df.at[idx, 'mid_lat'] = mid_lat
                df.at[idx, 'mid_lon'] = mid_lon
            else:
                # fallback: unknown node -> use default env
                env = {"AT":25.0, "AH":50.0, "AP":1013.25, "elevation":0.0}
            # Build full feature vector in order feature_names:
            feature_vec = []
            for i, fname in enumerate(feature_names):
                if fname in ['AT','AP','AH']:
                    feature_vec.append(env.get(fname, col_means[i]))
                else:
                    feature_vec.append(float(col_means[i]))
            Xrow = np.array(feature_vec, dtype=float).reshape(1, -1)
            Xs = scaler.transform(Xrow)
            with torch.no_grad():
                inp = torch.from_numpy(Xs.astype('float32'))
                pred = model(inp).cpu().numpy().squeeze().item()
            pred_co = float(pred)
            edge_em = pred_co * (length_km)
            # write back
            df.at[idx, 'AT'] = feature_vec[0]
            df.at[idx, 'AP'] = feature_vec[1]
            df.at[idx, 'AH'] = feature_vec[2]
            df.at[idx, 'pred_CO'] = pred_co
            df.at[idx, 'edge_emission'] = edge_em
        total_em = df['edge_emission'].sum()
        total_len_km = df['length_m'].sum() / 1000.0
        out_csv = os.path.join(out_dir, pf)
        df.to_csv(out_csv, index=False)
        summary.append({'path_file': pf, 'length_km': total_len_km, 'total_emission': total_em})
    # save summary & cache
    summary_df = pd.DataFrame(summary).sort_values('total_emission')
    summary_df.to_csv(SUMMARY_CSV, index=False)
    save_cache(cache)
    print("Saved weighted CSVs ->", out_dir)
    print("Saved summary ->", SUMMARY_CSV)
    return summary_df, path_files, G

# -------------------------
# Generate best_path.html
# -------------------------
def generate_best_path_html(G, weighted_dir, path_files, best_index, out_html=BEST_HTML):
    # compute center
    first_csv = pd.read_csv(os.path.join(weighted_dir, path_files[0]))
    try:
        first_node = int(first_csv["from"].iloc[0])
        center_lat, center_lon = G.nodes[first_node]['y'], G.nodes[first_node]['x']
    except Exception:
        nodes = list(G.nodes)
        n0 = nodes[0]
        center_lat, center_lon = G.nodes[n0]['y'], G.nodes[n0]['x']

    colors = {'best':'#00b894','other':'#888888'}

    html = f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8" />
  <title>Best Path (Lowest Emission)</title>
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <link rel="stylesheet" href="https://unpkg.com/leaflet@1.7.1/dist/leaflet.css" />
  <script src="https://unpkg.com/leaflet@1.7.1/dist/leaflet.js"></script>
  <style>body{{margin:0}}#map{{width:100%;height:100vh}}#info{{position:fixed;top:10px;right:10px;z-index:1000;background:white;padding:12px;border-radius:8px;box-shadow:0 2px 8px rgba(0,0,0,0.2);max-width:360px}}</style>
</head>
<body>
  <div id="map"></div>
  <div id="info"><h3>Best Path</h3><div id="bestInfo">Loading...</div></div>
  <script>
    var map = L.map('map').setView([{center_lat}, {center_lon}], 13);
    L.tileLayer('https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png', {{
        maxZoom: 19,
        attribution: '© OpenStreetMap'
    }}).addTo(map);
"""

    # ADD ALL OTHER PATHS
    for i, pf in enumerate(path_files):
        df = pd.read_csv(os.path.join(weighted_dir, pf))
        coords = []
        for _, r in df.iterrows():
            node_from = int(r["from"])
            node_to   = int(r["to"])
            lat1, lon1 = G.nodes[node_from]['y'], G.nodes[node_from]['x']
            lat2, lon2 = G.nodes[node_to]['y'], G.nodes[node_to]['x']
            coords.append([lat1, lon1])
            coords.append([lat2, lon2])

        col = colors['best'] if i == best_index else colors['other']
        html += f"""
    L.polyline({coords}, {{color: '{col}', weight: 5, opacity: 0.7}}).addTo(map);
"""

    html += """
  </script>
</body>
</html>
"""

    with open(out_html, "w") as f:
        f.write(html)

    print("Best path HTML saved:", out_html)

# -------------------------
# CLI / Auto-run main
# -------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true", help="Train model on gt_2011.csv")
    parser.add_argument("--predict", action="store_true", help="Predict & reweight paths")
    parser.add_argument("--csv", default=None, help="Path to gt CSV; if not specified tries to find gt_2011.csv")
    parser.add_argument("--graph", default=DEFAULTS['graph_file'], help="GraphML file (delhi_noida.graphml)")
    parser.add_argument("--paths_dir", default=DEFAULTS['paths_dir'], help="Directory containing path CSVs (allkpaths/)")
    parser.add_argument("--out_dir", default=DEFAULTS['out_dir'], help="Output directory for weighted CSVs")
    args = parser.parse_args()

    # Auto behavior when no flags provided:
    auto = not (args.train or args.predict)
    # If auto: train if model missing, otherwise predict
    do_train = args.train
    do_predict = args.predict
    if auto:
        if os.path.exists(MODEL_FILE) and os.path.exists(SCALER_FILE) and os.path.exists(META_FILE):
            print("Model exists -> skipping training and running prediction (auto mode).")
            do_predict = True
            do_train = False
        else:
            print("No trained model found -> will train then predict (auto mode).")
            do_train = True
            do_predict = True

    if do_train:
        csv = args.csv or find_gt_csv()
        if csv is None:
            raise FileNotFoundError("gt_2011.csv not found in current dir. Place the file here or pass --csv path")
        train_model(csv_path=csv, epochs=DEFAULTS['epochs'], lr=DEFAULTS['lr'], hidden=DEFAULTS['hidden'], batch_size=DEFAULTS['batch_size'])

    if do_predict:
        summary_df, path_files, G = reweight_all_paths(graph_file=args.graph, paths_dir=args.paths_dir, out_dir=args.out_dir)
        # determine best path
        sorted_df = summary_df.sort_values('total_emission').reset_index(drop=True)
        best_file = sorted_df.loc[0, 'path_file']
        best_index = path_files.index(best_file)
        generate_best_path_html(G, args.out_dir, path_files, best_index, out_html=BEST_HTML)
        print("\nTop 5 paths:\n", sorted_df.head())

if __name__ == "__main__":
    main()
