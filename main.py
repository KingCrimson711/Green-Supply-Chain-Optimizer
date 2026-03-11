import os
import osmnx as ox
import networkx as nx
import pandas as pd
from tqdm import tqdm
import random

ox.settings.log_console = True
ox.settings.use_cache = True

def download_or_load_graph(place_name, cache_file):
    if os.path.exists(cache_file):
        print(f"♻️ Loading cached graph: {cache_file}")
        G = ox.load_graphml(cache_file)
    else:
        print(f"🌍 Downloading graph for: {place_name}")
        G = ox.graph_from_place(place_name, network_type="drive")
        ox.save_graphml(G, cache_file)
    return G

def get_nearest_node(G, location):
    lat, lon = ox.geocode(location)
    return ox.distance.nearest_nodes(G, lon, lat)

def compute_diverse_paths(G, source, target, k=50, output_dir="allkpaths", diversity_factor=0.3):
    os.makedirs(output_dir, exist_ok=True)
    print(f"🔎 Computing {k} diverse paths...")
    
    # Create a copy of the graph for manipulation
    G_modified = G.copy()
    
    paths = []
    path_lengths = []
    
    for i in range(k):
        try:
            # Find the shortest path in the modified graph
            path = nx.shortest_path(G_modified, source, target, weight='length')
            paths.append(path)
            
            # Calculate the actual length from the original graph
            total_length = 0
            for u, v in zip(path[:-1], path[1:]):
                if G.has_edge(u, v):
                    edge_data = min(G[u][v].values(), key=lambda x: x.get('length', float('inf')))
                    total_length += edge_data.get('length', 0)
            
            path_lengths.append(total_length)
            print(f"Path {i+1} length: {total_length/1000:.2f} km")
            
            # Penalize edges in this path to encourage diversity
            for u, v in zip(path[:-1], path[1:]):
                if G_modified.has_edge(u, v):
                    # Increase the length of used edges to discourage reuse
                    for key in list(G_modified[u][v].keys()):
                        G_modified[u][v][key]['length'] *= (1 + diversity_factor)
            
            # Occasionally add some randomness to discover completely different routes
            if i % 5 == 0:
                # Randomly modify some edge weights to explore different areas
                edges = list(G_modified.edges(keys=True))
                for _ in range(100):  # Modify 100 random edges
                    u, v, k = random.choice(edges)
                    G_modified[u][v][k]['length'] *= random.uniform(0.8, 1.2)
                    
        except nx.NetworkXNoPath:
            print(f"No more paths found after {i} iterations")
            break
    
    # Now save all the found paths
    for i, path in enumerate(tqdm(paths, desc="Saving paths")):
        edges = []
        total_length = 0
        
        for u, v in zip(path[:-1], path[1:]):
            if G.has_edge(u, v):
                edge_data = min(G[u][v].values(), key=lambda x: x.get('length', float('inf')))
                edge_length = edge_data.get('length', 0)
                edges.append({
                    "from": u,
                    "to": v,
                    "length_m": edge_length
                })
                total_length += edge_length
        
        df = pd.DataFrame(edges)
        df.to_csv(f"{output_dir}/path{i+1}.csv", index=False)

    print(f"✅ Done! Saved {len(paths)} diverse paths in '{output_dir}/'")
    print(f"Path length range: {min(path_lengths)/1000:.2f} to {max(path_lengths)/1000:.2f} km")

def main():
    source = "Delhi, India"
    target = "Noida, India"
    GRAPH_CACHE = "delhi_noida.graphml"

    G = download_or_load_graph([source, target], GRAPH_CACHE)

    print("📍 Finding nearest nodes...")
    source_node = get_nearest_node(G, source)
    target_node = get_nearest_node(G, target)
    
    print(f"Source node: {source_node}, Target node: {target_node}")

    compute_diverse_paths(G, source_node, target_node, k=50)

if __name__ == "__main__":
    main()