import os
import pandas as pd
import osmnx as ox

def create_interactive_path_selector():
    # Load your data
    G = ox.load_graphml("delhi_noida.graphml")
    path_files = sorted([f for f in os.listdir("allkpaths") if f.endswith(".csv")])
    
    # Get center point for map
    df_first = pd.read_csv(os.path.join("allkpaths", path_files[0]))
    first_node = df_first["from"].iloc[0]
    center_lat, center_lon = G.nodes[first_node]["y"], G.nodes[first_node]["x"]
    
    # Generate HTML with interactive selector
    html_content = generate_html(G, path_files, center_lat, center_lon)
    
    # Save as standalone HTML file
    with open("path_selector.html", "w", encoding="utf-8") as f:
        f.write(html_content)
    
    print("✅ Interactive map created: 'path_selector.html'")
    print("📁 Open this file in ANY web browser (Chrome, Firefox, etc.)")
    print("🎯 Use dropdown or click paths to switch between them!")

def generate_html(G, path_files, center_lat, center_lon):
    colors = ['#ff6b6b', '#4ecdc4', '#45b7d1', '#f9ca24', '#6c5ce7', 
              '#e84393', '#00b894', '#fd79a8', '#fdcb6e', '#00cec9']
    
    html = f"""<!DOCTYPE html>
<html>
<head>
    <title>Interactive Path Selector</title>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <link rel="stylesheet" href="https://unpkg.com/leaflet@1.7.1/dist/leaflet.css" />
    <script src="https://unpkg.com/leaflet@1.7.1/dist/leaflet.js"></script>
    <style>
        body {{ margin: 0; padding: 0; }}
        #map {{ width: 100%; height: 100vh; }}
        #controls {{
            position: fixed; top: 10px; right: 10px; z-index: 1000;
            background: white; padding: 15px; border-radius: 8px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.3); max-width: 300px;
        }}
        select, button {{ 
            width: 100%; padding: 8px; margin: 5px 0; 
            border: 2px solid #007bff; border-radius: 4px; 
        }}
        .info-box {{ 
            margin-top: 10px; padding: 10px; background: #f8f9fa; 
            border-radius: 4px; border-left: 4px solid #007bff; 
        }}
    </style>
</head>
<body>
    <div id="map"></div>
    
    <div id="controls">
        <h3>🟢 Select Path</h3>
        <select id="pathSelector" onchange="selectPath(this.value)">
            <option value="">Choose a path...</option>
"""
    
    # Add dropdown options
    for i, file in enumerate(path_files):
        df = pd.read_csv(os.path.join("allkpaths", file))
        length_km = df["length_m"].sum() / 1000
        html += f'<option value="{i}">Path {i+1}: {length_km:.1f} km</option>\n'
    
    html += f"""
        </select>
        <div class="info-box" id="pathInfo">
            Select a path to see details
        </div>
        <button onclick="resetView()">🗺️ Reset View</button>
    </div>

    <script>
        var map = L.map('map').setView([{center_lat}, {center_lon}], 11);
        L.tileLayer('https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png', {{
            attribution: '© OpenStreetMap contributors'
        }}).addTo(map);
        
        var paths = [];
"""
    
    # Add each path to the JavaScript
    for i, file in enumerate(path_files):
        df = pd.read_csv(os.path.join("allkpaths", file))
        coords = []
        
        for _, row in df.iterrows():
            if row["from"] in G.nodes:
                node = G.nodes[row["from"]]
                coords.append([node["y"], node["x"]])
            if _ == len(df) - 1 and row["to"] in G.nodes:
                node = G.nodes[row["to"]]
                coords.append([node["y"], node["x"]])
        
        length_km = df["length_m"].sum() / 1000
        color = colors[i % len(colors)]
        
        html += f"""
        paths[{i}] = L.polyline({coords}, {{
            color: '{color}', weight: 3, opacity: 0.4
        }}).addTo(map);
        paths[{i}].bindPopup('Path {i+1}<br>{length_km:.1f} km');
        paths[{i}].on('click', function() {{ 
            document.getElementById('pathSelector').value = {i};
            selectPath({i}); 
        }});
"""
    
    html += """
        function selectPath(index) {
            // Reset all paths
            paths.forEach(p => p.setStyle({weight: 3, opacity: 0.4}));
            
            if (index === "") return;
            
            // Highlight selected path
            paths[index].setStyle({weight: 6, opacity: 0.9});
            paths[index].bringToFront();
            
            // Update info
            var select = document.getElementById('pathSelector');
            document.getElementById('pathInfo').innerHTML = 
                '<strong>' + select.options[select.selectedIndex].text + '</strong>' +
                '<br>Click path for details';
            
            // Center map
            map.fitBounds(paths[index].getBounds());
        }
        
        function resetView() {
            map.setView([""" + str(center_lat) + """, """ + str(center_lon) + """], 11);
            document.getElementById('pathSelector').value = "";
            paths.forEach(p => p.setStyle({weight: 3, opacity: 0.1}));
            document.getElementById('pathInfo').innerHTML = "Select a path to see details";
        }
    </script>
</body>
</html>
"""
    
    return html

if __name__ == "__main__":
    create_interactive_path_selector()