import webbrowser
import tempfile

import pandas as pd
import osmnx as ox
import folium

def plot_path_browser(graph, stops: pd.DataFrame):
    """
    Plots the given graph and stops on a folium map and opens it in a web browser.
    
    Parameters:
    - graph: The graph to plot.
    - stops: The stops to plot.
    """
    
    G_nodes, G_edges = ox.graph_to_gdfs(graph)
    
    frame_center_lat = stops['stop_lat'].mean()
    frame_center_lon = stops['stop_lon'].mean()
    
    m = folium.Map(location=[frame_center_lat, frame_center_lon], 
                zoom_start=11,
                width='100%', 
                height='100%')
    
    G_edges.explore(m = m)  
    G_nodes.explore(m = m, column = 'type', legend = True)
    
    with tempfile.NamedTemporaryFile(mode='w', delete_on_close = True, 
                                     delete = False, prefix = 'path', 
                                     suffix='.html') as outfp:
        m.save(outfp.name)
        webbrowser.open(outfp.name)
