import webbrowser
import tempfile
import itertools

import pandas as pd
import osmnx as ox
import folium

from converters import parse_time_to_seconds

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

#Требует обновления
def calculate_OD_matrix(G, departure_time_input, routing_alghoritm):
    """
    Calculates the shortest path for all OD pairs in a graph using a time-dependent Dijkstra function.

    Args:
    - G: The networkx graph representing the GTFS data.
    - departure_time_input: The departure time in the format 'HH:MM:SS'.
    - dijkstra_func: The time-dependent Dijkstra function to use for path finding.

    Returns:
    - all_paths_df: DataFrame with columns ['Source', 'Target', 'Path', 'Arrival Time', 'Travel Time'].
    """
    nodes = list(G.nodes)
    all_paths = []

    try:
        demand_time = parse_time_to_seconds(departure_time_input)
    except Exception:
        raise ValueError("Error parsing departure time.")
    
    # Перебор всех пар узлов
    for source, target in itertools.product(nodes, repeat=2):
        if source != target:
            try:
                path, arrival_time, travel_time = routing_alghoritm(G, 
                                                                    source=source, 
                                                                    target=target, 
                                                                    start_time=demand_time)
                if path:
                    all_paths.append({
                        'Source': source,
                        'Target': target,
                        'Path': path,
                        'Arrival Time': arrival_time,
                        'Travel Time': travel_time
                    })
            except Exception:
                # Если возникла ошибка, пропустить текущую пару узлов
                continue

    all_paths_df = pd.DataFrame(all_paths)

    return all_paths_df