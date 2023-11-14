import pandas as pd

import geopandas as gpd
from shapely.geometry import Point

from routers import single_source_time_dependent_dijkstra

# Функция для расчета матрицы источник-назначение между всеми остановками (БЕТА)
def calculate_OD_matrix(graph, stops, departure_time):
    """
    Calculates the Origin-Destination (OD) matrix for a given graph, stops, and departure time.
    
    Parameters:
    graph (networkx.Graph): The graph representing the transit network.
    stops (pandas.DataFrame): A DataFrame containing the stops information.
    departure_time (int): The departure time in seconds since midnight.
    
    Returns:
    pandas.DataFrame: A DataFrame containing the OD matrix with the following columns:
        - source_stop: The ID of the origin stop.
        - destination_stop: The ID of the destination stop.
        - arrival_time: The arrival time at the destination stop in seconds since midnight.
        - travel_time: The travel time from the origin stop to the destination stop in seconds.
    """
    
    stops_list = stops['stop_id'].tolist() #Список остановок
    results = []

    for source_stop in stops_list:
        # Вычисление времени прибытия и предшественников для каждой остановки
        arrival_times, _, travel_times = single_source_time_dependent_dijkstra(graph, source_stop, departure_time)
        
        # Итерация по всем остановкам, для их отбора в результатах работы алгоритма дейкстры
        for dest_stop in stops_list:
            if dest_stop in arrival_times:
                # Добавление результатов в список
                results.append({
                    'source_stop': source_stop,
                    'destination_stop': dest_stop,
                    'arrival_time': arrival_times[dest_stop],
                    'travel_time': travel_times.get(dest_stop, None)  # Use .get() to avoid KeyError if the key is not found
                })

    # Конвертация списка в датафрейм и в файл csv
    results_df = pd.DataFrame(results)
    results_df.to_csv(r"D:\Python_progs\Output\results2.csv", index=False)

def create_service_area(graph, source, start_time, cutoff, buffer_radius):
    """
    Creates a service area by buffering around all points within a travel time cutoff.

    Args:
        graph (networkx.DiGraph): The graph to search.
        source (hashable): The graph node to start the search from.
        start_time (float): The time to start the search from.
        cutoff (float): The travel time cutoff for including nodes in the service area.
        buffer_radius (float): The radius in meters for buffering around each point.

    Returns:
        tuple: A tuple containing two GeoDataFrames:
            - The first GeoDataFrame has a single geometry column containing the merged buffer polygon.
            - The second GeoDataFrame contains the points within the cutoff.
    """
    _, _, travel_times = single_source_time_dependent_dijkstra(graph, source, start_time)

    # Filter nodes by cutoff time and create Point geometries
    points_data = [{'node': node, 
                    'geometry': Point(graph.nodes[node]['x'], 
                                    graph.nodes[node]['y']
                                    ), 
                    'travel_time': time}
                   for node, time in travel_times.items() 
                   if time <= cutoff and 'x' in graph.nodes[node] 
                   and 'y' in graph.nodes[node]]

    # Create a GeoDataFrame with the points and travel times
    points_gdf = gpd.GeoDataFrame(points_data, geometry='geometry', 
                                  crs="EPSG:4326")

    # Buffer around each point and dissolve to merge them into a single polygon
    buffer_gdf = points_gdf.to_crs("EPSG:4087")
    buffer_gdf['geometry'] = buffer_gdf.buffer(buffer_radius)
    service_area_polygon = buffer_gdf.unary_union

    # Create a GeoDataFrame with the service area polygon
    service_area_gdf = gpd.GeoDataFrame({'geometry': [service_area_polygon]}, 
                                        crs="EPSG:4087").to_crs("EPSG:4326")

    return service_area_gdf, points_gdf