import os
import multiprocessing
import time
from functools import partial
from statistics import mean

import pandas as pd
import geopandas as gpd
import tqdm

from shapely.geometry import Point, Polygon
from pympler import asizeof

from .routers import single_source_time_dependent_dijkstra, single_source_time_dependent_dijkstra_hashed
from .other import estimate_ram, bytes_to_readable


def calculate_OD_matrix(graph, stops, departure_time, output_path):
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
                results.append(
                    {
                    'source_stop': source_stop,
                    'destination_stop': dest_stop,
                    'arrival_time': arrival_times[dest_stop],
                    'travel_time': travel_times.get(dest_stop, None)  # Use .get() to avoid KeyError if the key is not found
                })

    # Конвертация списка в датафрейм и в файл csv
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_path, index=False)

def _calculate_OD_worker(source_stop, stops_list, graph, departure_time, hash_table=None):
    """
    Internal worker function to calculate the OD matrix for a single source stop.
    """
    
    if hash_table:
        arrival_times, _, travel_times = single_source_time_dependent_dijkstra_hashed(
            graph, source_stop, departure_time, hash_table
            )
    else:
        arrival_times, _, travel_times = single_source_time_dependent_dijkstra(
            graph, source_stop, departure_time
            )
        
    return [{
        'source_stop': source_stop,
        'destination_stop': dest_stop,
        'arrival_time': arrival_times[dest_stop],
        'travel_time': travel_times.get(dest_stop, None)
    } for dest_stop in stops_list if dest_stop in arrival_times]

def calculate_OD_matrix_parallel(graph, stops, departure_time, output_path, num_processes=2, hash_table=None):
    """
    Calculates the Origin-Destination (OD) matrix for a given graph, 
    stops, and departure time using parallel processing.
    
    Parameters:
    -----------
    graph (networkx.Graph): The graph representing the transit network.
    stops (pandas.DataFrame): A DataFrame containing the stops information.
    departure_time (int): The departure time in seconds since midnight.
    num_processes (int): Number of parallel processes to use for computation.
        Strongly reccomended to use number of processes equal 
        to the number of physical CPU cores or less.
    
    Returns:
    ----------
    pandas.DataFrame: A DataFrame containing the OD matrix.
    """
    print(f'Выполняется расчет с использованием {num_processes} процессов')
    
    # Предварительная попытка оценить достаточность
    # доступной памяти для выполнения расчетов
    graph_size = asizeof.asizeof(graph)
    ram, free_ram = estimate_ram()
    
    # Эмпирическая формула :)
    expected_ram = graph_size * 5 + num_processes * graph_size * 2.5
    
    if expected_ram > free_ram:
        raise MemoryError(f'Размер графа {bytes_to_readable(graph_size)}, '
                          f'ожидаемые затраты {expected_ram} превышают доступную '
                          f'память {bytes_to_readable(free_ram)}')
    else:
        print(f'Размер графа {bytes_to_readable(graph_size)}, Ожидаемые затраты '
              f'{bytes_to_readable(expected_ram)}, доступная память '
              f'{bytes_to_readable(free_ram)}')

    stops_list = stops['stop_id'].tolist() #Список остановок
    results = []
    
    time_start = time.time()

    with multiprocessing.Pool(processes=num_processes) as pool:
        # Фиксаця аргументов функции calculate_OD_worker
        partial_worker = partial(_calculate_OD_worker, 
                                 stops_list=stops_list, 
                                 graph=graph, 
                                 departure_time=departure_time,
                                 hash_table=hash_table)
        results = pool.map(partial_worker, stops_list)

    # Группировка результатов в один список
    results = [item for sublist in results 
               for item in sublist]

    results_df = pd.DataFrame(results)

    # Запись результатов в файл csv
    print("Выполняется запись файла csv")
    results_df.to_csv(output_path, index=False)
    
    print(f"Time elapsed: {time.time() - time_start}")
    
    return results_df

def create_service_area(graph, source, start_time, cutoff, buffer_radius):
    """
    Creates a service area by buffering around all points within a travel time cutoff.

    Args:
        graph (networkx.DiGraph): The graph to search.
        source: The graph node to start the search from.
        start_time (float): The time to start the search from.
        cutoff (float): The travel time cutoff for including nodes in the service area.
        buffer_radius (float): The radius in meters for buffering around each point.

    Returns:
        tuple: A tuple containing two GeoDataFrames:
            - The first GeoDataFrame has a single geometry column containing the merged buffer polygon.
            - The second GeoDataFrame contains the points within the cutoff.
    """
    _, _, travel_times = single_source_time_dependent_dijkstra(graph, source, start_time)

    # Filter nodes that are reachable within the cutoff
    points_data = [{'node': node, 
                    'geometry': Point(
                        graph.nodes[node]['x'], 
                        graph.nodes[node]['y']
                        ), 
                    'travel_time': time}
                   for node, time in travel_times.items() 
                   if time <= cutoff 
                   and 'x' in graph.nodes[node] 
                   and 'y' in graph.nodes[node]]

    # GeoDataFrame containing nodes reachable within the cutoff
    points_gdf = gpd.GeoDataFrame(points_data, geometry='geometry', 
                                  crs="EPSG:4326")

    # Nodes buffered and merged into a single polygon
    # Reprojection to World Equidistant Cylindrical (EPSG:4087) for buffering in meters
    buffer_gdf = points_gdf.to_crs("EPSG:4087")
    buffer_gdf['geometry'] = buffer_gdf.buffer(buffer_radius)
    service_area_polygon = buffer_gdf.unary_union

    # Создание GeoDataFrame из полигона (Shapely Polygon) в EPSG:4326
    service_area_gdf = gpd.GeoDataFrame({'geometry': [service_area_polygon]}, 
                                        crs="EPSG:4087").to_crs("EPSG:4326")

    return service_area_gdf, points_gdf

def create_grid(geodataframe, cell_size):
    """
    Creates a grid within the bounding box of a GeoDataFrame.

    Args:
    --------
        geodataframe: GeoDataFrame containing the geometry to be gridded
        cell_size (float): size of the grid cells in the meters

    Returns:
    --------
        gpd.GeoDataFrame: Polygon grid
    """
    geodataframe = geodataframe.to_crs("EPSG:4087")
    xmin, ymin, xmax, ymax = geodataframe.total_bounds
    rows = int((ymax - ymin) / cell_size)
    cols = int((xmax - xmin) / cell_size)
    grid = []

    for i in range(cols):
        for j in range(rows):
            x1 = xmin + i * cell_size
            y1 = ymin + j * cell_size
            x2 = x1 + cell_size
            y2 = y1 + cell_size
            grid.append(Polygon([(x1, y1), (x2, y1), 
                                 (x2, y2), (x1, y2)])
                        )

    grid_geodataframe = gpd.GeoDataFrame(grid, 
                                         columns=['geometry'], 
                                         crs=geodataframe.crs
                                         ).to_crs("EPSG:4326")
    
    return grid_geodataframe

def calculate_edge_frequency(graph, start_time, end_time):
    """
    Calculates the frequency of edges in a graph 
    based on the schedules between start_time and end_time.
    
    Args:
        graph (networkx.Graph): The graph containing the edges and schedules.
        start_time (int): The start time in seconds from midnight.
        end_time (int): The end time in seconds from midnight.
    
    Returns:
        None
    """
    
    for edge in graph.edges(data = True):
        if 'schedules' in edge[2]:
            
            trips = edge[2]['sorted_schedules']
            seq = [(trips[i+1][0] - trips[i][0]) 
                   for i in range(len(trips)-1)
                   if trips[i][0] >= start_time and trips[i][0] <= end_time
                ] # list containing the headways between consecutive trips along the edge

            if len(seq) > 0:
                frequency = mean(seq)
            else:
                frequency = None
                
            edge[2]['frequency'] = frequency # mean vehicle headway in seconds along the edge
            
def calculate_node_frequency(graph, start_time, end_time):
    """
    Calculates the frequency of departures at nodes in a graph 
    based on the schedules of adjacent edges between start_time and end_time.
    
    Args:
        graph (networkx.Graph): The graph containing the nodes and adjacent edges with schedules.
        start_time (int): The start time in seconds from midnight.
        end_time (int): The end time in seconds from midnight.
    
    Returns:
        None
    """
    
    for node_view in graph.nodes(data = True):
        node = node_view[0]
        all_times = []
        
        if node_view[1]['type'] == 'transit':

            # Iterate through all edges adjacent to the current node
            for edge in graph.edges(node, data=True):
                if 'schedules' in edge[2]:

                    for schedule in edge[2]['schedules']:
                        departure_time = schedule[0]
                        if start_time <= departure_time <= end_time:
                            all_times.append(departure_time)
            
            all_times.sort()
            # Calculate the headways between consecutive departures (or arrivals ?)
            headways = [(all_times[i+1] - all_times[i]) for i in range(len(all_times)-1)]

            if len(headways) > 0:
                frequency = mean(headways)
            else:
                frequency = None

            graph.nodes[node]['frequency'] = frequency

def validate_feed(gtfs_path: str) -> bool:
    """
    Validates the GTFS feed located at the specified path.

    Args:
    - gtfs_path (str): Path to the GTFS dataset directory.

    Returns:
    - bool: True if the GTFS feed is valid, False otherwise.
    """

    # List of required GTFS files
    required_files = [
        "agency.txt", "stops.txt", "routes.txt",
        "trips.txt", "stop_times.txt", "calendar.txt"
    ]

    # Check for the existence of required GTFS files
    for file in required_files:
        if not os.path.isfile(os.path.join(gtfs_path, file)):
            print(f"Missing required file: {file}")
            return False

    try:
        # Load GTFS files
        agency_df = pd.read_csv(os.path.join(gtfs_path, "agency.txt"))
        stops_df = pd.read_csv(os.path.join(gtfs_path, "stops.txt"))
        routes_df = pd.read_csv(os.path.join(gtfs_path, "routes.txt"))
        trips_df = pd.read_csv(os.path.join(gtfs_path, "trips.txt"))
        stop_times_df = pd.read_csv(os.path.join(gtfs_path, "stop_times.txt"), low_memory=False)
        calendar_df = pd.read_csv(os.path.join(gtfs_path, "calendar.txt"))
        
        critical_erorrs = False

        # Validate agency.txt
        if agency_df.empty or 'agency_id' not in agency_df.columns:
            print("agency.txt is invalid or missing required 'agency_id' column.")

        # Validate stops.txt
        if stops_df.empty or 'stop_id' not in stops_df.columns:
            print("stops.txt is invalid or missing required 'stop_id' column.")
            critical_erorrs = True

        # Validate routes.txt
        if routes_df.empty or 'route_id' not in routes_df.columns or 'route_id' not in routes_df.columns:
            print("routes.txt is invalid or missing required columns (agency_id, route_id).")
            critical_erorrs = True
            
        if not set(routes_df['agency_id']).issubset(set(agency_df['agency_id'])):
            print("Mismatch in agency IDs between routes and agency files.")
            critical_erorrs = True
            
        # Validate trips.txt
        if trips_df.empty or 'trip_id' not in trips_df.columns or 'route_id' not in trips_df.columns:
            print("trips.txt is invalid or missing required columns.")
            critical_erorrs = True

        if not set(trips_df['route_id']).issubset(set(routes_df['route_id'])):
            print("Mismatch in route IDs between trips and routes files.")
            critical_erorrs = True
            
        # Validate stop_times.txt
        if stop_times_df.empty or 'trip_id' not in stop_times_df.columns or 'stop_id' not in stop_times_df.columns:
            print("stop_times.txt is invalid or missing required columns.")
            critical_erorrs = True

        if not set(stop_times_df['trip_id']).issubset(set(trips_df['trip_id'])):
            print("Mismatch in trip IDs between stop_times and trips files.")

        if not set(stop_times_df['stop_id']).issubset(set(stops_df['stop_id'])):
            print("Mismatch in stop IDs between stop_times and stops files.")

        # Validate calendar.txt
        if calendar_df.empty:
            print("calendar.txt is invalid or empty.")

        # Validate stop_times.txt for blank times and format of times
        if 'departure_time' not in stop_times_df.columns or 'arrival_time' not in stop_times_df.columns:
            print("stop_times.txt is missing required time columns.")

        # Check for blank times
        if stop_times_df['departure_time'].isnull().any() or stop_times_df['arrival_time'].isnull().any():
            print("Blank departure or arrival times found in stop_times.txt.")

        # Validate time format (HH:MM:SS)
        time_format_regex = r'^(\d{2}):([0-5]\d):([0-5]\d)$' #check for HH:MM:SS format
        invalid_departure_times = stop_times_df[~stop_times_df['departure_time'].str.match(time_format_regex)]
        invalid_arrival_times = stop_times_df[~stop_times_df['arrival_time'].str.match(time_format_regex)]

        if not invalid_departure_times.empty or not invalid_arrival_times.empty:
            print("Invalid time format found in departure or arrival times in stop_times.txt.")
        
        # Additional format and consistency checks= will be added
     
    except Exception as e:
        print(f"Error during validation: {e}")
        return False

    if critical_erorrs:
        ("GTFS feed contains critical errors.")
        return False
    else:
        print("GTFS feed is valid.")
        return True

def unpack_path_vertices(path):
    """
    This function separates pedestrian segments 
    of given path into list of lists
    """
    
    pedestrian_path = []
    current_sublist = []
    
    for vertex in path:
        if isinstance(vertex, int):
            current_sublist.append(vertex)
            
        elif current_sublist:
            pedestrian_path.append(current_sublist)
            current_sublist = []
    
    if current_sublist:
        pedestrian_path.append(current_sublist)
        
    return pedestrian_path

def calculate_pedestrian_time(pedestrian_path, graph):
    """
    Calculate total impedance (travel time) for pedestrian paths by summing up the edge weights.

    """
    impedance = 0
    for subpath in pedestrian_path:
        for i in range(len(subpath) - 1):
            start_node = subpath[i]
            end_node = subpath[i+1]
            
            impedance += graph[start_node][end_node]['weight']
            
    return impedance

def reconstruct_path(target, predecessors):
    """
    Reconstruct path from predecessors dictionary
    """
    
    path = []
    current_node = target

    while current_node is not None:
        path.insert(0, current_node)
        
        current_node = predecessors.get(current_node)
        
    return path

def separate_travel_times(graph, predecessors: dict, travel_times: dict, source) -> pd.DataFrame:
    """
    Separate the travel times into transit time and pedestrian time for each node in the graph.

    Args:
        graph (networkx.DiGraph): The graph representing the transit network.
        predecessors (dict): A dictionary containing the predecessors of each node in the graph.
        travel_times (dict): A dictionary containing the travel times for each node in the graph.
        source: The source node from which to calculate the travel times.

    Returns:
        pandas.DataFrame: A DataFrame containing the transit time and pedestrian time for each node.
    """
    
    results = []
    
    for node in tqdm.tqdm(graph.nodes(data = True)):
        
        if node[0] != source:

            path = reconstruct_path(node[0], predecessors)
            pedestrian_path = unpack_path_vertices(path)
            pedestrian_time = calculate_pedestrian_time(pedestrian_path, graph)
            
            transit_time = travel_times[node[0]] - pedestrian_time

            results.append({'node': node[0], 'transit_time': transit_time, 'pedestrian_time': pedestrian_time})
        
    results = pd.DataFrame(results)
    return results

def process_graph_to_hash_table(graph):
    """
    Process a graph and convert it into a hash table
    mapping edges to their sorted schedules or static weights.

    Args:
        graph (networkx.Graph): The input graph.

    Returns:
        dict: A hash table representing the processed graph.
    """
    schedules_hash = {}
    for from_node, to_node, data in graph.edges(data=True):
        if 'sorted_schedules' in data:
            schedules_hash[(from_node, to_node)] = data['sorted_schedules']
        else:
            # Static weight wrapped in a list of tuples to make it iterable
            static_weight = data['weight']
            schedules_hash[(from_node, to_node)] = [(static_weight,)]  # comma is to make it a tuple
            
    return schedules_hash