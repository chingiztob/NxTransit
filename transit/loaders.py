import os
import time
import multiprocessing as mp

import numpy as np
import pandas as pd
import networkx as nx
import geopandas as gpd
import osmnx as ox
from shapely.geometry import Point, LineString

from .converters import parse_time_to_seconds
from .connectors import  _fill_coordinates, connect_stops_to_streets_utm


def _preprocess_schedules(graph):
    # Сортировка расписаний для ускоренного поиска ближейших отправлений при помощи бинарного поиска
    for edge in graph.edges(data=True):
        if 'schedules' in edge[2]:
            # Сортировка
            edge[2]['sorted_schedules'] = sorted(edge[2]['schedules'], 
                                                 key=lambda x: x[0])

def _add_edges_to_graph(G: nx.MultiDiGraph, 
                        sorted_stop_times: pd.DataFrame, 
                        trips_df: pd.DataFrame, 
                        shapes: dict,
                        trip_to_shape_map: dict, 
                        read_shapes: bool = False
                        ):
    """
    Adds edges with schedule information and optionally shape geometry between stops to the graph.

    Parameters:
    G (nx.MultiDiGraph): The networkx graph to which the edges will be added.
    sorted_stop_times (pd.DataFrame): A DataFrame containing sorted stop times information.
    trips_df (pd.DataFrame): A DataFrame containing trip information, including shape_id.
    shapes (dict): A dictionary mapping shape_ids to their respective linestring geometries.
    trip_to_shape_map (dict): A dictionary mapping trip_ids to shape_ids.
    read_shapes (bool): If True, shape geometries will be added to the edges. Defaults to True.
    """
    # Для каждой последовательности остановок в группе создается ребро с расписанием
    # Если ребро между остановками существует, то добавляется расписание к существующему ребру

    # Предобработка с созданием словаря trip_id -> shape_id с целью
    # Избежать повторного поиска shape_id по датафрейму для каждого рейса

    for i in range(len(sorted_stop_times) - 1):
        start_stop = sorted_stop_times.iloc[i] # i-ая остановка (строка DF)
        end_stop = sorted_stop_times.iloc[i + 1]
        edge = (start_stop['stop_id'], end_stop['stop_id'])
        
        departure = parse_time_to_seconds(start_stop['departure_time'])
        arrival = parse_time_to_seconds(end_stop['arrival_time'])
        trip_id = start_stop['trip_id']
        
        # Получение route_id из trips_df (поиск по trip_id в trips_df и выбор колонки route_id)
        route_id = trips_df.loc[trips_df['trip_id'] == trip_id, 'route_id'].values[0]
        schedule_info = (departure, arrival, route_id)

        geometry = None
        if read_shapes:
            shape_id = trip_to_shape_map.get(trip_id)
            if shape_id in shapes:
                geometry = shapes[shape_id]
            else:
                # Возможно будет логирование
                pass

        # Если ребро уже существует, добавляем расписание к списку расписаний
        # Получается, что геометрия рейса добавляется по первому попавшемуся маршруту
        if G.has_edge(*edge):
            G[edge[0]][edge[1]]['schedules'].append(schedule_info)
        # Иначе создаем новое ребро
        else:
            if read_shapes:
                G.add_edge(*edge, schedules=[schedule_info], 
                           type='transit', geometry=geometry)
            else:
                G.add_edge(*edge, schedules=[schedule_info], type='transit')

def add_edges_parallel(graph, trips_chunk, trips_df, shapes, read_shapes, trip_to_shape_map):
    """
    Adds edges to the graph for a chunk of trips.

    Args:
    -----
    - graph: The graph to which edges will be added.
    - trips_chunk: A subset of trips data.
    - trips_df: DataFrame with trips information.
    - shapes: Geometries of the shapes.
    - read_shapes: Flag indicating whether to read shapes.
    - trip_to_shape_map: Mapping of trip_ids to shape_ids.
    """
    local_graph = graph.copy()  # Make a copy of the graph for local modifications
    for trip_id, group in trips_chunk.groupby('trip_id'):
        sorted_group = group.sort_values('stop_sequence')
        _add_edges_to_graph(local_graph, 
                            sorted_group, 
                            trips_df = trips_df, 
                            shapes = shapes, 
                            read_shapes = read_shapes,
                            trip_to_shape_map = trip_to_shape_map)
    return local_graph

def _filter_stop_times_by_time(stop_times: pd.DataFrame, departure_time: int, duration_seconds: int):
    """Filters stop_times to only include trips that occur within a specified time window."""
    
    stop_times['departure_time_seconds'] = stop_times['departure_time'].apply(parse_time_to_seconds)
    return stop_times[
        (stop_times['departure_time_seconds'] >= departure_time) &
        (stop_times['departure_time_seconds'] <= departure_time + duration_seconds)
    ]

def _load_GTFS(GTFSpath: str, departure_time_input: str, day_of_week: str, duration_seconds, read_shapes = False):
    """
    Загружает данные GTFS из заданного пути каталога и возвращает граф, а также датафрейм остановок.
    Функция использует параллельные вычисления для ускорения загрузки данных.
    
    Args:
    ---------
    - GTFSpath (str): Путь к каталогу, содержащему файлы данных GTFS.
    - departure_time_input (str): Время отправления в формате HH:MM:SS.
    - day_of_week (str): День недели в нижнем регистре, например "monday".
    - duration_seconds (int): Длительность временного окна загрузки в секундах.
    - read_shapes (bool): Флаг чтения геометрии, передаваемыый из "feed_to_graph"
    
    Returns:
    ---------
    - Tuple[nx.MultiDiGraph, str, int, pd.DataFrame]: Кортеж, содержащий следующее:
        - nx.MultiDiGraph: Граф, представляющий данные GTFS.
        - pd.DataFrame: Датафрейм, содержащий информацию об остановках.
    """
    # Initialize the graph and read data files.
    G = nx.DiGraph()

    stops_df = pd.read_csv(os.path.join(GTFSpath, "stops.txt"), 
                           usecols=['stop_id', 
                                    'stop_lat', 
                                    'stop_lon'])
    
    stop_times_df = pd.read_csv(os.path.join(GTFSpath, "stop_times.txt"), 
                                usecols=['departure_time', 
                                         'trip_id', 
                                         'stop_id', 
                                         'stop_sequence', 
                                         'arrival_time'
                                         ])
    
    trips_df = pd.read_csv(os.path.join(GTFSpath, "trips.txt"))
    
    routes = pd.read_csv(os.path.join(GTFSpath, "routes.txt"), 
                         usecols=['route_id', 
                                  'route_short_name'])
    
    # Load shapes.txt if read_shapes is True
    if read_shapes:
        if 'shapes.txt' not in os.listdir(GTFSpath):
            raise FileNotFoundError('shapes.txt not found')
        
        shapes_df = pd.read_csv(os.path.join(GTFSpath, "shapes.txt"))
        
        # Group geometry by shape_id, resulting in a Pandas Series
        # with trip_id (shape_id ?) as keys and LineString geometries as values
        shapes = shapes_df.groupby('shape_id').apply(
            lambda group: LineString(group[['shape_pt_lon', 'shape_pt_lat']].values)
            )
        # Mapping trip_id to shape_id for faster lookup
        trip_to_shape_map = trips_df.set_index('trip_id')['shape_id'].to_dict()
        
    else:
        shapes = None
        trip_to_shape_map = None
  
    # Join route information to trips
    trips_df = trips_df.merge(routes, on='route_id')
    
    # Check if calendar.txt exists in GTFS directory
    # If it does, filter by day of the week, otherwise raise an error
    if 'calendar.txt' in os.listdir(GTFSpath):
        calendar_df = pd.read_csv(os.path.join(GTFSpath, "calendar.txt"))
        # Filter for the day of the week
        service_ids = calendar_df[calendar_df[day_of_week] == 1]['service_id']
        trips_df = trips_df[trips_df['service_id'].isin(service_ids)]
    else:
        raise FileNotFoundError('calendar.txt not found')
        #print('calendar.txt not found, loading all data')
    
    # Filter stop_times to only include trips that occur within a specified time window
    valid_trips = stop_times_df['trip_id'].isin(trips_df['trip_id'])
    stop_times_df = stop_times_df[valid_trips].dropna()

    # Convert departure_time from HH:MM:SS o seconds
    departure_time_seconds = parse_time_to_seconds(departure_time_input)
    
    # Filtering stop_times by time window
    filtered_stops = _filter_stop_times_by_time(stop_times_df, 
                                                departure_time_seconds, 
                                                duration_seconds
                                                )

    print(f'GTFS data loaded\n{len(filtered_stops)} of {len(stop_times_df)} trips retained')

    # Adding stops as nodes to the graph
    for _, stop in stops_df.iterrows():
        G.add_node(stop['stop_id'], 
                   type = 'transit', 
                   pos=(stop['stop_lon'], stop['stop_lat']), 
                   x=stop['stop_lon'], 
                   y=stop['stop_lat']
                   )
        
    # Track time for benchmarking
    timestamp = time.time()
    # Divide filtered_stops into chunks for parallel processing
    # Use half of the available CPU logical cores (likely equal to the number of physical cores)
    num_cores = int(mp.cpu_count() / 2)
    chunks = np.array_split(filtered_stops, num_cores)

    # Create a pool of processes
    with mp.Pool(processes=num_cores) as pool:
        # Create a subgraph in each process
        # Each process will return a graph with edges for a subset of trips
        # The results will be combined into a single graph
        results = pool.starmap(add_edges_parallel, 
                               [
                                (G, chunk, trips_df, shapes, 
                                read_shapes, trip_to_shape_map) 
                                for chunk in chunks
                                ]
                            )
    
    # Merge results from all processes
    merged_graph = nx.DiGraph()
    
    for graph in results:
        merged_graph.add_nodes_from(graph.nodes(data=True))
    # Add edges from subgraphs to the merged graph    
    for graph in results:
        for u, v, data in graph.edges(data=True):
            # If edge already exists, merge schedules
            if merged_graph.has_edge(u, v):
                # Merge sorted_schedules attribute
                existing_schedules = merged_graph[u][v]['schedules']
                new_schedules = data['schedules']
                merged_graph[u][v]['schedules'] = existing_schedules + new_schedules
            # If edge does not exist, add it
            else:
                # Add new edge with data
                merged_graph.add_edge(u, v, **data)
    
    #print ('holy pepperoni, this was hard')
    print(f'Building graph in parallel complete in {time.time() - timestamp} seconds')
    
    # Deprecated as this solution leads to some edges attributes being overwritten
    """ # Combine results from all processes
    for result_graph in results:
        G = nx.compose(G, result_graph) """
        
    # Sorting schedules for faster lookup using binary search
    _preprocess_schedules(merged_graph)
    
    print('Transit graph created')
        
    return merged_graph, stops_df

def _load_osm(stops, save_graphml, path)-> nx.DiGraph:
    """
    Loads OpenStreetMap data within a convex hull of stops in GTFS feed, 
    creates a street network graph, and adds walking times as edge weights.

    Returns:
    ----------
    G_city : networkx multidigraph
        A street network graph with walking times as edge weights.
    """
    
    #Построение выпуклой оболочки по координатам остановок для загрузки OSM
    boundary = gpd.GeoSeries(
                            [Point(lon, lat) for lon, lat 
                            in zip(stops['stop_lon'], stops['stop_lat'])
                            ]
                            ).unary_union.convex_hull
    
    print('Loading OSM graph via OSMNX')
    #Загрузка OSM в пределеах полигона
    G_city = ox.graph_from_polygon(boundary, 
                                   network_type='walk', 
                                   simplify=True)
    print('Street network graph created')
    
    #Добавление времени пешего пути на улицах
    walk_speed_mps = 1.39  # 5кмч
    for _, _, _, data in G_city.edges(data=True, keys = True):
        distance = data['length']
        walk_time = distance / walk_speed_mps
        data['weight'] = walk_time
        data['type'] = 'street'
    
    for _, data in G_city.nodes(data = True):
        data['type'] = 'street'
    
    if save_graphml:
        ox.save_graphml(G_city, path)
        
    #Конвертация MultiGraph из OSMNX в DiGraph
    G_city = nx.DiGraph(G_city)
    
    return G_city
 
def feed_to_graph(
    GTFSpath: str, 
    departure_time_input: str, 
    day_of_week: str, 
    duration_seconds: int, 
    save_to_csv: bool = False, 
    read_shapes = False, 
    input_graph_path = None, 
    output_graph_path = None, 
    save_graphml = False, 
    load_graphml = False, 
    save_folder = None
    ) -> nx.DiGraph:
    """
    Создает мультимодальный граф, на основе данных о 
    траспорте в формате GTFS и OpenStreetMap.
    
    Args:
    ---------
    - GTFSpath (str): Путь к файлам GTFS.
    - departure_time_input (str): Время отправления в формате 'HH:MM:SS'.
    - day_of_week (str): День недели в нижнем регистре (например, 'monday').
    - duration_seconds (int): Период с момента отправления, для которого будет загружен граф.
    - save_to_csv (bool): Флаг сохранения узлов и ребер графа в csv.
    - read_shapes (bool, optional): Флаг чтения геометрии из файла shapes.txt.
    - input_graph_path (str, optional): Путь к файлу с графом OSM в формате GraphML.
    - output_graph_path (str, optional): Путь для сохранения графа OSM в формате GraphML.
    - save_graphml (bool, optional): Флаг сохранения графа OSM в формате GraphML.
    - load_graphml (bool, optional): Флаг загрузки графа OSM из файла GraphML.
    
    Returns:
    ----------
    - G_combined (nx.DiGraph): Объединенный граф.
    - stops (pd.DataFrame): Pd.Dataframe с информацией об остановках.
    """
    G_transit, stops = _load_GTFS(GTFSpath, departure_time_input, 
                                  day_of_week, duration_seconds, 
                                  read_shapes = read_shapes)
    
    if load_graphml:
        print('Loading OSM graph from GraphML file')
        
        # Словарь с типами данных для ребер
        edge_dtypes = {'weight': float, 'length': float}
        G_city = ox.load_graphml(input_graph_path, edge_dtypes = edge_dtypes)
        G_city = nx.DiGraph(G_city)
    else:
        #Импорт данных OSM
        G_city = _load_osm(stops, save_graphml, output_graph_path)
    
    # Обьединение OSM и GTFS
    G_c = nx.compose(G_transit, G_city)
    del(G_transit, G_city)

    # Заполнение прямоугольных координат UTM для узлов графа
    _fill_coordinates(G_c)
    
    print("Combining graphs")
    #Соединение остановок с улицами OSM
    G_combined = connect_stops_to_streets_utm(G_c, stops)
    
    del(G_c)
    print(f'Number of nodes: {G_combined.number_of_nodes()}\n'
            f'Number of edges: {G_combined.number_of_edges()}\n'
            'Connecting stops to streets complete')
    
    #Сохранение узлов и ребер графа в csv
    if save_to_csv:
        
        df = pd.DataFrame(G_combined.edges(data=True))
        df.to_csv(rf'{save_folder}/Edges.csv', index=False)

        df_nodes = pd.DataFrame(G_combined.nodes(data=True))
        df_nodes.to_csv(rf'{save_folder}/Nodes.csv', index=False)

        del(df, df_nodes)
    
    return G_combined, stops