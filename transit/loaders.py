import os

import pandas as pd
import networkx as nx
import geopandas as gpd
import osmnx as ox
from shapely.geometry import Point, LineString

from .converters import parse_time_to_seconds
from .connectors import connect_stops_to_streets, _fill_coordinates, connect_stops_to_streets_utm


def _preprocess_schedules(graph):
    # Сортировка расписаний для ускоренного поиска ближейших отправлений при помощи бинарного поиска
    for edge in graph.edges(data=True):
        if 'schedules' in edge[2]:
            # Сортировка
            edge[2]['sorted_schedules'] = sorted(edge[2]['schedules'], 
                                                 key=lambda x: x[0])

def _add_edges_to_graph(G: nx.MultiDiGraph, sorted_stop_times: pd.DataFrame, 
                                        trips_df: pd.DataFrame, shapes: dict, 
                                        read_shapes: bool = False):
    """
    Adds edges with schedule information and optionally shape geometry between stops to the graph.

    Parameters:
    G (nx.MultiDiGraph): The networkx graph to which the edges will be added.
    sorted_stop_times (pd.DataFrame): A DataFrame containing sorted stop times information.
    trips_df (pd.DataFrame): A DataFrame containing trip information, including shape_id.
    shapes (dict): A dictionary mapping shape_ids to their respective linestring geometries.
    read_shapes (bool): If True, shape geometries will be added to the edges. Defaults to True.
    """
    # Для каждой последовательности остановок в группе создается ребро с расписанием
    # Если ребро между остановками существует, то добавляется расписание к существующему ребру

    # Предобработка с созданием словаря trip_id -> shape_id с целью
    # Избежать повторного поиска shape_id по датафрейму для каждого рейса
    if read_shapes:
        trip_to_shape_map = trips_df.set_index('trip_id')['shape_id'].to_dict()

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
                G.add_edge(*edge, schedules=[schedule_info], type='transit', geometry=geometry)
            else:
                G.add_edge(*edge, schedules=[schedule_info], type='transit')

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
                           usecols=['stop_id', 'stop_lat', 'stop_lon'])
    
    stop_times_df = pd.read_csv(os.path.join(GTFSpath, "stop_times.txt"), 
                                usecols=['departure_time', 'trip_id', 
                                         'stop_id', 'stop_sequence', 'arrival_time'])
    
    trips_df = pd.read_csv(os.path.join(GTFSpath, "trips.txt"))
    
    routes = pd.read_csv(os.path.join(GTFSpath, "routes.txt"), 
                         usecols=['route_id', 'route_short_name'])
    
    # Загрузка файла shapes.txt и сгруппированной геометрии 
    if read_shapes:
        shapes_df = pd.read_csv(os.path.join(GTFSpath, "shapes.txt"))
        
        # Группировка геометрии по shape_id, получается Pandas Series 
        # с ключами trip_id и значениями геометрии LineString
        shapes = shapes_df.groupby('shape_id').apply(
            lambda group: LineString(group[['shape_pt_lon', 'shape_pt_lat']].values)
            )
    else:
        shapes = None
  
    # Присоединение информации о маршрутах к trips
    trips_df = trips_df.merge(routes, on='route_id')
    
    # Проверка наличия файла calendar.txt в каталоге GTFS
    # При наличии выполняется фильтрация по дню недели, при отсутствии загружаются все данные
    if 'calendar.txt' in os.listdir(GTFSpath):
        calendar_df = pd.read_csv(os.path.join(GTFSpath, "calendar.txt"))
        # Filter for the day of the week
        service_ids = calendar_df[calendar_df[day_of_week] == 1]['service_id']
        trips_df = trips_df[trips_df['service_id'].isin(service_ids)]
    else:
        print('calendar.txt not found, loading all data')
    
    # Фильтрация только тех рейсов, которые соответствуют условиям
    valid_trips = stop_times_df['trip_id'].isin(trips_df['trip_id'])
    stop_times_df = stop_times_df[valid_trips].dropna()

    # Перевод HH:MM:SS в секунды с полуночи
    departure_time_seconds = parse_time_to_seconds(departure_time_input)
    
    # Выборка только тех рейсов, которые соответствуют указанному временному окну
    filtered_stops = _filter_stop_times_by_time(stop_times_df, departure_time_seconds, 
                                                duration_seconds
                                                )

    print(f'GTFS data loaded\n{len(filtered_stops)} of {len(stop_times_df)} trips retained')

    # Добавление узлов в граф с координатами остановок
    for _, stop in stops_df.iterrows():
        G.add_node(stop['stop_id'], type = 'transit', pos=(stop['stop_lon'], stop['stop_lat']), 
                   x=stop['stop_lon'], y=stop['stop_lat'])
    
    # Разбиение всех trips на группы по trip_id, далее итеративная обработка каждой группы
    # Для каждой группы сортировка по stop_sequence, далее добавление ребер в граф
    for trip_id, group in filtered_stops.groupby('trip_id'):
        sorted_group = group.sort_values('stop_sequence')
        _add_edges_to_graph(G, sorted_group, trips_df = trips_df, 
                                 shapes = shapes, read_shapes = read_shapes)
        
    # Предварительная сортировка расписаний для ускоренного поиска при помощи бинарного поиска
    _preprocess_schedules(G) 
    
    print('Закончено формирование графа транспорта')
        
    return G, stops_df

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
                            ]).unary_union.convex_hull
    print('Выпуклая оболочка построена, начинается загрузка графа улиц OSM в сети')

    #Загрузка OSM в пределеах полигона
    G_city = ox.graph_from_polygon(boundary, 
                                   network_type='walk', 
                                   simplify=True)
    print('Граф улиц загружен')
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
 
def feed_to_graph(GTFSpath: str, departure_time_input: str, day_of_week: str, 
                      duration_seconds, save_to_csv: bool, read_shapes = False, input_graph_path = None, 
                      output_graph_path = None, save_graphml = False, load_graphml = False, 
                      ) -> nx.DiGraph:
    """
    Создает граф, объединяющий данные GTFS и OSM.
    
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
        print('Выполняется загрузка графа OSM из файла GraphML')
        
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
    
    print("Начинается соединение графов")
    #Соединение остановок с улицами OSM
    G_combined = connect_stops_to_streets_utm(G_c, stops)
    
    del(G_c)
    print(f'Число узлов: {G_combined.number_of_nodes()}\n'
            f'Число ребер: {G_combined.number_of_edges()}\n'
            'Соединение завершено')
    
    #Сохранение узлов и ребер графа в csv
    if save_to_csv:
        
        df = pd.DataFrame(G_combined.edges(data=True))
        df.to_csv(rf'd:\Python_progs\Output\Edges_G4.csv', index=False)

        df_nodes = pd.DataFrame(G_combined.nodes(data=True))
        df_nodes.to_csv(rf'd:\Python_progs\Output\Nodes_G4.csv', index=False)

        del(df, df_nodes)
    
    return G_combined, stops