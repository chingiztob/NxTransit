import os

import pandas as pd
import networkx as nx
import geopandas as gpd
import osmnx as ox
from shapely.geometry import Point

from .converters import parse_time_to_seconds
from .connectors import connect_stops_to_streets, _fill_coordinates, connect_stops_to_streets_utm


def _preprocess_schedules(graph):
    # Сортировка расписаний для ускоренного поиска ближейших отправлений при помощи бинарного поиска
    for edge in graph.edges(data=True):
        if 'schedules' in edge[2]:
            # Сортировка
            edge[2]['sorted_schedules'] = sorted(edge[2]['schedules'], 
                                                 key=lambda x: x[0])

def _add_stop_times_to_graph(G: nx.MultiDiGraph, sorted_stop_times: pd.DataFrame, trips_df):
    
    """Adds edges with schedule information between stops to the graph."""
    
    # Для каждой последовательности остановок в группе создается ребро с расписанием
    # Если ребро между остановками существует, то добавляется расписание к существующему ребру
    for i in range(len(sorted_stop_times) - 1):
        start_stop = sorted_stop_times.iloc[i]
        end_stop = sorted_stop_times.iloc[i + 1]
        edge = (start_stop['stop_id'], end_stop['stop_id'])
        departure = parse_time_to_seconds(start_stop['departure_time'])
        arrival = parse_time_to_seconds(end_stop['arrival_time'])
        
        # Получение route_id по внешнему ключу из trips_df
        route_id = trips_df.loc[trips_df['trip_id'] == start_stop['trip_id'], 'route_id'].values[0]

        schedule_info = (departure, arrival, route_id)  # Номер машрута добавляется в кортеж с расписанием
        
        # Если ребро уже существует, оно дополняется, иначе создается новое
        if G.has_edge(*edge):
            G[edge[0]][edge[1]]['schedules'].append(schedule_info)
        else:
            G.add_edge(*edge, schedules=[schedule_info], type = 'transit')

def _filter_stop_times_by_time(stop_times: pd.DataFrame, departure_time: int, duration_seconds: int):
    """Filters stop_times to only include trips that occur within a specified time window."""
    
    stop_times['departure_time_seconds'] = stop_times['departure_time'].apply(parse_time_to_seconds)
    return stop_times[
        (stop_times['departure_time_seconds'] >= departure_time) &
        (stop_times['departure_time_seconds'] <= departure_time + duration_seconds)
    ]

def _load_GTFS(GTFSpath: str, departure_time_input: str, day_of_week: str, duration_seconds):
    """
    Загружает данные GTFS из заданного пути каталога и возвращает граф, а также датафрейм остановок.
    
    Args:
    ---------
    - GTFSpath (str): Путь к каталогу, содержащему файлы данных GTFS.
    - departure_time_input (str): Время отправления в формате HH:MM:SS.
    - day_of_week (str): День недели в нижнем регистре, например "monday".
    - duration_seconds (int): Длительность временного окна загрузки в секундах.

    Returns:
    ---------
    - Tuple[nx.MultiDiGraph, str, int, pd.DataFrame]: Кортеж, содержащий следующее:
        - nx.MultiDiGraph: Граф, представляющий данные GTFS.
        - pd.DataFrame: Датафрейм, содержащий информацию об остановках.
    """
    # Initialize the graph and read data files.
    G = nx.DiGraph()
    city_name = os.path.basename(GTFSpath)
    
    stops_df = pd.read_csv(os.path.join(GTFSpath, "stops.txt"), 
                           usecols=['stop_id', 'stop_lat', 'stop_lon'])
    
    stop_times_df = pd.read_csv(os.path.join(GTFSpath, "stop_times.txt"), 
                                usecols=['departure_time', 'trip_id', 
                                         'stop_id', 'stop_sequence', 
                                         'arrival_time'])
    
    trips_df = pd.read_csv(os.path.join(GTFSpath, "trips.txt"))
    
    routes = pd.read_csv(os.path.join(GTFSpath, "routes.txt"), usecols=['route_id', 'route_short_name'])
    # Присоединение информации о маршрутах к trips
    trips_df = trips_df.merge(routes, on='route_id')
    
    # Проверка наличия файла calendar.txt в каталоге GTFS
    # При наличии выполняется фильтрация по дню недели, при отсутствии загружаются все данные
    if 'calendar.txt' in os.listdir(GTFSpath):
        calendar_df = pd.read_csv(os.path.join(GTFSpath, "calendar.txt"))
        # Filter for the day of the week
        service_ids = calendar_df[calendar_df[day_of_week] == 1]['service_id']
        trips_df = trips_df[trips_df['service_id'].isin(service_ids)]
    
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
        G.add_node(stop['stop_id'], pos=(stop['stop_lon'], stop['stop_lat']), 
                   x=stop['stop_lon'], y=stop['stop_lat'])
    
    # Разбиение всех trips на группы по trip_id, далее итеративная обработка каждой группы
    # Для каждой группы сортировка по stop_sequence, далее добавление ребер в граф
    for trip_id, group in filtered_stops.groupby('trip_id'):
        sorted_group = group.sort_values('stop_sequence')
        _add_stop_times_to_graph(G, sorted_group, trips_df = trips_df)
        
    # Предварительная сортировка расписаний для ускоренного поиска при помощи бинарного поиска
    _preprocess_schedules(G) 
    
    return G, stops_df

def _load_osm(stops)-> nx.DiGraph:
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
    print('Выпуклая оболочка построена, начинается загрузка графа улиц')

    #Загрузка OSM в пределеах полигона
    G_city = ox.graph_from_polygon(boundary, 
                                   network_type='walk', 
                                   simplify=True)
    
    #Добавление времени пешего пути на улицах
    walk_speed_mps = 1.39  # 5кмч
    for _, _, _, data in G_city.edges(data=True, keys = True):
        distance = data['length']
        walk_time = distance / walk_speed_mps
        data['weight'] = walk_time
        data['type'] = 'street'

    #Конвертация MultiGraph из OSMNX в DiGraph
    G_city = nx.DiGraph(G_city)

    return G_city
 
def feed_to_graph(GTFSpath: str, departure_time_input: str, day_of_week: str, 
                      duration_seconds, save_to_csv: bool) -> nx.DiGraph:
    """
    Создает граф, объединяющий данные GTFS и OSM.
    
    Args:
    ----------
    - GTFSpath (str): Путь к файлам GTFS.
    - departure_time_input (str): Время отправления в формате 'HH:MM:SS'.
    - day_of_week (str): День недели в нижнем регистре (например, 'monday').
    - duration_seconds (int): Период с момента отправления, для которого будет загружен граф.
    - save_to_csv (bool): Флаг сохранения узлов и ребер графа в csv.
    
    Returns:
    ----------
    - G_combined (nx.DiGraph): Объединенный граф.
    - stops (pd.DataFrame): Pd.Dataframe с информацией об остановках.
    """
    G_transit, stops = _load_GTFS(GTFSpath, departure_time_input, 
                                  day_of_week, duration_seconds)
    #Импорт данных OSM
    G_city = _load_osm(stops)
    print('Граф улиц загружен')
    
    # Обьединение OSM и GTFS
    G_c = nx.compose(G_transit, G_city)
    del(G_transit, G_city)

    #указание типа узлов
    transit_stop_ids = set(stops['stop_id'])
    for node in G_c.nodes():
        if node in transit_stop_ids:
            G_c.nodes[node]['type'] = 'transit'
        else:
            G_c.nodes[node]['type'] = 'street'
    
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