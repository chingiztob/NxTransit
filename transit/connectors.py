import pandas as pd

import utm
from scipy.spatial import cKDTree
import shapely.geometry

def latlon_to_utm(lat, lon):
    """
    Convert coordinates from WGS84 (latitude, longitude) to rectangular UTM coordinates.

    Args:
        lat (float): Latitude in decimal degrees.
        lon (float): Longitude in decimal degrees.

    Returns:
        tuple: A tuple containing the UTM easting and northing coordinates in meters.
    """
    x,y,_,_ = utm.from_latlon(lat,lon)
    return x,y

def connect_stops_to_streets_utm(graph, stops: pd.DataFrame):
    """
    Connects GTFS stops to the nearest street node in the graph
    using rectangular UTM coordinates.
    """
    # Создание списка кортежей узлов улиц (x, y, node_id)
    node_data = [(data['UTM_X'], data['UTM_Y'], n) 
                 for n, data in graph.nodes(data=True) 
                 if 'UTM_X' in data and 'UTM_Y' in data 
                 and data['type'] == 'street']
    
    node_data_wgs = [(data['x'], data['y'], n) 
                    for n, data in graph.nodes(data=True) 
                    if 'y' in data and 'x' in data 
                    and data['type'] == 'street']
    
    # Создание KD-дерева для решения задачи поиска ближайшего соседа
    # Дерево создается из списка кортежей узлов улиц (x, y, node_id)
    tree = cKDTree([(x, y) for x, y, _ in node_data])
    
    for _, stop in stops.iterrows():
        
        stop_wgs = (stop['stop_lon'], stop['stop_lat'])
        x, y = graph.nodes[stop['stop_id']]['UTM_X'], graph.nodes[stop['stop_id']]['UTM_Y']
        stop_coords = (x, y)   
        
        # query возращает расстояние до ближайшего соседа и его индекс в дереве
        distance, idx = tree.query(stop_coords)
        nearest_street_node = node_data[idx][2]

        # Добавляем соединительное ребро в граф
        # Соединение происходит только если найденный узел является улицей
        # Возможно эта дополнительная проверка и не нужна
        if graph.nodes[nearest_street_node]['type'] == 'street':  # Соединение 
            
            # Создаем геометрию ребра в формате Shapely LineString
            stop_geom = shapely.geometry.Point(stop_wgs)
            street_geom = shapely.geometry.Point((node_data_wgs[idx][0], node_data_wgs[idx][1]))
            linestring = shapely.geometry.LineString([stop_geom, street_geom])

            walk_speed_mps = 1.39
            # На данный момент при расчете используется некорректное расстояние в градусах
            # Впрочем влияние на результат незначительно
            # Когда-нибудь будет переделано с UTM
            walk_time = distance / walk_speed_mps
    
            # Заполняем атрибуты ребра
            graph.add_edge(stop['stop_id'], nearest_street_node,
                           weight=walk_time,
                           type='connector',
                           geometry=linestring
                           )
            graph.add_edge(nearest_street_node, stop['stop_id'],
                           weight=walk_time,
                           type='connector',
                           geometry=linestring
                           )

    return graph

def _fill_coordinates(graph):
    '''
    Populates the UTM_X, and UTM_Y attributes of each node in the graph based on the stop coordinates.
    
    Args:
    ----------
    - graph: NetworkX graph representing transit system.
    - stops: Pandas DataFrame containing stop_id, stop_lat, and stop_lon columns.
    
    Returns:
    ----------
    - graph: NetworkX graph with updated node attributes.
    '''
    for node in graph.nodes():
        try:

            UTM = latlon_to_utm(graph.nodes[node]['y'], 
                                graph.nodes[node]['x'])
            
            graph.nodes[node]['UTM_X'] = UTM[0]
            graph.nodes[node]['UTM_Y'] = UTM[1]
        except:
            continue
        
    return graph

def snap_points_to_network(graph, points):
    """
    Snaps point features from GeoDataFrame to 
    the nearest street node in the graph.
    
    Parameters:
    ----------
    - graph: NetworkX graph representing transit system.
    - points: GeoDataFrame containing point geometries.
    
    Returns:
    ----------
    - graph: NetworkX graph with added snapped points as nodes.
    """
    # Создание списка кортежей узлов улиц (x, y, node_id)
    node_data = [(data['x'], data['y'], n) 
                 for n, data in graph.nodes(data=True) 
                 if 'y' in data and 'x' in data 
                 and data['type'] == 'street']

    # Создание KD-дерева для решения задачи поиска ближайшего соседа
    # Дерево создается из списка кортежей узлов улиц (x, y, node_id)
    tree = cKDTree([(lon, lat) for lon, lat, _ in node_data])
    
    for index, row in points.iterrows():
        
        geometry = row['geometry']
        id = row['id']
        pnt_coords = (geometry.x, geometry.y)
        
        # query возращает расстояние до ближайшего соседа и его индекс в дереве
        distance, idx = tree.query(pnt_coords)
        nearest_street_node = node_data[idx][2]

        # Добавляем ребро коннектор в граф
        # Соединение происходит только если найденный узел 
        # является улицей (дополнительная проверка)
        # Возможно она и не нужна
        if graph.nodes[nearest_street_node]['type'] == 'street':  # Соединение 
            
            # Создаем геометрию ребра в формате Shapely LineString
            street_geom = shapely.geometry.Point((node_data[idx][0], 
                                                  node_data[idx][1]))
            linestring = shapely.geometry.LineString([geometry, street_geom])

            walk_speed_mps = 1.39
            # На данный момент при расчете используется некорректное расстояние
            # Будет переделано с UTM
            walk_time = distance / walk_speed_mps
            
            graph.add_node(id, x=geometry.x, y=geometry.y, type='snapped')
    
            # Заполняем атрибуты ребра
            graph.add_edge(id, nearest_street_node,
                           weight=walk_time,
                           type='connector',
                           geometry=linestring
                           )
            graph.add_edge(nearest_street_node, id,
                           weight=walk_time,
                           type='connector',
                           geometry=linestring
                           )

    return graph