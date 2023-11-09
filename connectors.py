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

def connect_stops_to_streets(graph, stops: pd.DataFrame):
    """
    Connects GTFS stops to the nearest street node in the graph.
    """
    # Создание списка кортежей узлов улиц (x, y, node_id)
    node_data = [(data['x'], data['y'], n) 
                 for n, data in graph.nodes(data=True) 
                 if 'y' in data and 'x' in data 
                 and data['type'] == 'street']

    # Создание KD-дерева (www.wikipedia.org/wiki/K-d_tree) для 
    # решения задачи поиска ближайшего соседа
    # непосредственно дерево создается из списка кортежей узлов улиц (x, y, node_id)
    tree = cKDTree([(lon, lat) for lon, lat, _ in node_data])
    
    for _, stop in stops.iterrows():
        stop_coords = (stop['stop_lon'], stop['stop_lat'])
        # query возращает расстояние до ближайшего соседа и его индекс в дереве
        distance, idx = tree.query(stop_coords)
        nearest_street_node = node_data[idx][2]

        # Добавляем ребро коннектор в граф
        # Соединение происходит только если найденный узел является улицей (дополнительная проверка)
        # Возможно она и не нужна
        if graph.nodes[nearest_street_node]['type'] == 'street':  # Соединение 
            
            # Создаем геометрию ребра в формате Shapely LineString
            stop_geom = shapely.geometry.Point(stop_coords)
            street_geom = shapely.geometry.Point((node_data[idx][0], node_data[idx][1]))
            linestring = shapely.geometry.LineString([stop_geom, street_geom])

            walk_speed_mps = 1.39
            # На данный момент при расчете используется некорректное расстояние
            # Будет переделано с UTM
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