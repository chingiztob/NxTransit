"""Tools for connecting GTFS stops to the nearest street node in the graph."""
import pandas as pd
import shapely.geometry
from scipy.spatial import KDTree
from pyproj import Transformer, CRS


def _fill_coordinates(graph):
    
    crs_4326 = CRS.from_epsg(4326)
    crs_4087 = CRS.from_epsg(4087)
    transformer = Transformer.from_crs(crs_4326, crs_4087)

    for node in graph.nodes():
        try:
            coords = transformer.transform(graph.nodes[node]['y'], graph.nodes[node]['x'])

            graph.nodes[node]['metric_X'] = coords[0]
            graph.nodes[node]['metric_Y'] = coords[1]
        except Exception:
            continue

    return graph


def connect_stops_to_streets(graph, stops: pd.DataFrame):
    """
    Connects GTFS stops to the nearest street node in the graph
    using projected coordinates in EPSG:4087.
    """
    # Create a list of street node tuples (x, y, node_id)
    node_data = [(data['metric_X'], data['metric_Y'], n)
                 for n, data in graph.nodes(data=True)
                 if 'metric_X' in data and 'metric_Y' in data
                 and data['type'] == 'street']

    node_data_wgs = [(data['x'], data['y'], n)
                     for n, data in graph.nodes(data=True)
                     if 'y' in data and 'x' in data
                     and data['type'] == 'street']

    # Create a KD-tree for nearest neighbor search
    # The tree is created from a list of street node tuples (x, y, node_id)
    tree = KDTree([(x, y) for x, y, _ in node_data])

    for _, stop in stops.iterrows():

        stop_wgs = (stop['stop_lon'], stop['stop_lat'])
        x, y = graph.nodes[stop['stop_id']]['metric_X'], graph.nodes[stop['stop_id']]['metric_Y']
        stop_coords = (x, y)

        # query returns the distance to the nearest neighbor and its index in the tree
        distance, idx = tree.query(stop_coords)
        nearest_street_node = node_data[idx][2]

        # Add a connector edge to the graph
        # The connection only happens if the found node is a street
        # Maybe this additional check is not needed
        if graph.nodes[nearest_street_node]['type'] == 'street':  # Соединение

            # Создаем геометрию ребра в формате Shapely LineString
            stop_geom = shapely.geometry.Point(stop_wgs)
            street_geom = shapely.geometry.Point((node_data_wgs[idx][0], node_data_wgs[idx][1]))
            linestring = shapely.geometry.LineString([stop_geom, street_geom])

            walk_speed_mps = 1.39
            
            walk_time = distance / walk_speed_mps

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


def snap_points_to_network(graph, points):
    """
    Snaps point features from GeoDataFrame to the nearest street node in the graph.

    Parameters
    ----------
        graph : networkx.Graph
            NetworkX graph representing the transit system.
        points : geopandas.GeoDataFrame
            GeoDataFrame containing point geometries.

    Returns
    -------
        graph: networkx.Graph
            NetworkX graph with added snapped points as nodes.
    
    Notes
    -----
    points CRS must be EPSG:4326
    """
    # Create a list of street node tuples (x, y, node_id)
    node_data_wgs = [(data['x'], data['y'], n)
                 for n, data in graph.nodes(data=True)
                 if 'y' in data and 'x' in data
                 and data['type'] == 'street']

    node_data_metric = [(data['metric_X'], data['metric_Y'], n)
                 for n, data in graph.nodes(data=True)
                 if 'metric_X' in data and 'metric_Y' in data
                 and data['type'] == 'street']

    # Create a KD-tree for nearest neighbor search
    # The tree is created from a list of street node tuples (x, y, node_id)
    tree = KDTree([(x, y) for x, y, _ in node_data_metric])
    
    crs_4326 = CRS.from_epsg(4326)
    crs_4087 = CRS.from_epsg(4087)
    transformer = Transformer.from_crs(crs_4326, crs_4087)

    for index, row in points.iterrows():

        geometry = row['geometry']
        id = row['origin_id']
        
        pnt_x, pnt_y = transformer.transform(geometry.y, geometry.x)

        # query returns the distance to the nearest neighbor and its index in the tree
        distance, idx = tree.query((pnt_x, pnt_y))
        nearest_street_node = node_data_metric[idx][2]

        # Add a connector edge to the graph
        # The connection only happens if the found node is a street
        # Maybe this additional check is not needed
        if graph.nodes[nearest_street_node]['type'] == 'street':
            # Создаем геометрию ребра в формате Shapely LineString
            street_geom = shapely.geometry.Point((node_data_wgs[idx][0],
                                                  node_data_wgs[idx][1]))
            linestring = shapely.geometry.LineString([geometry, street_geom])

            walk_speed_mps = 1.39
            walk_time = distance / walk_speed_mps

            graph.add_node(id, x=geometry.x, y=geometry.y, type='snapped')

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
