"""Tools for connecting GTFS stops to the nearest street node in the graph."""
import pandas as pd
import shapely.geometry
from scipy.spatial import KDTree
from pyproj import Transformer, CRS


def _fill_coordinates(graph):
    """
    Transforms the coordinates of nodes in the graph from EPSG:4326 to EPSG:4087.
    Populates 'metric_X' and 'metric_Y' attributes of the nodes.

    Raises
    ------
    Exception
        If an error occurs during coordinate transformation for a node.
    """
    
    crs_4326 = CRS.from_epsg(4326)
    crs_4087 = CRS.from_epsg(4087)
    transformer = Transformer.from_crs(crs_4326, crs_4087)

    for node in graph.nodes():
        try:
            coords = transformer.transform(graph.nodes[node]['y'], graph.nodes[node]['x'])

            graph.nodes[node]['metric_X'] = coords[0]
            graph.nodes[node]['metric_Y'] = coords[1]
        except Exception as e:
            raise Exception(f'{e} occurred for node {node}')


def connect_stops_to_streets(graph, stops: pd.DataFrame):
    """
    Connects GTFS stops to the nearest street node in the graph
    using projected coordinates in EPSG:4087.
    """
    # Create a list of street node tuples (x, y, node_id)
    node_data = [
        (data["metric_X"], data["metric_Y"], idx, data["x"], data["y"])
        for idx, data in graph.nodes(data=True)
        if data["type"] == "street"
    ]

    # Create a KD-tree for nearest neighbor search
    # The tree is created from a list of street node tuples (x, y, node_id)
    tree = KDTree([(x, y) for x, y, _, _, _ in node_data])

    for index, stop in stops.iterrows():

        stop_wgs = (stop['stop_lon'], stop['stop_lat'])
        x, y = graph.nodes[stop['stop_id']]['metric_X'], graph.nodes[stop['stop_id']]['metric_Y']
        stop_coords = (x, y)

        # query returns the distance to the nearest neighbor and its index in the tree
        distance, idx = tree.query(stop_coords)
        nearest_street_node = node_data[idx][2]

        # Add a connector edge to the graph
        # Create a LineString geometry for the connector edge
        stop_geom = shapely.geometry.Point(stop_wgs)
        street_geom = shapely.geometry.Point((node_data[idx][3], node_data[idx][4]))
        linestring = shapely.geometry.LineString([stop_geom, street_geom])

        walk_time = distance / 1.39  # walk speed in m/s

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
        None. The input graph is modified in-place with added snapped points as nodes.
    
    Notes
    -----
    points CRS must be EPSG:4326
    """
    # Create a list of street node tuples (x, y, node_id)
    node_data = [
        (data["metric_X"], data["metric_Y"], idx, data["x"], data["y"])
        for idx, data in graph.nodes(data=True)
        if data["type"] == "street"
    ]

    # Create a KD-tree for nearest neighbor search
    # The tree is created from a list of street node tuples (x, y, node_id)
    tree = KDTree([(x, y) for x, y, _, _, _ in node_data])
    
    crs_4326 = CRS.from_epsg(4326)
    crs_4087 = CRS.from_epsg(4087)
    transformer = Transformer.from_crs(crs_4326, crs_4087)
    
    if 'origin_id' not in points.columns:
        points['origin_id'] = points.index

    for index, row in points.iterrows():

        geometry = row['geometry']
        point_id = row['origin_id']
        pnt_x, pnt_y = transformer.transform(geometry.y, geometry.x)

        # query returns the distance to the nearest neighbor and its index in the tree
        distance, idx = tree.query((pnt_x, pnt_y))
        nearest_street_node = node_data[idx][2]

        # Add a connector edge to the graph
        # Create a LineString geometry for the connector edge
        street_geom = shapely.geometry.Point((node_data[idx][3],
                                                node_data[idx][4]))
        linestring = shapely.geometry.LineString([geometry, street_geom])

        walk_time = distance / 1.39  # walk speed in m/s

        graph.add_node(point_id, x=geometry.x, y=geometry.y, type='snapped')

        graph.add_edge(point_id, nearest_street_node,
                        weight=walk_time,
                        type='connector',
                        geometry=linestring
                        )
        graph.add_edge(nearest_street_node, point_id,
                        weight=walk_time,
                        type='connector',
                        geometry=linestring
                        )
