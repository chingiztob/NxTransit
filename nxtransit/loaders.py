"""Load combined GTFS and OSM data into a graph."""
import multiprocessing as mp
import os
import time

import geopandas as gpd
import networkx as nx
import numpy as np
import osmnx as ox
import pandas as pd
from shapely.geometry import LineString, Point

from .connectors import _fill_coordinates, connect_stops_to_streets
from .converters import parse_time_to_seconds


def _preprocess_schedules(graph: nx.DiGraph):
    # Sorting schedules for faster lookup using binary search
    for edge in graph.edges(data=True):
        if 'schedules' in edge[2]:
            edge[2]['sorted_schedules'] = sorted(
                edge[2]['schedules'],
                key=lambda x: x[0])
            
            edge[2]['departure_times'] = sorted([elem[0] for elem
                                          in edge[2]['sorted_schedules']], 
                                                key=lambda x: x)


def _add_edges_to_graph(G: nx.MultiDiGraph,
                        sorted_stop_times: pd.DataFrame,
                        trips_df: pd.DataFrame,
                        shapes: dict,
                        trip_to_shape_map: dict,
                        read_shapes: bool = False
                        ):
    """
    Adds edges with schedule information and optionally shape geometry between stops to the graph.

    Parameters
    ----------
    G : nx.DiGraph
        The networkx graph to which the edges will be added.
    sorted_stop_times : pd.DataFrame
        A DataFrame containing sorted stop times information.
    trips_df : pd.DataFrame
        A DataFrame containing trip information, including shape_id.
    shapes : dict
        A dictionary mapping shape_ids to their respective linestring geometries.
    trip_to_shape_map : dict
        A dictionary mapping trip_ids to shape_ids.
    read_shapes : bool, optional
        If True, shape geometries will be added to the edges. Defaults to True.
    """
    # For each pair of consecutive stops in the group (trip),
    # create an edge with schedule information
    # If the edge already exists, add the schedule to the list of schedules

    # Uses trip_id -> shape_id mapping to add shape geometry to the edge
    # In order to avoid searching the shapes DataFrame for each trip_id

    for i in range(len(sorted_stop_times) - 1):
        start_stop = sorted_stop_times.iloc[i]
        end_stop = sorted_stop_times.iloc[i + 1]
        edge = (start_stop['stop_id'], end_stop['stop_id'])

        departure = parse_time_to_seconds(start_stop['departure_time'])
        arrival = parse_time_to_seconds(end_stop['arrival_time'])
        trip_id = start_stop['trip_id']

        # Getting route_id from trips_df
        # (searching by trip_id in trips_df and selecting route_id column)
        route_id = trips_df.loc[trips_df['trip_id']
                                == trip_id, 'route_id'
                                ].values[0]

        if 'wheelchair_accessible' in trips_df.columns:
            wheelchair_accessible = trips_df.loc[
                trips_df['trip_id'] == trip_id, 
                'wheelchair_accessible'].values[0]
        else:
            wheelchair_accessible = None

        schedule_info = (departure, arrival, route_id, wheelchair_accessible)

        geometry = None
        if read_shapes:
            shape_id = trip_to_shape_map.get(trip_id)
            if shape_id in shapes:
                geometry = shapes[shape_id]

        # if edge already exists, add schedule to the list of schedules
        # Currently the geometry is added to the first edge found

        if G.has_edge(*edge):
            G[edge[0]][edge[1]]['schedules'].append(schedule_info)
        # Create a new edge otherwise
        else:
            if read_shapes:
                G.add_edge(*edge, schedules=[schedule_info],
                           type='transit', geometry=geometry)
            else:
                G.add_edge(*edge, schedules=[schedule_info], type='transit')


def _add_edges_parallel(graph, trips_chunk, trips_df, shapes, read_shapes, trip_to_shape_map):
    """
    Adds edges to the graph for a chunk of trips.
    """
    local_graph = graph.copy()  # Make a copy for local modifications
    for trip_id, group in trips_chunk.groupby(['trip_id']):
        sorted_group = group.sort_values('stop_sequence')
        _add_edges_to_graph(local_graph,
                            sorted_group,
                            trips_df=trips_df,
                            shapes=shapes,
                            read_shapes=read_shapes,
                            trip_to_shape_map=trip_to_shape_map)
    return local_graph


def _filter_stop_times_by_time(stop_times: pd.DataFrame, departure_time: int, duration_seconds: int):
    """Filters stop_times to only include trips that occur within a specified time window."""

    stop_times['departure_time_seconds'] = stop_times['departure_time'].apply(parse_time_to_seconds)
    return stop_times[
        (stop_times['departure_time_seconds'] >= departure_time) &
        (stop_times['departure_time_seconds'] <= departure_time + duration_seconds)
    ]


def _load_GTFS(
    GTFSpath: str,
    departure_time_input: str,
    day_of_week: str,
    duration_seconds,
    read_shapes=False,
    multiprocessing=False
    ):
    """
    Loads GTFS data from the specified directory path and returns a graph and a dataframe of stops.
    The function uses parallel processing to speed up data loading.

    Parameters
    ----------
    GTFSpath : str
        Path to the directory containing GTFS data files.
    departure_time_input : str
        The departure time in 'HH:MM:SS' format.
    day_of_week : str
        Day of the week in lower case, e.g. "monday".
    duration_seconds : int
        Duration of the time window to load in seconds.
    read_shapes : bool
        Geometry reading flag, passed from feed_to_graph.

    Returns
    -------
    tuple
        A tuple containing:
            - nx.MultiDiGraph: Graph representing GTFS data.
            - pd.DataFrame: DataFrame containing stop information.
    """
    # Initialize the graph and read data files.
    G = nx.DiGraph()

    stops_df = pd.read_csv(os.path.join(GTFSpath, "stops.txt"),
                           usecols=['stop_id', 'stop_lat', 'stop_lon'])

    stop_times_df = pd.read_csv(os.path.join(GTFSpath, "stop_times.txt"),
                                usecols=['departure_time',
                                         'trip_id',
                                         'stop_id',
                                         'stop_sequence',
                                         'arrival_time'])

    trips_df = pd.read_csv(os.path.join(GTFSpath, "trips.txt"))

    routes = pd.read_csv(os.path.join(GTFSpath, "routes.txt"), 
                         usecols=['route_id', 'route_short_name'])

    # Load shapes.txt if read_shapes is True
    if read_shapes:
        if 'shapes.txt' not in os.listdir(GTFSpath):
            raise FileNotFoundError('shapes.txt not found')

        shapes_df = pd.read_csv(os.path.join(GTFSpath, "shapes.txt"))

        # Group geometry by shape_id, resulting in a Pandas Series
        # with trip_id (shape_id ?) as keys and LineString geometries as values
        shapes = shapes_df.groupby('shape_id', group_keys=False).apply(
            lambda group: LineString(group[['shape_pt_lon', 'shape_pt_lat']].values)
            )
        # Mapping trip_id to shape_id for faster lookup
        trip_to_shape_map = trips_df.set_index('trip_id')['shape_id'].to_dict()

    else:
        shapes = None
        trip_to_shape_map = None

    # Join route information to trips``
    trips_df = trips_df.merge(routes, on='route_id')

    # Check if calendar.txt exists in GTFS directory
    # If it does, filter by day of the week, otherwise raise an error
    if 'calendar.txt' in os.listdir(GTFSpath):
        calendar_df = pd.read_csv(os.path.join(GTFSpath, "calendar.txt"))
        # Filter for the day of the week
        service_ids = calendar_df[calendar_df[day_of_week] == 1]['service_id']
        trips_df = trips_df[trips_df['service_id'].isin(service_ids)]
    else:
        raise FileNotFoundError('Required file calendar.txt not found')

    # Filter stop_times to only include trips that occur 
    # within a specified time window
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
                   type='transit',
                   pos=(stop['stop_lon'], stop['stop_lat']),
                   x=stop['stop_lon'],
                   y=stop['stop_lat']
                   )

    # Track time for benchmarking
    timestamp = time.perf_counter()
    if multiprocessing:

        print('Building graph in parallel')
        # Divide filtered_stops into chunks for parallel processing
        # Use half of the available CPU logical cores
        # (likely equal to the number of physical cores)
        num_cores = int(mp.cpu_count() / 2)
        chunks = np.array_split(filtered_stops, num_cores)

        # Create a pool of processes
        with mp.Pool(processes=num_cores) as pool:
            # Create a subgraph in each process
            # Each process will return a graph with edges for a subset of trips
            # The results will be combined into a single graph
            results = pool.starmap(_add_edges_parallel,
                                   [(G, chunk, trips_df, shapes,
                                     read_shapes, trip_to_shape_map) for chunk in chunks
                                    ])

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

        print(f'Building graph in parallel complete in {time.perf_counter() - timestamp} seconds')

        # Sorting schedules for faster lookup using binary search
        _preprocess_schedules(merged_graph)

        print('Transit graph created')

        return merged_graph, stops_df

    else:

        print('Building graph in a single process')
        # Splitting trips into groups by trip_id, then iteratively processing each group
        # For each group, sort by stop_sequence, then add edges to the graph
        for trip_id, group in filtered_stops.groupby('trip_id'):
            sorted_group = group.sort_values('stop_sequence')
            _add_edges_to_graph(G,
                                sorted_group,
                                trips_df = trips_df,
                                shapes = shapes,
                                read_shapes = read_shapes,
                                trip_to_shape_map = trip_to_shape_map)

        # Sorting schedules for faster lookup using binary search
        _preprocess_schedules(G)

        print('Transit graph created')

        return G, stops_df


def _load_osm(stops, save_graphml, path)-> nx.DiGraph:
    """
    Loads OpenStreetMap data within a convex hull of stops in GTFS feed, 
    creates a street network graph, and adds walking times as edge weights.

    Returns
    -------
    G_city : networkx.MultiDigraph
        A street network graph with walking times as edge weights.
    """
    # Building a convex hull from stop coordinates for OSM loading
    boundary = gpd.GeoSeries(
        [Point(lon, lat) for lon, lat
         in zip(stops['stop_lon'], stops['stop_lat'])
         ]).unary_union.convex_hull

    print('Loading OSM graph via OSMNX')
    # Loading OSM data within the convex hull
    G_city = ox.graph_from_polygon(boundary,
                                   network_type='walk',
                                   simplify=True)
    print('Street network graph created')

    for u, v, key, data in G_city.edges(keys=True, data=True):
        attributes_to_keep = {'length', 'highway', 'name'}
        for attribute in list(data):
            if attribute not in attributes_to_keep:
                del data[attribute]

    # Adding walking times on streets
    for u, v, key, data in G_city.edges(data=True, keys = True):
        distance = data['length']

        data['weight'] = distance / 1.39
        data['type'] = 'street'
        
        u_geom = Point(G_city.nodes[u]['x'], G_city.nodes[u]['y'])
        v_geom = Point(G_city.nodes[v]['x'], G_city.nodes[v]['y'])
        data['geometry'] = LineString([u_geom, v_geom])

    for _, data in G_city.nodes(data = True):
        data['type'] = 'street'

    if save_graphml:
        ox.save_graphml(G_city, path)

    # Convert MultiDiGraph from OSMNX to DiGraph
    G_city = nx.DiGraph(G_city)

    return G_city


def feed_to_graph(
    GTFSpath: str, 
    departure_time_input: str,
    day_of_week: str,
    duration_seconds: int,
    read_shapes: bool = False,
    multiprocessing: bool = False,
    input_graph_path: str = None,
    output_graph_path: str = None,
    save_graphml: bool = False,
    load_graphml: bool = False,
) -> tuple[nx.DiGraph, pd.DataFrame]:
    """
    Creates a directed graph (DiGraph) based on General Transit Feed Specification (GTFS) and OpenStreetMap (OSM) data.

    Parameters
    ----------
    GTFSpath : str
        Path to the GTFS files.
    departure_time_input : str
        Departure time in 'HH:MM:SS' format.
    day_of_week : str
        Day of the week in lowercase (e.g., 'monday').
    duration_seconds : int
        Time period from departure for which the graph will be loaded.
    read_shapes : bool, optional
        Flag for reading geometry from shapes.txt file. Default is False.
    multiprocessing : bool, optional
        Flag for using multiprocessing. Default is False.
    input_graph_path : str, optional
        Path to the OSM graph file in GraphML format. Default is None.
    output_graph_path : str, optional
        Path for saving the OSM graph in GraphML format. Default is None.
    save_graphml : bool, optional
        Flag for saving the OSM graph in GraphML format. Default is False.
    load_graphml : bool, optional
        Flag for loading the OSM graph from a GraphML file. Default is False.

    Returns
    -------
    G_combined : nx.DiGraph
        Combined directed graph.
    stops : pd.DataFrame
        DataFrame with information about the stops.
    """
    G_transit, stops = _load_GTFS(GTFSpath, departure_time_input,
                                  day_of_week, duration_seconds,
                                  read_shapes=read_shapes,
                                  multiprocessing=multiprocessing)

    if load_graphml:
        print('Loading OSM graph from GraphML file')
        # Dictionary with data types for edges
        edge_dtypes = {'weight': float, 'length': float}
        G_city = ox.load_graphml(input_graph_path, edge_dtypes=edge_dtypes)
        G_city = nx.DiGraph(G_city)
    else:
        # Import OSM data
        G_city = _load_osm(stops, save_graphml, output_graph_path)

    # Combining OSM and GTFS data
    G_combined = nx.compose(G_transit, G_city)

    # Filling projected coordinates for graph nodes
    _fill_coordinates(G_combined)

    print("Combining graphs")
    # Connecting stops to OSM streets
    G_combined = connect_stops_to_streets(G_combined, stops)

    print(f'Number of nodes: {G_combined.number_of_nodes()}\n'
          f'Number of edges: {G_combined.number_of_edges()}\n'
          'Connecting stops to streets complete')

    return G_combined, stops
