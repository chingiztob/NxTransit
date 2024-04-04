"""Load combined GTFS and OSM data into a graph."""
import multiprocessing as mp
import os
from functools import partial

import geopandas as gpd
import networkx as nx
import osmnx as ox
import pandas as pd
from shapely.geometry import LineString, Point

from .connectors import _fill_coordinates, connect_stops_to_streets
from .converters import parse_time_to_seconds
from .functions import validate_feed
from .other import logger


def _preprocess_schedules(graph: nx.DiGraph):
    """
    Sorts the schedules on each edge for faster lookup.
    """
    for _, _, data in graph.edges(data=True):
        if "schedules" in data:
            sorted_schedules = sorted(data["schedules"], key=lambda x: x[0])
            data["sorted_schedules"] = sorted_schedules
            data["departure_times"] = [schedule[0] for schedule in sorted_schedules]


def _add_edge_with_geometry(graph, start_stop, end_stop, schedule_info, geometry):
    """
    Adds or updates an edge in the graph with schedule information and geometry.
    """
    edge = (start_stop["stop_id"], end_stop["stop_id"])
    if graph.has_edge(*edge):
        graph[edge[0]][edge[1]]["schedules"].append(schedule_info)
        if "geometry" not in graph[edge[0]][edge[1]]:
            graph[edge[0]][edge[1]]["geometry"] = geometry
    else:
        graph.add_edge(*edge, schedules=[schedule_info], type="transit", geometry=geometry)


def _process_trip_group(
    group, graph, trips_df, shapes, trip_to_shape_map, stops_df, read_shapes
):
    """
    Processes a group of sorted stops for a single trip, adding edges between them to the graph.

    Parameters
    ----------
    group : pd.DataFrame
        A group of sorted stops for a single trip.
    graph : networkx.DiGraph
        The graph to which the edges will be added.
    trips_df : pd.DataFrame
        DataFrame containing trip information.
    shapes : dict
        Dictionary mapping shape IDs to shape geometries.
    trip_to_shape_map : dict
        Dictionary mapping trip IDs to shape IDs.
    stops_df : pd.DataFrame
        DataFrame containing stop information.
    read_shapes : bool
        Flag indicating whether to read shape geometries from shapes.txt.

    Returns
    -------
    None
    """
    # Mapping stop_id to coordinates for faster lookup
    stop_coords_mapping = stops_df.set_index("stop_id")[["stop_lat", "stop_lon"]].to_dict("index")
    trip_route_mapping = trips_df.set_index("trip_id")["route_id"].to_dict()

    # Some GTFS feeds do not have wheelchair_accessible information
    if "wheelchair_accessible" in trips_df.columns:
        trip_wheelchair_mapping = trips_df.set_index("trip_id")["wheelchair_accessible"].to_dict()
    else:
        trip_wheelchair_mapping = {}

    # For each pair of consecutive stops in the group, add an edge to the graph
    for i in range(len(group) - 1):
        start_stop, end_stop = group.iloc[i], group.iloc[i + 1]
        departure, arrival = (
            parse_time_to_seconds(start_stop["departure_time"]),
            parse_time_to_seconds(end_stop["arrival_time"]),
        )
        if departure > arrival:
            raise ValueError(
                f"Departure time {departure} is greater than arrival time {arrival} for edge {start_stop['stop_id']} -> {end_stop['stop_id']}\n"
                "Negative travel time not allowed\n"
                "Check the GTFS feed for errors in stop_times.txt or calendar.txt, or adjust the departure time\n"
            )

        trip_id = start_stop["trip_id"]
        route_id = trip_route_mapping.get(trip_id)
        wheelchair_accessible = trip_wheelchair_mapping.get(trip_id, None)
        schedule_info = (departure, arrival, route_id, wheelchair_accessible)

        # If read_shapes is True, use the shape geometry from shapes.txt
        geometry = None
        if read_shapes:
            shape_id = trip_to_shape_map.get(trip_id)
            geometry = shapes.get(shape_id)
        # Otherwise, use the stop coordinates to create a simple LineString geometry
        else:
            start_coords, end_coords = (
                stop_coords_mapping.get(start_stop["stop_id"]),
                stop_coords_mapping.get(end_stop["stop_id"]),
            )
            geometry = LineString(
                [
                    (start_coords["stop_lon"], start_coords["stop_lat"]),
                    (end_coords["stop_lon"], end_coords["stop_lat"]),
                ]
            )

        _add_edge_with_geometry(
            graph=graph,
            start_stop=start_stop,
            end_stop=end_stop,
            schedule_info=schedule_info,
            geometry=geometry,
        )


def _add_edges_parallel(
    trips_chunks, graph, trips_df, shapes, read_shapes, trip_to_shape_map, stops_df
):
    """
    Adds edges to the graph for chunks of trips in parallel.
    """
    local_graph = graph.copy()
    for _, group in trips_chunks.groupby(["trip_id"]):
        sorted_group = group.sort_values("stop_sequence")
        _process_trip_group(
            group=sorted_group,
            graph=local_graph,
            trips_df=trips_df,
            shapes=shapes,
            trip_to_shape_map=trip_to_shape_map,
            stops_df=stops_df,
            read_shapes=read_shapes,
        )
    return local_graph


def _filter_stop_times_by_time(stop_times: pd.DataFrame, departure_time: int, duration_seconds: int):
    """Filters stop_times to only include trips that occur within a specified time window."""

    stop_times['departure_time_seconds'] = stop_times['departure_time'].apply(parse_time_to_seconds)
    return stop_times[
        (stop_times['departure_time_seconds'] >= departure_time) &
        (stop_times['departure_time_seconds'] <= departure_time + duration_seconds)
    ]


def _split_dataframe(df: pd.DataFrame, n_splits: int) -> list[pd.DataFrame]:
    """
    Splits a DataFrame into n equal parts by rows.
    This function replaces np.split_array which will be deprecated soon.
    
    Parameters
    ----------
    df : pandas DataFrame
        The DataFrame to be split.
    n_splits : int
        The number of parts to split the DataFrame into.
    
    Returns
    -------
    list of pandas DataFrames
        A list of DataFrame parts.
    """
    # Calculate split sizes
    total_rows = len(df)
    base_size = total_rows // n_splits
    remainder = total_rows % n_splits

    # Determine the number of rows each split will have
    split_sizes = [
        base_size + 1 if i < remainder 
        else base_size for i in range(n_splits)
    ]
    # Calculate the start indices for each split
    start_indices = [sum(split_sizes[:i]) for i in range(n_splits)]

    return [
        df.iloc[start : start + size] for start, size 
        in zip(start_indices, split_sizes)
    ]


def _load_GTFS(
        GTFSpath: str,
        departure_time_input: str,
        day_of_week: str,
        duration_seconds,
        read_shapes=False,
        multiprocessing=False
) -> tuple[nx.DiGraph, pd.DataFrame]:
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
            - nx.DiGraph: Graph representing GTFS data.
            - pd.DataFrame: DataFrame containing stop information.
    """
    # Initialize the graph and read data files.
    G = nx.DiGraph()
    stops_df = pd.read_csv(
        os.path.join(GTFSpath, "stops.txt"), 
        usecols=["stop_id", "stop_lat", "stop_lon"]
    )
    stop_times_df = pd.read_csv(
        os.path.join(GTFSpath, "stop_times.txt"),
        usecols=["departure_time", "trip_id", "stop_id", "stop_sequence", "arrival_time"]
    )
    routes = pd.read_csv(
        os.path.join(GTFSpath, "routes.txt"), usecols=["route_id", "route_short_name"]
    )
    trips_df = pd.read_csv(os.path.join(GTFSpath, "trips.txt"))
    calendar_df = pd.read_csv(os.path.join(GTFSpath, "calendar.txt"))
    
    # Load shapes.txt if read_shapes is True
    if read_shapes:
        logger.warning("Reading shapes is currently not working as intended")
        if "shapes.txt" not in os.listdir(GTFSpath):
            raise FileNotFoundError("shapes.txt not found")

        shapes_df = pd.read_csv(os.path.join(GTFSpath, "shapes.txt"))
        # Group geometry by shape_id, resulting in a Pandas Series
        # with trip_id (shape_id ?) as keys and LineString geometries as values
        # This is definitely not working as intended
        shapes = shapes_df.groupby("shape_id")[["shape_pt_lon", "shape_pt_lat"]].apply(
            lambda group: LineString(group.values)
        )
        # Mapping trip_id to shape_id for faster lookup
        trip_to_shape_map = trips_df.set_index("trip_id")["shape_id"].to_dict()

    else:
        shapes = None
        trip_to_shape_map = None

    # Join route information to trips
    trips_df = trips_df.merge(routes, on="route_id")
    # Filter trips by day of the week
    service_ids = calendar_df[calendar_df[day_of_week] == 1]["service_id"]
    trips_df = trips_df[trips_df["service_id"].isin(service_ids)]

    # Filter stop_times by valid trips
    valid_trips = stop_times_df['trip_id'].isin(trips_df['trip_id'])
    stop_times_df = stop_times_df[valid_trips].dropna()

    # Convert departure_time from HH:MM:SS o seconds
    departure_time_seconds = parse_time_to_seconds(departure_time_input)
    # Filtering stop_times by time window
    filtered_stops = _filter_stop_times_by_time(
        stop_times_df, departure_time_seconds, duration_seconds
    )

    print(f'{len(filtered_stops)} of {len(stop_times_df)} trips retained')

    # Adding stops as nodes to the graph
    for _, stop in stops_df.iterrows():
        G.add_node(
            stop["stop_id"],
            type="transit",
            pos=(stop["stop_lon"], stop["stop_lat"]),
            x=stop["stop_lon"],
            y=stop["stop_lat"],
        )

    if multiprocessing:
        print("Building graph in parallel")
        # Divide filtered_stops into chunks for parallel processing
        # Use half of the available CPU logical cores
        # (likely equal to the number of physical cores)
        num_cores = int(mp.cpu_count() / 2) if mp.cpu_count() > 1 else 1
        chunks = _split_dataframe(filtered_stops, num_cores)

        # Create a pool of processes
        with mp.Pool(processes=num_cores) as pool:
            # Create a subgraph in each process
            # Each will return a graph with edges for a subset of trips
            # The results will be combined into a single graph
            add_edges_partial = partial(_add_edges_parallel,
                graph=G,
                trips_df=trips_df,
                shapes=shapes,
                read_shapes=read_shapes,
                trip_to_shape_map=trip_to_shape_map,
                stops_df=stops_df
            )
            results = pool.map(add_edges_partial, chunks)

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
                    existing_schedules = merged_graph[u][v]["schedules"]
                    new_schedules = data["schedules"]
                    merged_graph[u][v]["schedules"] = existing_schedules + new_schedules
                # If edge does not exist, add it
                else:
                    # Add new edge with data
                    merged_graph.add_edge(u, v, **data)

        # Sorting schedules for faster lookup using binary search
        _preprocess_schedules(merged_graph)
        logger.info("Transit graph created")

        return merged_graph, stops_df

    else:
        for trip_id, group in filtered_stops.groupby("trip_id"):
            sorted_group = group.sort_values("stop_sequence")
            _process_trip_group(
                group=sorted_group,
                graph=G,
                trips_df=trips_df,
                shapes=shapes,
                trip_to_shape_map=trip_to_shape_map,
                stops_df=stops_df,
                read_shapes=read_shapes,
            )

        # Sorting schedules for faster lookup using binary search
        _preprocess_schedules(graph=G)
        logger.info("Transit graph created")

        return G, stops_df


def _load_osm(stops, save_graphml, path) -> nx.DiGraph:
    """
    Loads OpenStreetMap data within a convex hull of stops in GTFS feed, 
    creates a street network graph, and adds walking times as edge weights.

    Parameters
    ----------
    stops : pandas.DataFrame
        DataFrame containing the stops information from the GTFS feed.
    save_graphml : bool
        Flag indicating whether to save the resulting graph as a GraphML file.
    path : str
        The file path to save the GraphML file (if save_graphml is True).

    Returns
    -------
    G_city : networkx.DiGraph
        A street network graph with walking times as edge weights.
    """
    # Building a convex hull from stop coordinates for OSM loading
    stops_gdf = gpd.GeoDataFrame(
        stops, geometry=gpd.points_from_xy(stops.stop_lon, stops.stop_lat)
    )
    boundary = stops_gdf.unary_union.convex_hull

    logger.info("Loading OSM graph via OSMNX")
    # Loading OSM data within the convex hull
    G_city = ox.graph_from_polygon(boundary, network_type="walk", simplify=True)

    attributes_to_keep = {"length", "highway", "name"}
    for u, v, key, data in G_city.edges(keys=True, data=True):
        # Clean extra attributes
        for attribute in list(data):
            if attribute not in attributes_to_keep:
                del data[attribute]

        # Calculate walking time in seconds
        data["weight"] = data["length"] / 1.39
        data["type"] = "street"

        # Add geometry to the edge
        u_geom = Point(G_city.nodes[u]["x"], G_city.nodes[u]["y"])
        v_geom = Point(G_city.nodes[v]["x"], G_city.nodes[v]["y"])
        data["geometry"] = LineString([u_geom, v_geom])

    nx.set_node_attributes(G_city, "street", "type")

    if save_graphml:
        ox.save_graphml(G_city, path)

    logger.info("Street network graph created")

    return nx.DiGraph(G_city)


def feed_to_graph(
    GTFSpath: str, 
    departure_time_input: str,
    day_of_week: str,
    duration_seconds: int,
    read_shapes: bool = False,
    multiprocessing: bool = True,
    input_graph_path: str = None,
    output_graph_path: str = None,
    save_graphml: bool = False,
    load_graphml: bool = False,
) -> nx.DiGraph:
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
        Flag for reading geometry from shapes.txt file. Default is False. This parameter is currently not working as intended.
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
        Combined multimodal graph representing transit network.
    """
    # Validate the GTFS feed
    bool_feed_valid = validate_feed(GTFSpath)
    if not bool_feed_valid:
        raise ValueError("The GTFS feed is not valid")
    
    G_transit, stops = _load_GTFS(
        GTFSpath,
        departure_time_input,
        day_of_week,
        duration_seconds,
        read_shapes=read_shapes,
        multiprocessing=multiprocessing,
    )

    if load_graphml:
        print("Loading OSM graph from GraphML file")
        # Dictionary with data types for edges
        edge_dtypes = {"weight": float, "length": float}
        G_city = ox.load_graphml(input_graph_path, edge_dtypes=edge_dtypes)
        G_city = nx.DiGraph(G_city)
    else:
        # Import OSM data
        G_city = _load_osm(stops, save_graphml, output_graph_path)

    # Combining OSM and GTFS data
    G_combined = nx.compose(G_transit, G_city)
    # Filling projected coordinates for graph nodes
    _fill_coordinates(G_combined)
    # Connecting stops to OSM streets
    connect_stops_to_streets(G_combined, stops)

    logger.info(
        f"Nodes: {G_combined.number_of_nodes()}, Edges: {G_combined.number_of_edges()}"
    )

    return G_combined


def load_stops_gdf(path) -> gpd.GeoDataFrame:
    """
    Load stops data from a specified path and return a GeoDataFrame.

    Parameters
    ----------
    path: str
        The path to the directory containing the stops data.

    Returns
    -------
    stops_gdf: gpd.GeoDataFrame
        GeoDataFrame containing the stops data with geometry information.

    """
    stops_df = pd.read_csv(os.path.join(path, "stops.txt"))
    stops_gdf = gpd.GeoDataFrame(
        stops_df,
        geometry=gpd.points_from_xy(stops_df.stop_lon, stops_df.stop_lat),
        crs="epsg:4326",
    )
    return stops_gdf