"""Tools for calculating accessibility metrics"""

import multiprocessing
import time
from functools import partial
from typing import Any, Dict, Optional

import geopandas as gpd
import pandas as pd
import xarray as xr
from geocube.api.core import make_geocube
from geocube.vector import vectorize
from networkx import DiGraph
from shapely.geometry import Point

from .functions import determine_utm_zone
from .routers import single_source_time_dependent_dijkstra


def calculate_od_matrix(
    graph: DiGraph, nodes: list, departure_time: float, hashtable: dict = None, algorithm="sorted"
):
    """
    Calculates the Origin-Destination (OD) matrix for a given graph, nodes, and departure time.

    Parameters
    ----------
    graph : networkx.DiGraph
        The graph representing the transit network.
    nodes : list
        A list of node IDs in the graph.
    departure_time : float
        The departure time in seconds since midnight.
    hashtable: dict, optional
        Hash table for the graph.
    algorithm: str, optional
        Algorithm to use for the OD matrix calculation (default: 'sorted').

    Returns
    -------
    pandas.DataFrame
        A DataFrame containing the OD matrix with the following columns:
            - source_node: The ID of the origin node.
            - destination_node: The ID of the destination node.
            - arrival_time: The arrival time at the destination node in seconds since midnight.
            - travel_time: The travel time from the origin node to the destination node in seconds.
    """

    results = []

    for source_node in nodes:
        # Calculate arrival times and travel times
        # for each node using the specified algorithm
        arrival_times, _, travel_times = single_source_time_dependent_dijkstra(
            graph, source_node, departure_time, hashtable, algorithm=algorithm
        )

        # Iterate through all nodes to select them
        # in the results of Dijkstra's algorithm
        for dest_node in nodes:
            if dest_node in arrival_times:
                # Add results to the list
                results.append(
                    {
                        "source_node": source_node,
                        "destination_node": dest_node,
                        "arrival_time": arrival_times[dest_node],
                        "travel_time": travel_times.get(dest_node, None)
                    }
                )

    # Convert the list of results to a DataFrame
    results_df = pd.DataFrame(results)

    return results_df


def _calculate_od_worker(
    source_node, nodes_list, graph, departure_time, hashtable=None
):
    """
    Internal worker function to calculate the OD matrix for a single source node.
    """

    if hashtable:
        arrival_times, _, travel_times = single_source_time_dependent_dijkstra(
            graph, source_node, departure_time, hashtable, algorithm="hashed"
        )
    else:
        arrival_times, _, travel_times = single_source_time_dependent_dijkstra(
            graph, source_node, departure_time, algorithm="sorted"
        )

    return [
        {
            "source_node": source_node,
            "destination_node": dest_node,
            "arrival_time": arrival_times[dest_node],
            "travel_time": travel_times.get(dest_node, None),
        }
        for dest_node in nodes_list
        if dest_node in arrival_times
    ]


def calculate_od_matrix_parallel(
    graph: DiGraph,
    nodes,
    departure_time: float,
    target_nodes: list = None,
    num_processes: int = 2,
    hashtable: dict = None,
) -> pd.DataFrame:
    """
    Calculates the Origin-Destination (OD) matrix for a given graph,
    nodes, and departure time using parallel processing.

    Parameters
    ----------
    graph : networkx.DiGraph
        The graph representing the transit network.
    nodes : list
        A list of node IDs in the graph.
    departure_time : int
        The departure time in seconds since midnight.
    target_nodes: list, optional
        A list of target node IDs in the graph. If not specified, source nodes are used.
    num_processes : int
        Number of parallel processes to use for computation.
    hashtable : dict, optional
        Optional hash table for the graph.

    Returns
    -------
    results_df : pandas.DataFrame
        A DataFrame containing the OD matrix.
    """
    print(f"Calculating the OD using {num_processes} processes")
    time_start = time.perf_counter()

    if not target_nodes:
        target_nodes = nodes

    with multiprocessing.Pool(processes=num_processes) as pool:
        # Fix the arguments of the calculate_OD_worker function for nodes list
        partial_worker = partial(
            _calculate_od_worker,
            nodes_list=target_nodes,
            graph=graph,
            departure_time=departure_time,
            hashtable=hashtable,
        )
        results = pool.map(partial_worker, nodes)

    print(f"Time elapsed: {time.perf_counter() - time_start}")
    # Return flattened list of lists
    return pd.DataFrame([item for sublist in results for item in sublist])


def service_area(
    graph: DiGraph,
    source: Any,
    start_time: int,
    cutoff: float,
    buffer_radius: float = 100,
    algorithm: str = "sorted",
    hashtable: Optional[Dict] = None,
) -> gpd.GeoDataFrame:
    """
    Creates a service area by buffering around all street edges within a travel time cutoff.

    Parameters
    ----------
    graph : networkx.DiGraph
        The graph to search.
    source : node
        The graph node to start the search from.
    start_time : int
        The time to start the search from.
    cutoff : float
        The travel time cutoff for including nodes in the service area.
    buffer_radius : float
        The radius in meters for buffering around each point.
    algorithm : str, optional
        Algorithm to use for the service area calculation (default: "sorted").
    hashtable : dict, optional
        Hashtable required for the "hashed" algorithm.

    Returns
    -------
    geopandas.GeoDataFrame
        A GeoDataFrame containing the service area polygon.

    Notes
    -----
    Output crs is EPSG:4087 (World Equidistant Cylindrical).

    See Also
    --------
    nxtransit.accessibility.percent_access_service_area : Service area reachable with specified chance
    nxtransit.accessibility.service_area_multiple_sources : Service areas for multiple sources using multiprocessing
    nxtransit.functions.determine_utm_zone : Determine the UTM zone of a GeoDataFrame.
    """

    _, _, travel_times = single_source_time_dependent_dijkstra(
        graph, source, start_time, hashtable, algorithm
    )

    # Filter nodes that are reachable within the cutoff
    points_data = [
        {
            "node": node,
            "geometry": Point(graph.nodes[node]["x"], graph.nodes[node]["y"]),
            "travel_time": travel_time,
        }
        for node, travel_time in travel_times.items()
        if travel_time <= cutoff
        and "x" in graph.nodes[node]
        and "y" in graph.nodes[node]
    ]

    reached_nodes = set([data["node"] for data in points_data])

    # Filter edges so that both nodes are reached
    reached_edges = [
        {"edge": (u, v), "geometry": data["geometry"]}
        for u, v, data in graph.edges(data=True)
        if u in reached_nodes
        and v in reached_nodes
        and "geometry" in data
        and data["type"] == "street"
    ]

    edges_gdf = gpd.GeoDataFrame(reached_edges, geometry="geometry", crs="EPSG:4326")
    utm_crs = determine_utm_zone(edges_gdf)
    # Re-projection to World Equidistant Cylindrical (EPSG:4087) for buffering in meters
    buffer_gdf = edges_gdf.to_crs(crs=utm_crs).buffer(buffer_radius)

    service_area_polygon = buffer_gdf.unary_union
    # overlap_count is needed for percent_access calculation
    service_area_gdf = gpd.GeoDataFrame(
        {"geometry": [service_area_polygon], "id": source, "overlap_count": 1},
        crs=utm_crs,
    ).to_crs("EPSG:4087")
    return service_area_gdf


def _rasterize_service_areas(service_areas, threshold, resolution=(100, 100)):
    """
    Rasterize given service area GeoDataFrames and summarize intersections.

    Parameters
    ----------
    service_areas : list
        List of GeoDataFrames representing service areas.
    threshold : float
        Threshold value for filtering polygons.
    resolution : tuple, optional
        Resolution of the raster. Defaults to (100, 100).

    Returns
    -------
    result_gdf : GeoDataFrame
        GeoDataFrame containing the vectorized result.
    """
    rasters = []
    # Rasterize each service area and append to the list
    for sa in service_areas:
        # Rasterize each service area
        cube = make_geocube(
            vector_data=sa,
            resolution=resolution,
            measurements=["overlap_count"],
        )
        rasters.append(cube)

    # Combine the rasters into 3-dimensional xarray DataSet
    # Then sum the values along the 'summary' dimension
    summarized_raster = xr.concat(rasters, dim="summary").sum(dim="summary")
    # Vectorize the summarized raster back into a GeoDataFrame
    vectorized_result = vectorize(summarized_raster.overlap_count.astype("float32"))

    # Filter the vectorized result to include only polygons that cover at least the threshold of the service areas
    polygons_needed = int(len(service_areas) * threshold)
    vectorized_result = vectorized_result[
        vectorized_result["overlap_count"] >= polygons_needed
    ].unary_union
    result_gdf = gpd.GeoDataFrame({"geometry": [vectorized_result]}, crs="EPSG:4087")

    return result_gdf


def percent_access_service_area(
    graph: DiGraph,
    source: Any,
    start_time: int,
    end_time: int,
    sample_interval: int,
    cutoff: int,
    buffer_radius: float,
    threshold: float,
    algorithm: str = "sorted",
    hashtable: Optional[Dict] = None,
) -> gpd.GeoDataFrame:
    """
    Calculate service area reachable with specified chance within the given time period.

    This tool rasterize service areas for each time step and overlays them.
    Part of the raster that is covered by at least the threshold of
    the service areas is returned as a vectorized GeoDataFrame.

    Parameters
    ----------
    graph : networkx.DiGraph
        The graph representing the transit network.
    source : Any
        The source node from which to service areas.
    start_time : int
        The start time of the time period in seconds from midnight.
    end_time : int
        The end time of the time period in seconds from midnight.
    sample_interval : int
        The interval between samples.
    cutoff : int
        The maximum travel time allowed to reach the service area.
    buffer_radius : float
        The radius of the buffer around the service area.
    threshold : float
        The threshold value for rasterizing the service areas.
    algorithm : str
        optional. Algorithm to use for the service area calculation (default: 'sorted').
    hashtable : dict, optional
        Hashtable required for the algorithm (default: None).

    Returns
    -------
    gpd.GeoDataFrame
        GeoDataFrame containing the vectorized result.
    """

    service_areas = [
        service_area(
            graph=graph,
            source=source,
            start_time=timestamp,
            cutoff=cutoff,
            buffer_radius=buffer_radius,
            algorithm=algorithm,
            hashtable=hashtable,
        )
        for timestamp in range(start_time, end_time, sample_interval)
    ]

    return _rasterize_service_areas(service_areas=service_areas, threshold=threshold)


def service_area_multiple_sources(
    graph: DiGraph,
    sources: list,
    start_time: int,
    cutoff: int,
    buffer_radius: float,
    algorithm: str = "sorted",
    hashtable: Optional[Dict] = None,
    num_processes: int = 6,
) -> gpd.GeoDataFrame:
    """
    Calculates service areas for multiple sources using multiprocessing, returning a combined service area polygon.

    Parameters
    ----------
    graph : networkx.DiGraph
        NetworkX graph representing the transportation network.
    sources : list
        List of source nodes from which to calculate service areas.
    start_time : int
        Start time for the service area calculation.
    cutoff : int
        Maximum travel time or distance for the service area.
    buffer_radius : float
        Radius to buffer the service area polygons.
    algorithm : str, optional
        Algorithm to use for the service area calculation (default: 'sorted').
    hashtable : dict, optional
        Hashtable to store calculated service areas (default: None).
    num_processes : int, optional
        Number of processes to use for parallel execution (default: 6).

    Returns
    -------
    combined_service_area : GeoDataFrame
        A GeoDataFrame containing the combined service area polygon for all sources.
    """
    # Prepare arguments for each task
    tasks = [
        (graph, source, start_time, cutoff, buffer_radius, algorithm, hashtable)
        for source in sources
    ]

    with multiprocessing.Pool(processes=num_processes) as pool:
        results = pool.starmap(service_area, tasks)

    # At this point, 'results' is a list of GeoDataFrames, each containing the service area polygon for a source
    # Combine all service area polygons into a single GeoDataFrame
    combined_service_area = gpd.GeoDataFrame(
        pd.concat(results, ignore_index=True), crs="EPSG:4087"
    )

    return combined_service_area


def last_service(graph: DiGraph):
    """
    Calculate the last service time for each stop in the graph.
    Populates the 'last_service' attribute of each transit stop.

    Parameters
    ----------
    graph : networkx.DiGraph
        The graph representing the transit network.

    Returns
    -------
    None
    """
    for node, data in graph.nodes(data=True):
        if data["type"] == "transit":
            last_service_time = float("-inf")

            for _, _, edge_data in graph.edges(node, data=True):
                if "sorted_schedules" in edge_data:
                    # Get the last arrival and departure times
                    # from the sorted schedules
                    last_arrival = edge_data["sorted_schedules"][-1][0]
                    last_departure = edge_data["sorted_schedules"][-1][1]
                    # Update the last service time if the current edge
                    # has a later service time
                    last_service_time = max(
                        last_service_time, last_arrival, last_departure
                    )

            # if the stop is not serviced, set last_service_time to None
            if last_service_time == float("-inf"):
                last_service_time = None

            data["last_service"] = last_service_time
