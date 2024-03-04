import os

import geopandas as gpd
import pandas as pd
import tqdm
from shapely.geometry import Polygon


def create_grid(geodataframe, cell_size):
    """
    Creates a grid within the bounding box of a GeoDataFrame.

    Parameters
    ----------
    geodataframe : GeoDataFrame
        GeoDataFrame containing the geometry to be gridded.
    cell_size : float
        Size of the grid cells in the meters.

    Returns
    -------
    gpd.GeoDataFrame
        Polygon grid.
    """

    geodataframe = geodataframe.to_crs("EPSG:4087")  # Project to metric CRS for accurate cell size
    xmin, ymin, xmax, ymax = geodataframe.total_bounds
    rows = int((ymax - ymin) / cell_size)
    cols = int((xmax - xmin) / cell_size)
    grid = []
    ids = []  # List to hold the unique IDs

    for i in range(cols):
        for j in range(rows):
            x1 = xmin + i * cell_size
            y1 = ymin + j * cell_size
            x2 = x1 + cell_size
            y2 = y1 + cell_size
            grid.append(Polygon([(x1, y1), (x2, y1), (x2, y2), (x1, y2)]))
            ids.append(f"grid_{i*rows + j + 1}")  # Generate unique ID

    grid_geodataframe = gpd.GeoDataFrame({'id': ids, 'geometry': grid}, crs="EPSG:4087")
    
    return grid_geodataframe


def create_centroids_dataframe(polygon_gdf):
    """
    Creates a GeoDataFrame with the centroids of polygons from the given GeoDataFrame.

    Parameters
    ----------
    polygon_gdf : gpd.GeoDataFrame
        GeoDataFrame containing polygons.

    Returns
    -------
    gpd.GeoDataFrame
        GeoDataFrame with Point geometries of the centroids.
    """
    # Calculate centroids
    centroids = polygon_gdf.geometry.centroid

    # Create a GeoDataFrame with these centroids
    # and include the 'origin_id' from the parent polygon
    centroids_gdf = gpd.GeoDataFrame(polygon_gdf[['id']].copy(),
                                     geometry=centroids, crs=polygon_gdf.crs
                                     )
    centroids_gdf.rename(columns={'id': 'origin_id'}, inplace=True)

    return centroids_gdf


def validate_feed(gtfs_path: str) -> bool:
    """
    Validates the GTFS feed located at the specified path.

    Parameters
    ----------
    gtfs_path : str
        Path to the GTFS dataset directory.

    Returns
    -------
    bool
        True if the GTFS feed is valid, False otherwise.
    """

    # List of required GTFS files
    required_files = [
        "agency.txt", "stops.txt", "routes.txt",
        "trips.txt", "stop_times.txt", "calendar.txt"
    ]

    # Check for the existence of required GTFS files
    for file in required_files:
        if not os.path.isfile(os.path.join(gtfs_path, file)):
            print(f"Missing required file: {file}")
            return False

    try:
        # Load GTFS files
        agency_df = pd.read_csv(os.path.join(gtfs_path, "agency.txt"))
        stops_df = pd.read_csv(os.path.join(gtfs_path, "stops.txt"))
        routes_df = pd.read_csv(os.path.join(gtfs_path, "routes.txt"))
        trips_df = pd.read_csv(os.path.join(gtfs_path, "trips.txt"))
        stop_times_df = pd.read_csv(os.path.join(gtfs_path, "stop_times.txt"), low_memory=False)
        calendar_df = pd.read_csv(os.path.join(gtfs_path, "calendar.txt"))
        
        critical_erorrs = False

        # Validate agency.txt
        if agency_df.empty or 'agency_id' not in agency_df.columns:
            print("agency.txt is invalid or missing required 'agency_id' column.")

        # Validate stops.txt
        if stops_df.empty or 'stop_id' not in stops_df.columns:
            print("stops.txt is invalid or missing required 'stop_id' column.")
            critical_erorrs = True

        # Validate routes.txt
        if routes_df.empty or 'route_id' not in routes_df.columns or 'route_id' not in routes_df.columns:
            print("routes.txt is invalid or missing required columns (agency_id, route_id).")
            critical_erorrs = True
            
        if not set(routes_df['agency_id']).issubset(set(agency_df['agency_id'])):
            print("Mismatch in agency IDs between routes and agency files.")
            critical_erorrs = True
            
        # Validate trips.txt
        if trips_df.empty or 'trip_id' not in trips_df.columns or 'route_id' not in trips_df.columns:
            print("trips.txt is invalid or missing required columns.")
            critical_erorrs = True

        if not set(trips_df['route_id']).issubset(set(routes_df['route_id'])):
            print("Mismatch in route IDs between trips and routes files.")
            critical_erorrs = True
            
        # Validate stop_times.txt
        if stop_times_df.empty or 'trip_id' not in stop_times_df.columns or 'stop_id' not in stop_times_df.columns:
            print("stop_times.txt is invalid or missing required columns.")
            critical_erorrs = True

        if not set(stop_times_df['trip_id']).issubset(set(trips_df['trip_id'])):
            print("Mismatch in trip IDs between stop_times and trips files.")

        if not set(stop_times_df['stop_id']).issubset(set(stops_df['stop_id'])):
            print("Mismatch in stop IDs between stop_times and stops files.")

        # Validate calendar.txt
        if calendar_df.empty:
            print("calendar.txt is invalid or empty.")

        # Validate stop_times.txt for blank times and format of times
        if 'departure_time' not in stop_times_df.columns or 'arrival_time' not in stop_times_df.columns:
            print("stop_times.txt is missing required time columns.")

        # Check for blank times
        if stop_times_df['departure_time'].isnull().any() or stop_times_df['arrival_time'].isnull().any():
            print("Blank departure or arrival times found in stop_times.txt.")

        # Validate time format (HH:MM:SS)
        time_format_regex = r'^(\d{2}):([0-5]\d):([0-5]\d)$'  # check for HH:MM:SS format
        invalid_departure_times = stop_times_df[~stop_times_df['departure_time'].str.match(time_format_regex)]
        invalid_arrival_times = stop_times_df[~stop_times_df['arrival_time'].str.match(time_format_regex)]

        if not invalid_departure_times.empty or not invalid_arrival_times.empty:
            print("Invalid time format found in departure or arrival times in stop_times.txt.")
        
        # Additional format and consistency checks= will be added
     
    except Exception as e:
        print(f"Error during validation: {e}")
        return False

    if critical_erorrs:
        ("GTFS feed contains critical errors.")
        return False
    else:
        print("GTFS feed is valid.")
        return True


def _unpack_path_vertices(path):
    """
    This function separates pedestrian segments 
    of given path into list of lists
    """
    
    pedestrian_path = []
    current_sublist = []
    
    for vertex in path:
        if isinstance(vertex, int):
            current_sublist.append(vertex)
            
        # if vertex is not an integer, it means that it is the end of the current pedestrian segment
        # if current_sublist is not empty, push it to the pedestrian_path list
        elif current_sublist:
            pedestrian_path.append(current_sublist)
            current_sublist = []
    
    if current_sublist:
        pedestrian_path.append(current_sublist)
        
    return pedestrian_path


def _calculate_pedestrian_time(pedestrian_path, graph):
    """
    Calculate total impedance (travel time) for pedestrian paths by summing up the edge weights.
    """
    impedance = 0
    for subpath in pedestrian_path:
        for i in range(len(subpath) - 1):
            start_node = subpath[i]
            end_node = subpath[i+1]
            
            impedance += graph[start_node][end_node]['weight']
            
    return impedance


def _reconstruct_path(target, predecessors):
    """
    Reconstruct path from predecessors dictionary
    """

    path = []
    current_node = target

    while current_node is not None:
        path.insert(0, current_node)

        current_node = predecessors.get(current_node)

    return path


def separate_travel_times(graph, predecessors: dict, travel_times: dict, source) -> pd.DataFrame:
    """
    Separate the travel times into transit time and pedestrian time for each node in the graph.

    It calculates the pedestrian time by reconstructing the path from the source node 
    to each destination node and then estimating the time spent walking. 

    Parameters
    ----------
    graph : networkx.DiGraph
        The graph representing the transit network.
    predecessors : dict
        A dictionary containing the predecessors of each node in the graph.
    travel_times : dict
        A dictionary containing the travel times for each node in the graph.
    source : hashable
        The source node from which to calculate the travel times.

    Returns
    -------
    pandas.DataFrame
        A DataFrame containing the transit time and pedestrian time for each node.
    """
    
    results = []
    
    for node in tqdm.tqdm(graph.nodes(data = True)):
        
        if node[0] != source:

            path = _reconstruct_path(node[0], predecessors)
            pedestrian_path = _unpack_path_vertices(path)
            pedestrian_time = _calculate_pedestrian_time(pedestrian_path, graph)
            
            transit_time = travel_times[node[0]] - pedestrian_time

            results.append({'node': node[0], 'transit_time': transit_time, 'pedestrian_time': pedestrian_time})
        
    results = pd.DataFrame(results)
    return results


def process_graph_to_hash_table(graph):
    """
    Process a graph and convert it into a hash table
    mapping edges to their sorted schedules or static weights.

    Parameters
    ----------
    graph : networkx.DiGraph
        The input graph.

    Returns
    -------
    dict
        A dict mapping edges to their sorted schedules or static weights.
    """
    schedules_hash = {}
    for from_node, to_node, data in graph.edges(data=True):
        if 'sorted_schedules' in data:
            schedules_hash[(from_node, to_node)] = data['sorted_schedules']
        else:
            # Static weight wrapped in a list of tuples to make it iterable
            static_weight = data['weight']
            schedules_hash[(from_node, to_node)] = [
                (static_weight,)
            ]  # comma is to make it a tuple

    return schedules_hash
