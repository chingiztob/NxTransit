import math
import os
import warnings

import geopandas as gpd
import pandas as pd
from shapely.geometry import Polygon


def determine_utm_zone(gdf: gpd.GeoDataFrame) -> str:
    """
    Determines the UTM zone for a GeoDataFrame based on its centroid.
    
    Parameters
    ----------
    gdf : GeoDataFrame
        The input geospatial data.
    
    Returns
    -------
    epsg_code: str
        UTM EPSG code as string.
    """
    # Ensure the GeoDataFrame is in geographic coordinates (EPSG:4326)
    gdf_proj = gdf.to_crs("EPSG:4326")

    minx, miny, maxx, maxy = gdf_proj.total_bounds
    centroid_longitude = (minx + maxx) / 2

    # Determine UTM zone number
    utm_zone = math.floor((centroid_longitude + 180) / 6) + 1
    # Determine hemisphere (north or south)
    hemisphere = "north" if (miny + maxy) / 2 >= 0 else "south"
    # Construct EPSG code for UTM
    epsg_code = (
        f"EPSG:326{utm_zone}" if hemisphere == "north" else f"EPSG:327{utm_zone}"
    )

    return epsg_code


def aggregate_to_grid(gdf: gpd.GeoDataFrame, cell_size: float) -> gpd.GeoDataFrame:
    """
    Creates a grid of square cells covering the extent of the input GeoDataFrame, 
    and keeps cells that contain at least one feature from the source GeoDataFrame.
    
    Parameters
    ----------
    gdf : gpd.GeoDataFrame
        The input GeoDataFrame representing the spatial extent and features.
    cell_size : float
        The size of each square cell in the grid in meters.
    
    Returns
    -------
    gpd.GeoDataFrame
        The resulting grid GeoDataFrame, with cells containing at least
        one feature from the source GeoDataFrame, and a 'id' for each cell.
    """
    utm_crs = determine_utm_zone(gdf)
    gdf_utm = gdf.to_crs(utm_crs)
    minx, miny, maxx, maxy = gdf_utm.total_bounds

    nx = math.ceil((maxx - minx) / cell_size)
    ny = math.ceil((maxy - miny) / cell_size)
    grid_cells = []
    grid_indices = []
    index = 0  # Initialize a counter for the grid index
    for i in range(nx):
        for j in range(ny):
            x1 = minx + i * cell_size
            y1 = miny + j * cell_size
            x2 = x1 + cell_size
            y2 = y1 + cell_size
            cell = Polygon([(x1, y1), (x2, y1), (x2, y2), (x1, y2)])
            grid_cells.append(cell)
            grid_indices.append(f"grid_{index}")  # Add the current index to the list
            index += 1  # Increment the index for the next cell

    # Create the initial grid GeoDataFrame
    grid = gpd.GeoDataFrame({'id': grid_indices, 'geometry': grid_cells}, crs=utm_crs)

    # Perform a spatial join between the grid and the original GeoDataFrame
    filtered_grid = gpd.sjoin(grid, gdf_utm[['geometry']], how='inner')

    # Drop duplicates to ensure each cell is unique, keeping only 'geometry' and 'grid_index'
    filtered_grid = filtered_grid[['geometry', 'id']].drop_duplicates(subset=['id'])
    filtered_grid.reset_index(drop=True, inplace=True)

    return filtered_grid


def create_centroids_dataframe(polygon_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
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
    # Create id column if it doesn't exist
    # ID is required for the explicit origin_id column in the final output
    if "id" not in polygon_gdf.columns:
        polygon_gdf["id"] = polygon_gdf.index
    # Create a GeoDataFrame with these centroids
    # and include the 'origin_id' from the parent polygon
    centroids_gdf = gpd.GeoDataFrame(
        polygon_gdf[["id"]].copy(),
        geometry=polygon_gdf.to_crs("EPSG:4087").geometry.centroid,
        crs="EPSG:4087",
    )
    centroids_gdf.rename(columns={"id": "origin_id"}, inplace=True)

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
    if not os.path.isdir(gtfs_path):
        warnings.warn("Invalid GTFS path.")
        return False

    # List of required GTFS files
    required_files = [
        "agency.txt", "stops.txt", "routes.txt",
        "trips.txt", "stop_times.txt", "calendar.txt"
    ]

    # Check for the existence of required GTFS files
    for file in required_files:
        if not os.path.isfile(os.path.join(gtfs_path, file)):
            warnings.warn(f"Missing required file: {file}")
            return False

    try:
        # Load GTFS files
        agency_df = pd.read_csv(os.path.join(gtfs_path, "agency.txt"))
        stops_df = pd.read_csv(os.path.join(gtfs_path, "stops.txt"))
        routes_df = pd.read_csv(os.path.join(gtfs_path, "routes.txt"))
        trips_df = pd.read_csv(os.path.join(gtfs_path, "trips.txt"))
        stop_times_df = pd.read_csv(os.path.join(gtfs_path, "stop_times.txt"), low_memory=False)
        calendar_df = pd.read_csv(os.path.join(gtfs_path, "calendar.txt"))
        
        critical_errors = False

        # Validate agency.txt
        if agency_df.empty or 'agency_id' not in agency_df.columns:
            print("agency.txt is invalid or missing required 'agency_id' column.")

        # Validate stops.txt
        if stops_df.empty or 'stop_id' not in stops_df.columns:
            print("stops.txt is invalid or missing required 'stop_id' column.")
            critical_errors = True

        # Validate routes.txt
        if routes_df.empty or 'route_id' not in routes_df.columns or 'route_id' not in routes_df.columns:
            print("routes.txt is invalid or missing required columns (agency_id, route_id).")
            critical_errors = True
            
        if not set(routes_df['agency_id']).issubset(set(agency_df['agency_id'])):
            print("Mismatch in agency IDs between routes and agency files.")
            critical_errors = True
            
        # Validate trips.txt
        if trips_df.empty or 'trip_id' not in trips_df.columns or 'route_id' not in trips_df.columns:
            print("trips.txt is invalid or missing required columns.")
            critical_errors = True

        if not set(trips_df['route_id']).issubset(set(routes_df['route_id'])):
            print("Mismatch in route IDs between trips and routes files.")
            critical_errors = True
            
        # Validate stop_times.txt
        if stop_times_df.empty or 'trip_id' not in stop_times_df.columns or 'stop_id' not in stop_times_df.columns:
            print("stop_times.txt is invalid or missing required columns.")
            critical_errors = True

        if not set(stop_times_df['trip_id']).issubset(set(trips_df['trip_id'])):
            print("Mismatch in trip IDs between stop_times and trips files.")
            critical_errors = True

        if not set(stop_times_df['stop_id']).issubset(set(stops_df['stop_id'])):
            print("Mismatch in stop IDs between stop_times and stops files.")
            critical_errors = True

        # Validate calendar.txt
        if calendar_df.empty:
            print("calendar.txt is invalid or empty.")
            critical_errors = True

        # Validate stop_times.txt for blank times and format of times
        if 'departure_time' not in stop_times_df.columns or 'arrival_time' not in stop_times_df.columns:
            print("stop_times.txt is missing required time columns.")
            critical_errors = True

        # Check for blank times
        if stop_times_df['departure_time'].isnull().any() or stop_times_df['arrival_time'].isnull().any():
            print("Blank departure or arrival times found in stop_times.txt.")

        # Validate time format (HH:MM:SS)
        time_format_regex = r'^(\d{2}):([0-5]\d):([0-5]\d)$'  # check for HH:MM:SS format
        invalid_departure_times = stop_times_df[~stop_times_df['departure_time'].str.match(time_format_regex)]
        invalid_arrival_times = stop_times_df[~stop_times_df['arrival_time'].str.match(time_format_regex)]

        if not invalid_departure_times.empty or not invalid_arrival_times.empty:
            print("Invalid time format found in departure or arrival times in stop_times.txt.")
            print(f"Invalid departure times: {invalid_departure_times['departure_time'].values}")
            print(f"Invalid arrival times: {invalid_arrival_times['arrival_time'].values}")
        
        # Additional format and consistency checks= will be added
     
    except Exception as e:
        print(f"Error during validation: {e}")
        return False

    if critical_errors:
        print("GTFS feed contains critical errors.")
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
    
    # Transit verteces are always float or string (idk why lol)
    # while pedestrian verteces (osmid) are integers
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
    for node in graph.nodes(data=True):
        if node[0] != source:
            path = _reconstruct_path(node[0], predecessors)
            pedestrian_path = _unpack_path_vertices(path)
            pedestrian_time = _calculate_pedestrian_time(pedestrian_path, graph)

            transit_time = travel_times[node[0]] - pedestrian_time
            results.append(
                {
                    "node": node[0],
                    "transit_time": transit_time,
                    "pedestrian_time": pedestrian_time,
                }
            )

    results = pd.DataFrame(results)
    return results


def process_graph_to_hash_table(graph) -> dict:
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
        if "sorted_schedules" in data:
            schedules_hash[(from_node, to_node)] = data["sorted_schedules"]
        else:
            # Static weight wrapped in a list of tuples to make it iterable
            static_weight = data["weight"]
            schedules_hash[(from_node, to_node)] = [
                (static_weight,)
            ]  # comma is to make it a tuple

    return schedules_hash
