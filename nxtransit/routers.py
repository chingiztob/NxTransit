"""Main routing algorithms for time-dependent graphs."""
from heapq import heappop, heappush
import bisect


def _calculate_delay_sorted_nr(graph, from_node, to_node, current_time, wheelchair=False):
    """
    Calculates the delay and route for a given graph, from_node, to_node, and current_time.
    Used in the time-dependent Dijkstra algorithm.
    This version does not return the route, and is used for the single_source_time_dependent_dijkstra_sorted function.
    """
    edge = graph[from_node][to_node]
    if 'sorted_schedules' in edge:
        schedules = edge['sorted_schedules']
        departure_times = edge['departure_times']
        idx = bisect.bisect_left(departure_times, current_time)
        
        if idx < len(schedules):
            next_departure, next_arrival, _, wheelchair_acc = schedules[idx]
            if not wheelchair or wheelchair_acc == 1:
                return next_departure - current_time + (next_arrival - next_departure)
        
        return float('inf')
    else:
        return edge.get('weight', float('inf'))


def _calculate_delay_sorted(graph, from_node, to_node, current_time, wheelchair=False):
    """
    Calculates the delay and route for a given graph, from_node, to_node, and current_time.
    Used in the time-dependent Dijkstra algorithm.
    """
    edge = graph[from_node][to_node]
    if 'sorted_schedules' in edge:
        schedules = edge['sorted_schedules']
        departure_times = edge['departure_times']
        idx = bisect.bisect_left(departure_times, current_time)
        
        if idx < len(schedules):
            next_departure, next_arrival, route, wheelchair_acc = schedules[idx]
            if not wheelchair or wheelchair_acc == 1:
                return next_departure - current_time + (next_arrival - next_departure), route
        
        return float('inf'), None
    else:
        return edge.get('weight', float('inf')), None


def _calculate_delay_hashed(from_node, to_node, current_time, hashtable, wheelchair=False):
    """
    Calculates the delay and route for a given graph, from_node, to_node, and current_time.
    Used in the time-dependent Dijkstra algorithm.
    This version uses a precomputed hash table for quicker access to sorted schedules.
    """
    schedule_info = hashtable.get((from_node, to_node), [(float('inf'),)])

    # Directly return static weight if present
    if len(schedule_info[0]) == 1:
        return schedule_info[0][0]  # Static weight case

    idx = bisect.bisect_left([d[0] for d in schedule_info], current_time)

    if idx < len(schedule_info):
        next_departure, next_arrival, _, wheelchair_acc = schedule_info[idx]

        # Early return for non-accessible routes when wheelchair is required
        if wheelchair and wheelchair_acc != '1':
            return float('inf')

        return next_departure - current_time + (next_arrival - next_departure)
    else:
        return float('inf')


def time_dependent_dijkstra(graph, source, target, start_time, track_used_routes=False, wheelchair=False):
    # https://doi.org/10.1016/j.entcs.2003.12.019
    # https://bradfieldcs.com/algos/graphs/dijkstras-algorithm/
    """
    Finds the shortest path between two nodes in a graph using a time-dependent Dijkstra algorithm.

    Parameters
    ----------
    graph : networkx.Graph
        The graph to search for the shortest path.
    source
        The starting node.
    target
        The target node.
    start_time : float
        The starting time.
    track_used_routes : bool, optional
        If set to True, the algorithm will return a set of used transit routes.
    wheelchair : bool, optional
        If set to True, the algorithm will only use wheelchair accessible routes.

    Returns
    -------
    tuple
        A tuple containing the following elements:
            - list: The shortest path from the source to the target node.
            - float: The arrival time at the target node.
            - float: The travel time from the source to the target node.
    """
    # abort immediately if the source or target node does not exist in the graph
    if source not in graph or target not in graph:
        raise ValueError("The source or target node does not exist in the graph.")

    # Initialize arrival times and predecessors for the current node in the queue
    arrival_times = {node: float('inf') for node in graph.nodes}
    predecessors = {node: None for node in graph.nodes}
    arrival_times[source] = start_time
    queue = [(start_time, source)]
    visited = set()
    # Track used routes
    routes = {}

    # while the queue is not empty and the target node has not been visited
    while queue:
        # Extract the node with the smallest arrival time from the queue
        current_time, u = heappop(queue)
        # If the node is the target, stop the execution
        if u == target:
            break
        # If the node has already been visited with a better result, skip it
        if u in visited and current_time > arrival_times[u]:
            continue
        # Add the node to the visited set to avoid visiting it again
        visited.add(u)

        # Iterate over all neighbors of the node
        for v in graph.neighbors(u):
            # If the neighbor has not been visited yet
            if v not in visited:
                delay, route = _calculate_delay_sorted(graph,
                                                       u, v,
                                                       current_time,
                                                       wheelchair=wheelchair
                                                       )
                # Skip the neighbor if the arrival time is infinite
                if delay == float('inf'):
                    continue
                # Calculate the new arrival time for the neighbor
                new_arrival_time = current_time + delay
                # If the new arrival time is better, update the arrival time and predecessor
                if new_arrival_time < arrival_times[v]:
                    arrival_times[v] = new_arrival_time

                    if track_used_routes:
                        routes[v] = route

                    # Assign the current node U as the predecessor of the neighbor V (in the loop)
                    predecessors[v] = u
                    # Add the neighbor to the queue with the new arrival time
                    heappush(queue, (new_arrival_time, v))
    
    # reconstruct the path
    path = []
    current_node = target
    # Iterate over all predecessors of the target node
    while current_node is not None:
        path.append(current_node)
        current_node = predecessors[current_node]
    path.reverse()

    travel_time = arrival_times[target] - start_time

    if track_used_routes:
        if path[0] == source:
            # Empty set to track used routes
            used_routes = set()
            # Iterate over all nodes in the path
            for i in range(len(path) - 1):
                v = path[i + 1]
                # Add route, used to go from node U to node V
                used_routes.add(routes[v])
            return path, arrival_times[target], travel_time, used_routes
        else:
            # If the path does not start with the source node, something went wrong, the path was not found
            return [], float('inf'), -1, set()

    else:

        if path[0] == source:
            return path, arrival_times[target], travel_time
        else:
            # If the path does not start with the source node, something went wrong, the path was not found
            return [], float('inf'), -1


def single_source_time_dependent_dijkstra_sorted(graph, source, start_time):
    """
    Finds the shortest path from a source node to all other nodes in a time-dependent graph using Dijkstra's algorithm.

    Parameters
    ----------
    graph : networkx.DiGraph
        The graph to search.
    source : hashable
        The node to start the search from.
    start_time : float
        The time to start the search from.

    Returns
    -------
    tuple
        A tuple containing three dictionaries:
            - arrival_times: A dictionary mapping each node to the earliest arrival time from the source node.
            - predecessors: A dictionary mapping each node to its predecessor on the shortest path from the source node.
            - travel_times: A dictionary mapping each node to the travel time from the source node.
    """
    if source not in graph:
        raise ValueError(f"The source node {source} does not exist in the graph.")
    
    if not isinstance(start_time, (int, float)):
        raise ValueError("The start time must be a number.")

    arrival_times = {node: float("inf") for node in graph.nodes}
    predecessors = {node: None for node in graph.nodes}
    arrival_times[source] = start_time
    travel_times = {}
    queue = [(start_time, source)]

    while queue:
        current_time, current_node = heappop(queue)

        for neighbor in graph.neighbors(current_node):
            delay = _calculate_delay_sorted_nr(
                graph, current_node, neighbor, current_time
            )
            new_arrival_time = current_time + delay

            if new_arrival_time < arrival_times[neighbor]:
                arrival_times[neighbor] = new_arrival_time
                predecessors[neighbor] = current_node
                heappush(queue, (new_arrival_time, neighbor))

                travel_times[neighbor] = new_arrival_time - start_time
    # Path reconstruction is not performed
    # However, based on 'predecessors', you can reconstruct the path to any node

    return arrival_times, predecessors, travel_times


def single_source_time_dependent_dijkstra_hashed(graph, source, start_time, hashtable):
    """
    Finds the shortest path from a source node to all other nodes in a time-dependent graph using Dijkstra's algorithm.
    This version uses a precomputed hash table for quick access to sorted schedules.
    You can use the `process_graph_to_hash_table` function to create the hash table.

    Parameters
    ----------
    graph : networkx.DiGraph
        The graph to search.
    source : hashable
        The node to start the search from.
    start_time : float
        The time to start the search from.
    hashtable : dict
        A hash table for quick access to sorted schedules.

    Returns
    -------
    tuple
        A tuple containing three dictionaries:
            - arrival_times: A dictionary mapping each node to the earliest arrival time from the source node.
            - predecessors: A dictionary mapping each node to its predecessor on the shortest path from the source node.
            - travel_times: A dictionary mapping each node to the travel time from the source node.
    
    See Also
    --------
    nxtransit.functions.process_graph_to_hash_table : Create a hash table for quick access to sorted schedules.
    """
    if hashtable is None:
        raise ValueError("The hash table is required for this algorithm.")

    if source not in graph:
        raise ValueError(f"The source node {source} does not exist in the graph.")

    if not isinstance(start_time, (int, float)):
        raise ValueError("The start time must be a number.")

    arrival_times = {node: float("inf") for node in graph.nodes}
    predecessors = {node: None for node in graph.nodes}
    arrival_times[source] = start_time
    travel_times = {}
    queue = [(start_time, source)]

    while queue:
        current_time, current_node = heappop(queue)

        for neighbor in graph.neighbors(current_node):
            delay = _calculate_delay_hashed(
                current_node, neighbor, current_time, hashtable
            )
            new_arrival_time = current_time + delay

            if new_arrival_time < arrival_times[neighbor]:
                arrival_times[neighbor] = new_arrival_time
                predecessors[neighbor] = current_node
                heappush(queue, (new_arrival_time, neighbor))

                travel_times[neighbor] = new_arrival_time - start_time

    return arrival_times, predecessors, travel_times


def single_source_time_dependent_dijkstra(graph, source, start_time: int, hashtable: dict = None, algorithm='sorted'):
    """
    Compute the shortest paths and travel times from a single source node to all other nodes in a time-dependent graph.
    You can use the `process_graph_to_hash_table` function to create the hash table.

    Parameters
    ----------
    graph : nx.DiGraph
        The time-dependent graph.
    source : Node
        The source node of the graph.
    start_time : int
        The starting time in seconds since midnight.
    hashtable : dict, optional
        A hashtable for storing precomputed values. Required for 'hashed' algorithm.
    algorithm : str, optional
        The algorithm to use for computing the shortest paths. Can be 'sorted', 'hashed'. Defaults to 'sorted'.
        - Sorted: using binary search to find the next departure time in graph attributes (slowest)
        - Hashed: using a precomputed hashtable for quick access to sorted schedules. Works 20 - 30% faster than 'sorted'

    Returns
    -------
    tuple
        A tuple containing three dictionaries:
            - arrival_times: A dictionary mapping each node to the earliest arrival time from the source node.
            - predecessors: A dictionary mapping each node to its predecessor on the shortest path from the source node.
            - travel_times: A dictionary mapping each node to the travel time from the source node.
            
    See Also
    --------
    nxtransit.functions.process_graph_to_hash_table : Create a hash table for quick access to sorted schedules.
    """
    if algorithm == "sorted":
        return single_source_time_dependent_dijkstra_sorted(graph, source, start_time)
    elif algorithm == "hashed":
        return single_source_time_dependent_dijkstra_hashed(
            graph, source, start_time, hashtable
        )
    else:
        raise (ValueError, "Invalid algorithm. Use 'sorted' or 'hashed'")
