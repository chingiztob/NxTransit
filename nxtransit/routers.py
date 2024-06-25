"""Main routing algorithms for time-dependent graphs."""

import bisect
from heapq import heappop, heappush
from typing import Dict, List, Tuple, Optional

from networkx import DiGraph
from .functions import _reconstruct_path


def _calculate_delay(
    graph, from_node, to_node, current_time, wheelchair=False
) -> Tuple[float, Optional[str]]:
    """
    Calculates the delay and route for a given graph, from_node, to_node, and current_time.
    Used in the time-dependent Dijkstra algorithm.
    """
    edge = graph[from_node][to_node]
    schedules = edge.get("sorted_schedules")
    
    if schedules:
        departure_times = edge["departure_times"]
        idx = bisect.bisect_left(departure_times, current_time)

        if idx < len(schedules):
            next_departure, next_arrival, route, wheelchair_acc = schedules[idx]
            if not wheelchair or wheelchair_acc == 1:
                delay = (next_departure - current_time) + (next_arrival - next_departure)
                return delay, route

        return float("inf"), None
    else:
        return edge.get('weight', float('inf')), None


def time_dependent_dijkstra(
    graph: DiGraph,
    source: str,  
    target: str,
    start_time: float,
    track_used_routes: bool = False,
    wheelchair: bool = False,
) -> Tuple[List[str], float, float, Optional[set]]:
    """
    Finds the shortest path between two nodes in a time-dependent graph using Dijkstra's algorithm.
    
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
    
    Examples
    --------
    >>> G = nt.feed_to_graph(feed)
    >>> path, arrival_time, travel_time = nt.time_dependent_dijkstra(G, 'A', 'B', 86400)

    Implementation
    --------------
    This function uses a priority queue to explore the graph with
    almost classic Dijkstra's algorithm. The main difference is that the
    delay between two nodes is calculated based on the ``current time``
    and the sorted schedules of the edge. The function also keeps
    track of the routes used in the path if the ``track_used_routes``
    parameter is set to True.

    >>> G = nx.DiGraph()
    >>> G.add_edge('A', 'B')
    >>> G.edges['A', 'B']['sorted_schedules'] = [(10, 20, 'route_1', None),(30, 40, 'route_2', None), (50, 60, 'route_3', None)]
    
    So the edge from 'A' to 'B' has three schedules: route_1 departs at 10 and arrives at 20, route_2 departs at 30 and arrives at 40, and route_3 departs at 50 and arrives at 60.
    Internal function ``_calculate_delay`` will return the delay and the route for the next departure from 'A' to 'B' at a given time.
    i.e. if the current time is 25, the next departure is at 30, so the delay is 5 and time to arrival is 5 + (40 - 30) = 10.
    
    References
    ----------
    .. [1] Gerth Stølting Brodal, Riko Jacob:
       Time-dependent Networks as Models to Achieve Fast Exact Time-table Queries.
       Electronic Notes in Theoretical Computer Science, 92:3-15, 2004.
       https://doi.org/10.1016/j.entcs.2003.12.019 [1]_
    .. [2] Bradfield:
       Shortest Path with Dijkstra's Algorithm
       Practical Algorithms and Data Structures
       https://bradfieldcs.com/algos/graphs/dijkstras-algorithm/ [2]_
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
                delay, route = _calculate_delay(
                    graph, u, v, current_time, wheelchair=wheelchair
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
    
    travel_time = arrival_times[target] - start_time
    # reconstruct the path
    path = _reconstruct_path(target=target, predecessors=predecessors)
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


def single_source_time_dependent_dijkstra(
    graph: DiGraph,
    source: str,
    start_time: int,
) -> Tuple[Dict[str, float], Dict[str, str], Dict[str, float]]:
    """
    Compute the shortest paths and travel times from a single source node to all other nodes in a time-dependent graph.

    Parameters
    ----------
    graph : nx.DiGraph
        The time-dependent graph.
    source : Node
        The source node of the graph.
    start_time : int
        The starting time in seconds since midnight.

    Returns
    -------
    tuple
        A tuple containing three dictionaries:
            - arrival_times: A dictionary mapping each node to the earliest arrival time from the source node.
            - predecessors: A dictionary mapping each node to its predecessor on the shortest path from the source node.
            - travel_times: A dictionary mapping each node to the travel time from the source node.

    See Also
    --------
    nxtransit.routers.time_dependent_dijkstra : Point to point routing algorithm for time-dependent graphs.
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
            delay, _ = _calculate_delay(
                graph=graph,
                from_node=current_node,
                to_node=neighbor,
                current_time=current_time,
            )
            new_arrival_time = current_time + delay
            if new_arrival_time < arrival_times[neighbor]:
                arrival_times[neighbor] = new_arrival_time
                predecessors[neighbor] = current_node
                heappush(queue, (new_arrival_time, neighbor))
                travel_times[neighbor] = new_arrival_time - start_time

    return arrival_times, predecessors, travel_times
