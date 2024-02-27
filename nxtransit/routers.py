"""Main routing algorithms for time-dependent graphs."""
from heapq import heappop, heappush
import bisect


def _calculate_delay_sorted(graph, from_node, to_node, current_time, wheelchair=False):
    """
    Calculates the delay and route for a given graph, from_node, to_node, and current_time.
    Used in the time-dependent Dijkstra algorithm.
    """

    if 'sorted_schedules' in graph[from_node][to_node]:
        schedules = graph[from_node][to_node]['sorted_schedules']
        departure_times = graph[from_node][to_node]['departure_times']
        # Binary search to find the next departure time in the sorted list of schedules
        idx = bisect.bisect_left(departure_times, current_time)
        # If the next departure time exists, calculate the delay and route
        if idx < len(schedules):
            
            next_departure, next_arrival, route, wheelchair_acc = schedules[idx]
            
            if wheelchair and wheelchair_acc != 1:
                return float('inf'), None
            else:
                return next_departure - current_time + (next_arrival - next_departure), route
        # If the next departure time does not exist, return 'inf' as the delay
        else:
            return float('inf'), None
        # If the edge does not have sorted schedules, edge is not time-dependent and has a static weight
    else:
        return graph[from_node][to_node]['weight'], None


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
    routes = {node: None for node in graph.nodes}
    
    # while the queue is not empty and the target node has not been visited
    # Пока очередь не пуста и целевой узел не посещен
    while queue:
        # Extract the node with the smallest arrival time from the queue
        # Извлечение узла с наименьшим временем прибытия в очереди
        current_time, u = heappop(queue)
        # If the node is the target, stop the execution
        # Если узел является целевым, прекратить выполнение
        if u == target:
            break
        # If the node has already been visited with a better result, skip it
        # Если узел уже посещен с лучшим результатом, пропустить его
        if u in visited and current_time > arrival_times[u]:
            continue
        # Add the node to the visited set to avoid visiting it again
        # Добавить узел в список посещенных, чтобы не посещать его снова
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

    # Восстановление пути
    path = []
    current_node = target
    # Перебор всех предшественников целевого узла
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
                u = path[i]
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

    arrival_times = {node: float('inf') for node in graph.nodes}
    predecessors = {node: None for node in graph.nodes}
    arrival_times[source] = start_time
    travel_times = {}
    queue = [(start_time, source)]

    while queue:
        current_time, current_node = heappop(queue)

        for neighbor in graph.neighbors(current_node):
            delay, _ = _calculate_delay_sorted(graph, current_node, neighbor, current_time)
            new_arrival_time = current_time + delay

            if new_arrival_time < arrival_times[neighbor]:
                arrival_times[neighbor] = new_arrival_time
                predecessors[neighbor] = current_node
                heappush(queue, (new_arrival_time, neighbor))

                travel_times[neighbor] = new_arrival_time - start_time
    # Реконструкция пути не производится
    # Однако на основе 'predecessors' можно восстановить путь к любому узлу

    return arrival_times, predecessors, travel_times


def _calculate_delay_hashed(from_node, to_node, current_time, hashtable, wheelchair =False):
    """
    Calculates the delay and route for a given graph, from_node, to_node, and current_time.
    Used in the time-dependent Dijkstra algorithm.
    This version uses a precomputed hash table for quicker access to sorted schedules.
    """
    # Default to 'inf'
    schedule_info = hashtable.get((from_node, to_node), [(float('inf'),)])

    # Handle static weights differently
    # Check if it's a static weight
    if isinstance(schedule_info[0], tuple) and len(schedule_info[0]) == 1:
        return schedule_info[0][0], None  # Return the static weight

    else:
        departure_times = [d[0] for d in schedule_info]
        idx = bisect.bisect_left(departure_times, current_time)

        if idx < len(schedule_info):

            next_departure, next_arrival, route, wheelchair_acc = schedule_info[idx]

            if wheelchair and wheelchair_acc != '1':
                return float('inf'), None
            else:
                return next_departure - current_time + (next_arrival - next_departure), route
        else:

            return float('inf'), None


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
    schedules_hash : dict
        A hash table for quick access to sorted schedules.

    Returns
    -------
    tuple
        A tuple containing three dictionaries:
            - arrival_times: A dictionary mapping each node to the earliest arrival time from the source node.
            - predecessors: A dictionary mapping each node to its predecessor on the shortest path from the source node.
            - travel_times: A dictionary mapping each node to the travel time from the source node.
    """
    if hashtable is None:
        raise ValueError("The hash table is required for this algorithm.")

    if source not in graph:
        raise ValueError(f"The source node {source} does not exist in the graph.")

    arrival_times = {node: float('inf') for node in graph.nodes}
    predecessors = {node: None for node in graph.nodes}
    arrival_times[source] = start_time
    travel_times = {}
    queue = [(start_time, source)]

    while queue:
        current_time, current_node = heappop(queue)

        for neighbor in graph.neighbors(current_node):

            delay, _ = _calculate_delay_hashed(current_node, neighbor, current_time, hashtable)
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
        The algorithm to use for computing shortest paths. Can be 'sorted', 'hashed'. Defaults to 'sorted'.
        - Sorted: using binary search to find the next departure time in graph attributes (slowest)
        - Hashed: using a precomputed hashtable for quick access to sorted schedules. Works 20 - 30% faster than 'sorted'

    Returns
    -------
    tuple
        A tuple containing three dictionaries:
            - arrival_times: A dictionary mapping each node to the earliest arrival time from the source node.
            - predecessors: A dictionary mapping each node to its predecessor on the shortest path from the source node.
            - travel_times: A dictionary mapping each node to the travel time from the source node.
    """

    if algorithm == 'sorted':
        arrival_times, predecessors, travel_times = single_source_time_dependent_dijkstra_sorted(
            graph, source, start_time
            )
    elif algorithm == 'hashed':
        arrival_times, predecessors, travel_times = single_source_time_dependent_dijkstra_hashed(
            graph, source, start_time, hashtable
            )

    return arrival_times, predecessors, travel_times
