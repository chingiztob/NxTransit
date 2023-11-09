#ROUTERS ON BUG FOX OR DEV
from heapq import heappop, heappush
import math
import bisect
import networkx as nx

def calculate_delay(graph, from_node, to_node, current_time):
    """
    Calculates the delay between two nodes in a graph, taking into account public transit schedules if available.

    Args:
        graph (networkx.Graph): The graph to calculate the delay on.
        from_node (int): The starting node.
        to_node (int): The ending node.
        current_time (float): The current time.

    Returns:
        float: The delay between the two nodes.
    """
    # Check if this edge has a schedule (public transit)
    # Проверка наличия расписания (schedules) на ребре графа
    if 'schedules' in graph.edges[from_node, to_node]:
        # Выбор кортежа с расписанием, ближайшего к текущему времени
        closest_future_departure = None
        for departure, arrival in graph.edges[from_node, to_node]['schedules']:
            # Фильтрация отправлений, которые стартуют позже текущего времени
            if departure >= current_time:
                if closest_future_departure is None or departure < closest_future_departure[0]:
                    closest_future_departure = (departure, arrival)
        # Если ближайшее отправление найдено, то возвращается время до отправления плюс время пути
        if closest_future_departure:
            next_departure, next_arrival = closest_future_departure
            return next_departure - current_time + (next_arrival - next_departure)
        # Если ближайшее отправление не найдено, то возвращается бесконечность
        else:
            return float('inf')
    else:
        # if this edge does not have a schedule, use the fixed travel time (streets from osmnx, or connectors)
        return graph.edges[from_node, to_node]['weight']

def time_dependent_dijkstra(graph, source, target, start_time):
    """
    Implements time-dependent Dijkstra's algorithm, considering arrival times at each node.
    
    :param graph: a networkx graph object
    :param source: the source node
    :param target: the target node
    :param start_time: the start time of the journey
    
    :return: a tuple containing the path from source to target, the arrival time at the target node, and the travel time from source to target
    """
    # Initialize arrival times to infinity and predecessors to None
    arrival_times = {node: float('inf') for node in graph.nodes()}
    predecessors = {node: None for node in graph.nodes()}
    arrival_times[source] = start_time

    # Initialize the priority queue with the source node and its start time
    queue = [(start_time, source)]
    entry_finder = {source: (start_time, source)}  # Map of nodes to entries in the priority queue
    invalid_entries = set()  # Set of nodes with invalidated entries in the priority queue

    while queue:
        current_time, u = heappop(queue)

        if u == target:
            break  # Early stopping: we found the shortest path to the target

        if u in invalid_entries:
            continue  # Skip processing if this node's entry is invalid

        for v in graph.neighbors(u):
            delay = calculate_delay(graph, u, v, current_time)
            if delay == float('inf'):
                continue

            new_arrival_time = current_time + delay
            if new_arrival_time < arrival_times[v]:
                arrival_times[v] = new_arrival_time
                predecessors[v] = u
                
                # Invalidate the old entry in the priority queue
                if v in entry_finder:
                    invalid_entries.add(v)

                # Add the new or updated entry for node v to the queue
                entry = (new_arrival_time, v)
                heappush(queue, entry)
                entry_finder[v] = entry

    # Reconstruct the path from the target back to the source
    path = []
    current_node = target
    while current_node is not None:
        path.append(current_node)
        current_node = predecessors[current_node]
    path.reverse()

    # Calculate travel time
    travel_time = arrival_times[target] - start_time if target in path else float('inf')

    return path if target in path else None, arrival_times[target] if target in path else float('inf'), travel_time if target in path else None

'''IN DEVELOPMENT'''
#Голый алгоритм А* без учета расписаний и вспомогательные функции
def haversine(lat1, lon1, lat2, lon2):
    R = 6371  # радиус Земли в километрах
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat/2) * math.sin(dlat/2) + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon/2) * math.sin(dlon/2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    distance = R * c
    return distance

def heuristic(graph, node1, node2):
    lat1, lon1 = graph.nodes[node1]['y'], graph.nodes[node1]['x']
    lat2, lon2 = graph.nodes[node2]['y'], graph.nodes[node2]['x']
    return haversine(lat1, lon1, lat2, lon2)

def a_star(graph, start, end):
    queue = [(0, start)]
    g_costs = {start: 0}
    came_from = {start: None}
    f_costs = {start: heuristic(start, end)}

    while queue:
        current = heappop(queue)[1]

        if current == end:
            path = []
            while current is not None:
                path.append(current)
                current = came_from[current]
            path.reverse()
            return path

        for neighbour in graph[current]:
            tentative_g_cost = g_costs[current] + graph[current][neighbour]
            if neighbour not in g_costs or tentative_g_cost < g_costs[neighbour]:
                came_from[neighbour] = current
                g_costs[neighbour] = tentative_g_cost
                f_costs[neighbour] = tentative_g_cost + heuristic(neighbour, end)
                heappush(queue, (f_costs[neighbour], neighbour))

    return None

'''IN TESTING'''

def calculate_delay_sorted(graph, from_node, to_node, current_time):
    # Check if the edge has a sorted schedule
    if 'sorted_schedules' in graph[from_node][to_node]:
        schedules = graph[from_node][to_node]['sorted_schedules']
        # To avoid recomputation, consider storing and using precomputed departure times
        departure_times = [d[0] for d in schedules]
        idx = bisect.bisect_left(departure_times, current_time)
        if idx < len(schedules):
            next_departure, next_arrival = schedules[idx]
            # Return the total time from current_time to next_arrival
            return next_departure - current_time + (next_arrival - next_departure)
        else:
            return float('inf')  # No more departures today
    else:
        # If no schedule, use static weight
        return graph[from_node][to_node]['weight']

def time_dependent_dijkstra_2(graph, source, target, start_time):
    if source not in graph or target not in graph:
        raise ValueError("The source or target node does not exist in the graph.")

    # Initialize arrival times and predecessors
    arrival_times = {node: float('inf') for node in graph.nodes}
    predecessors = {node: None for node in graph.nodes}
    arrival_times[source] = start_time
    queue = [(start_time, source)]
    visited = set()

    while queue:
        current_time, u = heappop(queue)
        
        if u == target:
            break
        
        # If the node has been visited with a better time, skip it
        if u in visited and current_time > arrival_times[u]:
            continue
        
        visited.add(u)  # Mark this node as visited

        for v in graph.neighbors(u):
            if v not in visited:
                delay = calculate_delay_sorted(graph, u, v, current_time)
                if delay == float('inf'):
                    continue
                new_arrival_time = current_time + delay

                if new_arrival_time < arrival_times[v]:
                    arrival_times[v] = new_arrival_time
                    predecessors[v] = u
                    heappush(queue, (new_arrival_time, v))

    # Reconstruct the path
    path = []
    current_node = target
    while current_node is not None:
        path.append(current_node)
        current_node = predecessors[current_node]
    path.reverse()

    travel_time = arrival_times[target] - start_time
    
    if path[0] == source:
        return path, arrival_times[target], travel_time
    else:
        return [], float('inf'), -1

'''DEPRECATED'''

def calculate_delay_sorted(graph, from_node, to_node, current_time):
    if 'sorted_schedules' in graph.edges[from_node, to_node]:
        schedules = graph.edges[from_node, to_node]['sorted_schedules']
        # Find the index of the first departure time that is greater than or equal to the current time
        idx = bisect.bisect_left([d[0] for d in schedules], current_time)
        if idx < len(schedules):
            next_departure, next_arrival = schedules[idx]
            return next_departure - current_time + (next_arrival - next_departure)
        else:
            return float('inf')
    else:
        return graph.edges[from_node, to_node]['weight']   
    
def time_dependent_dijkstra(graph, source, target, start_time):
    """
    Реализует алгоритм Дейкстры с учетом времени прибытия на каждую станцию.
    
    Args:
    ---------
    graph -- граф, представленный в виде объекта NetworkX Graph
    source -- идентификатор исходной вершины
    target -- идентификатор конечной вершины
    start_time -- время отправления в секундах с начала дня
    
    Returns:
    ---------
    path -- список идентификаторов вершин, составляющих кратчайший путь от исходной вершины до конечной
    arrival_times[target] -- время прибытия на конечную вершину в виде секунд после полуночи
    travel_time -- время пути от исходной вершины до конечной в секундах
    """
    # Initialize arrival times to infinity and predecessors to None
    arrival_times = {node: float('inf') for node in graph.nodes()}
    predecessors = {node: None for node in graph.nodes()}
    
    # Set the arrival time of the source to the start time
    arrival_times[source] = start_time
    
    # Initialize the priority queue with the source node and its start time
    queue = [(start_time, source)]
    
    # While the queue is not empty, process the nodes
    while queue:
        # Pop the node with the smallest arrival time
        current_time, u = heappop(queue)
        
        # If we have reached the target node, break
        if u == target:
            break

        for v in graph.neighbors(u):
            # Calculate the delay using the new calculate_delay function
            delay = calculate_delay_sorted(graph, u, v, current_time)
            # Skip processing if the delay is infinity (no suitable departure time found)
            if delay == float('inf'):
                continue
            new_arrival_time = current_time + delay      
            
            # If the new arrival time is better, update it in the priority queue
            if new_arrival_time < arrival_times[v]:
                arrival_times[v] = new_arrival_time
                predecessors[v] = u  # Update the predecessor
                heappush(queue, (new_arrival_time, v))
    
    # Construct the path from target to source by following the predecessors
    path = []
    current_node = target
    while current_node is not None:
        path.append(current_node)
        current_node = predecessors[current_node]
    
    # Reverse the path to get it from source to target
    path.reverse()
    
    travel_time = arrival_times[target] - start_time
    
    if path[0] == source:
        return path, arrival_times[target], travel_time
    else:
        return None, float('inf'), None
