from heapq import heappop, heappush
import bisect

from .calculate_delay import calculate_delay_hashed

def calculate_delay_sorted(graph, from_node, to_node, current_time):
    # Проверка наличия отсортированных расписаний в атрибутах текущего ребра
    if 'sorted_schedules' in graph[from_node][to_node]:
        schedules = graph[from_node][to_node]['sorted_schedules']

        departure_times = [d[0] for d in schedules]
        
        #bisect проводит бинарный поиск и определяет индекс, в который нужно вставить текущее время
        #Если индекс меньше длины списка, значит, есть отправление, которое стартует позже текущего времени
        #иначе возвращается бесконечность
        idx = bisect.bisect_left(departure_times, current_time)
        if idx < len(schedules):
            next_departure, next_arrival, route = schedules[idx]
            # Возвращается задержка от текущего времени до следующего отправления
            return next_departure - current_time + (next_arrival - next_departure), route
        else:
            return float('inf'), None  # Сегодня больше нет отправлений
    else:
        # # Если в атрибутах нет расписаний - использовать постоянный вес ребра
        return graph[from_node][to_node]['weight'], None

def time_dependent_dijkstra(graph, source, target, start_time, track_used_routes = False):
    # https://doi.org/10.1016/j.entcs.2003.12.019
    # https://bradfieldcs.com/algos/graphs/dijkstras-algorithm/
    """
    Finds the shortest path between two nodes in a graph using a time-dependent Dijkstra algorithm.

    Args:
        graph (networkx.Graph): The graph to search for the shortest path.
        source: The starting node.
        target: The target node.
        start_time (float): The starting time.

    Returns:
        tuple: A tuple containing the following elements:
            - list: The shortest path from the source to the target node.
            - float: The arrival time at the target node.
            - float: The travel time from the source to the target node.
    """
    # Немедленно прекратить выполнение, если исходный или целевой узел не существует в графе
    if source not in graph or target not in graph:
        raise ValueError("The source or target node does not exist in the graph.")

    # Инициализация времени прибытия и предшественников текущего узла в очереди
    # Initialize arrival times and predecessors
    arrival_times = {node: float('inf') for node in graph.nodes}
    predecessors = {node: None for node in graph.nodes}
    arrival_times[source] = start_time
    queue = [(start_time, source)]
    visited = set()
    # Отслеживаем использованные маршруты
    routes = {node: None for node in graph.nodes}
    
    # Пока очередь не пуста и целевой узел не посещен
    while queue:
        # Извлечение узла с наименьшим временем прибытия в очереди
        current_time, u = heappop(queue)
        # Если узел является целевым, прекратить выполнение
        if u == target:
            break
        # Если узел уже посещен с лучшим результатом, пропустить его
        if u in visited and current_time > arrival_times[u]:
            continue
        # Добавить узел в список посещенных, чтобы не посещать его снова
        visited.add(u)
        
        # Перебор всех соседей узла
        for v in graph.neighbors(u):
            # Если сосед еще не посещен
            if v not in visited:
                delay, route = calculate_delay_sorted(graph, u, v, current_time)
                # Пропустить соседа, если время прибытия бесконечно
                if delay == float('inf'):
                    continue
                # Вычисление нового времени прибытия для соседа
                new_arrival_time = current_time + delay
                # Если новое время прибытия лучше, обновить время прибытия и предшественника
                if new_arrival_time < arrival_times[v]:
                    arrival_times[v] = new_arrival_time
                    
                    if track_used_routes:
                        routes[v] = route
                    
                    # Назначиить текущий узел U (в очереди) предшественником соседа V (в цикле)
                    predecessors[v] = u
                    # Добавить соседа в очередь с новым временем прибытия
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
            # Пустое множество для отслеживания использованных маршрутов
            used_routes = set()
            # Итерация по всем узлам пути
            for i in range(len(path) - 1):
                u = path[i] # ???????
                v = path[i + 1]
                # Добавление маршрута, используемого для перехода от узла U к узлу V
                used_routes.add(routes[v])
            return path, arrival_times[target], travel_time, used_routes
        else:
            # Если путь не начинается с исходного узла, значит что-то пошло не так, путь не найден
            return [], float('inf'), -1, set()
        
    else:
        
        if path[0] == source:
            return path, arrival_times[target], travel_time
        else:
            # Если путь не начинается с исходного узла, значит что-то пошло не так, путь не найден
            return [], float('inf'), -1
        
def single_source_time_dependent_dijkstra(graph, source, start_time):
    """
    Finds the shortest path from a source node to all other nodes in a time-dependent graph using Dijkstra's algorithm.

    Args:
        graph (networkx.DiGraph): The graph to search.
        source (hashable): The node to start the search from.
        start_time (float): The time to start the search from.

    Returns:
        tuple: A tuple containing three dictionaries:
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
            delay, _ = calculate_delay_sorted(graph, current_node, neighbor, current_time)
            new_arrival_time = current_time + delay

            if new_arrival_time < arrival_times[neighbor]:
                arrival_times[neighbor] = new_arrival_time
                predecessors[neighbor] = current_node
                heappush(queue, (new_arrival_time, neighbor))
            
                travel_times[neighbor] = new_arrival_time - start_time
    # Реконструкция пути не производится
    # Однако на основе 'predecessors' можно восстановить путь к любому узлу

    return arrival_times, predecessors, travel_times

# Experimental implementations using hashed schedules and Cython compiled function for calculating the delay

def calculate_delay_hashed_noncython(from_node, to_node, current_time, schedules_hash):
    schedule_info = schedules_hash.get((from_node, to_node), [(float('inf'),)])  # Default to 'inf'
    
    # Handle static weights differently
    if isinstance(schedule_info[0], tuple) and len(schedule_info[0]) == 1:  # Check if it's a static weight
        return schedule_info[0][0], None  # Return the static weight
    
    else:
        departure_times = [d[0] for d in schedule_info]
        idx = bisect.bisect_left(departure_times, current_time)
        
        if idx < len(schedule_info):
            
            next_departure, next_arrival, route = schedule_info[idx]
            return next_departure - current_time + (next_arrival - next_departure), route
        else:
            
            return float('inf'), None

def single_source_time_dependent_dijkstra_hashed(graph, source, start_time, edges_hash):
    """
    Finds the shortest path from a source node to all other nodes in a time-dependent graph using Dijkstra's algorithm.
    This version uses a precomputed hash table for quick access to sorted schedules.

    Args:
        graph (networkx.DiGraph): The graph to search.
        source (hashable): The node to start the search from.
        start_time (float): The time to start the search from.
        schedules_hash (dict): A hash table for quick access to sorted schedules.

    Returns:
        tuple: A tuple containing three dictionaries:
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

            delay, _ = calculate_delay_hashed_noncython(current_node, neighbor, current_time, edges_hash)
            new_arrival_time = current_time + delay

            if new_arrival_time < arrival_times[neighbor]:
                arrival_times[neighbor] = new_arrival_time
                predecessors[neighbor] = current_node
                heappush(queue, (new_arrival_time, neighbor))
            
                travel_times[neighbor] = new_arrival_time - start_time

    return arrival_times, predecessors, travel_times

def single_source_time_dependent_dijkstra_cython(graph, source, start_time, edges_hash):
    """
    Finds the shortest path from a source node to all other nodes in a time-dependent graph using Dijkstra's algorithm.
    This version uses a precomputed hash table for quick access to sorted schedules 
    and a Cython compiled function for calculating the delay.
    
    Non compatible with graphs with nodes that are not integers
    -----------------------------------------------------------

    Args:
        graph (networkx.DiGraph): The graph to search.
        source (hashable): The node to start the search from.
        start_time (float): The time to start the search from.
        schedules_hash (dict): A hash table for quick access to sorted schedules.

    Returns:
        tuple: A tuple containing three dictionaries:
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

            delay, _ = calculate_delay_hashed(current_node, neighbor, current_time, edges_hash)
            new_arrival_time = current_time + delay

            if new_arrival_time < arrival_times[neighbor]:
                arrival_times[neighbor] = new_arrival_time
                predecessors[neighbor] = current_node
                heappush(queue, (new_arrival_time, neighbor))
            
                travel_times[neighbor] = new_arrival_time - start_time

    return arrival_times, predecessors, travel_times