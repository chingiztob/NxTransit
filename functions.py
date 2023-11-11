import pandas as pd

from routers import single_source_time_dependent_dijkstra

# Функция для расчета матрицы источник-назначение между всеми остановками (БЕТА)
def calculate_OD_matrix(graph, stops, departure_time):
    """
    Calculates the Origin-Destination (OD) matrix for a given graph, stops, and departure time.
    
    Parameters:
    graph (networkx.Graph): The graph representing the transit network.
    stops (pandas.DataFrame): A DataFrame containing the stops information.
    departure_time (int): The departure time in seconds since midnight.
    
    Returns:
    pandas.DataFrame: A DataFrame containing the OD matrix with the following columns:
        - source_stop: The ID of the origin stop.
        - destination_stop: The ID of the destination stop.
        - arrival_time: The arrival time at the destination stop in seconds since midnight.
        - travel_time: The travel time from the origin stop to the destination stop in seconds.
    """
    
    stops_list = stops['stop_id'].tolist() #Список остановок
    results = []

    for source_stop in stops_list:
        # Вычисление времени прибытия и предшественников для каждой остановки
        arrival_times, predecessors, travel_times = single_source_time_dependent_dijkstra(graph, source_stop, departure_time)
        
        # Итерация по всем остановкам, для их отбора в результатах работы алгоритма дейкстры
        for dest_stop in stops_list:
            if dest_stop in arrival_times:
                # Добавление результатов в список
                results.append({
                    'source_stop': source_stop,
                    'destination_stop': dest_stop,
                    'arrival_time': arrival_times[dest_stop],
                    'travel_time': travel_times.get(dest_stop, None)  # Use .get() to avoid KeyError if the key is not found
                })

    # Конвертация списка в датафрейм и в файл csv
    results_df = pd.DataFrame(results)
    results_df.to_csv(r"D:\Python_progs\Output\results2.csv", index=False)