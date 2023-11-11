import random

from converters import parse_time_to_seconds
from routers import time_dependent_dijkstra

# Функция для проверки корректности работы алгоритма Дейкстры при расчете матрицы источник-назначение
# Производится сверка матрицы с результатами одиночного поиска аналогичного пути
def test_random_stops(graph, stops_list, results_df, departure_time_input, num_tests=100):
    for _ in range(num_tests):
        # Берем две случайные остановки из списка
        source_stop, dest_stop = random.sample(stops_list, 2)
        
        # Расчет для одиночного пути
        path, arrival_time, calculated_travel_time, _ = time_dependent_dijkstra(graph, source_stop, dest_stop, parse_time_to_seconds(departure_time_input), track_used_routes=True)
        
        # Обнаружение аналогичного пути в матрице источник-назначение
        expected_travel_time = results_df.loc[
            (results_df['source_stop'] == source_stop) & 
            (results_df['destination_stop'] == dest_stop),
            'travel_time'
        ].values[0]
        
        # Проверка совпадения времени пути
        if calculated_travel_time != expected_travel_time:
            print(f"Error: Travel time mismatch for source {source_stop} to destination {dest_stop}:")
            print(f"Calculated: {calculated_travel_time}, Expected: {expected_travel_time}")
        else:
            print(f"Travel time match for source {source_stop} to destination {dest_stop}.")