import multiprocessing
import time
from functools import partial

import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, Polygon
from pympler import asizeof

from routers import single_source_time_dependent_dijkstra
from other import estimate_ram, bytes_to_readable

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
        arrival_times, _, travel_times = single_source_time_dependent_dijkstra(graph, source_stop, departure_time)
        
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

def calculate_OD_worker(source_stop, stops_list, graph, departure_time):
    """
    Internal worker function to calculate the OD matrix for a single source stop.
    """
    arrival_times, _, travel_times = single_source_time_dependent_dijkstra(
        graph, source_stop, departure_time
        )
    return [{
        'source_stop': source_stop,
        'destination_stop': dest_stop,
        'arrival_time': arrival_times[dest_stop],
        'travel_time': travel_times.get(dest_stop, None)
    } for dest_stop in stops_list if dest_stop in arrival_times]

def calculate_OD_matrix_parallel(graph, stops, departure_time, num_processes=2):
    """
    Calculates the Origin-Destination (OD) matrix for a given graph, 
    stops, and departure time using parallel processing.
    
    Parameters:
    -----------
    graph (networkx.Graph): The graph representing the transit network.
    stops (pandas.DataFrame): A DataFrame containing the stops information.
    departure_time (int): The departure time in seconds since midnight.
    num_processes (int): Number of parallel processes to use for computation.
        Strongly reccomended to use number of processes equal 
        to the number of physical CPU cores or less.
    
    Returns:
    ----------
    pandas.DataFrame: A DataFrame containing the OD matrix.
    """
    print(f'Выполняется расчет с использованием {num_processes} процессов')
    
    
    # Предварительная попытка оценить достаточность
    # доступной памяти для выполнения расчетов
    graph_size = asizeof.asizeof(graph)
    ram, free_ram = estimate_ram()
    
    expected_ram = graph_size * 5 + num_processes * graph_size * 2.5
    
    if expected_ram > free_ram:
        raise MemoryError(f'Размер графа {bytes_to_readable(graph_size)}, ожидаемые затраты {expected_ram} превышают доступную память {bytes_to_readable(free_ram)}')
    else:
        print(f'Размер графа {bytes_to_readable(graph_size)}, Ожидаемые затраты {bytes_to_readable(expected_ram)}, доступная память {bytes_to_readable(free_ram)}')

    stops_list = stops['stop_id'].tolist() #Список остановок
    results = []
    
    time_start = time.time()

    with multiprocessing.Pool(processes=num_processes) as pool:
        # Фиксаця аргументов функции calculate_OD_worker
        partial_worker = partial(calculate_OD_worker, stops_list=stops_list, 
                                 graph=graph, departure_time=departure_time)
        results = pool.map(partial_worker, stops_list)

    # Группировка результатов в один список
    results = [item for sublist in results 
               for item in sublist]

    results_df = pd.DataFrame(results)

    # Запись результатов в файл csv
    print("Выполняется запись файла csv")
    results_df.to_csv(r"D:\Python_progs\Output\results_3.csv", index=False)
    
    print(f"Time elapsed: {time.time() - time_start}")
    
    return results_df

def create_service_area(graph, source, start_time, cutoff, buffer_radius):
    """
    Creates a service area by buffering around all points within a travel time cutoff.

    Args:
        graph (networkx.DiGraph): The graph to search.
        source: The graph node to start the search from.
        start_time (float): The time to start the search from.
        cutoff (float): The travel time cutoff for including nodes in the service area.
        buffer_radius (float): The radius in meters for buffering around each point.

    Returns:
        tuple: A tuple containing two GeoDataFrames:
            - The first GeoDataFrame has a single geometry column containing the merged buffer polygon.
            - The second GeoDataFrame contains the points within the cutoff.
    """
    _, _, travel_times = single_source_time_dependent_dijkstra(graph, source, start_time)

    # Фильтрация узлов, достижимых за указанное время
    points_data = [{'node': node, 
                    'geometry': Point(
                        graph.nodes[node]['x'], 
                        graph.nodes[node]['y']
                        ), 
                    'travel_time': time}
                   for node, time in travel_times.items() 
                   if time <= cutoff 
                   and 'x' in graph.nodes[node] 
                   and 'y' in graph.nodes[node]]

    # Создание GeoDataFrame из узлов, достижимых за указанное время
    points_gdf = gpd.GeoDataFrame(points_data, geometry='geometry', 
                                  crs="EPSG:4326")

    # Буфферизация узлов, обьединение буферов в один полигон
    # Перевод в World Equidistant Cylindrical
    buffer_gdf = points_gdf.to_crs("EPSG:4087")
    buffer_gdf['geometry'] = buffer_gdf.buffer(buffer_radius)
    service_area_polygon = buffer_gdf.unary_union

    # Создание GeoDataFrame из полигона (Shapely Polygon) в EPSG:4326
    service_area_gdf = gpd.GeoDataFrame({'geometry': [service_area_polygon]}, 
                                        crs="EPSG:4087").to_crs("EPSG:4326")

    return service_area_gdf, points_gdf

def create_grid(geodataframe, cell_size):
    """
    Creates a grid within the bounding box of a GeoDataFrame.

    Args:
    --------
        geodataframe: GeoDataFrame containing the geometry to be gridded
        cell_size (float): size of the grid cells in the meters

    Returns:
    --------
        modified graph
    """
    geodataframe = geodataframe.to_crs("EPSG:4087")
    xmin, ymin, xmax, ymax = geodataframe.total_bounds
    rows = int((ymax - ymin) / cell_size)
    cols = int((xmax - xmin) / cell_size)
    grid = []

    for i in range(cols):
        for j in range(rows):
            x1 = xmin + i * cell_size
            y1 = ymin + j * cell_size
            x2 = x1 + cell_size
            y2 = y1 + cell_size
            grid.append(Polygon([(x1, y1), (x2, y1), 
                                 (x2, y2), (x1, y2)])
                        )

    grid_geodataframe = gpd.GeoDataFrame(grid, 
                                         columns=['geometry'], 
                                         crs=geodataframe.crs
                                         ).to_crs("EPSG:4326")
    
    return grid_geodataframe
