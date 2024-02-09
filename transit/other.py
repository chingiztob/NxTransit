import webbrowser
import tempfile
import ctypes
import platform

import pandas as pd
import osmnx as ox
import folium

# Not updated
def plot_path_browser(graph, stops: pd.DataFrame):
    """
    Plots the given graph and stops on a folium map and opens it in a web browser.
    
    Parameters:
    - graph: The graph to plot.
    - stops: The stops to plot.
    """
    
    G_nodes, G_edges = ox.graph_to_gdfs(graph)
    
    frame_center_lat = stops['stop_lat'].mean()
    frame_center_lon = stops['stop_lon'].mean()
    
    m = folium.Map(location=[frame_center_lat, frame_center_lon], 
                zoom_start=11,
                width='100%', 
                height='100%')
    
    G_edges.explore(m = m)  
    G_nodes.explore(m = m, column = 'type', legend = True)
    
    with tempfile.NamedTemporaryFile(mode='w', delete_on_close = True, 
                                     delete = False, prefix = 'path', 
                                     suffix='.html') as outfp:
        m.save(outfp.name)
        webbrowser.open(outfp.name)
        
#https://systemweakness.com/how-to-use-the-win32api-with-python3-3adde999211b
def estimate_ram():
    """
    This function estimates the total and free physical memory available on a Windows based system.
    :return: A tuple containing the total and free physical memory in bytes.
    :raises: OSError if the function is called on a non-Windows based system or if the memory retrieval fails.
    """
    if platform.system() != "Windows":
        raise OSError("This method is intended for Windows based systems.")
    
    kernel32 = ctypes.WinDLL('kernel32', use_last_error=True)
    
    # Function to retrieve system memory information
    class MEMORYSTATUSEX(ctypes.Structure):
        _fields_ = [
            ('dwLength', ctypes.c_ulong),
            ('dwMemoryLoad', ctypes.c_ulong),
            ('ullTotalPhys', ctypes.c_ulonglong),
            ('ullAvailPhys', ctypes.c_ulonglong),
            ('ullTotalPageFile', ctypes.c_ulonglong),
            ('ullAvailPageFile', ctypes.c_ulonglong),
            ('ullTotalVirtual', ctypes.c_ulonglong),
            ('ullAvailVirtual', ctypes.c_ulonglong),
            ('ullAvailExtendedVirtual', ctypes.c_ulonglong),
        ]

    mem_info = MEMORYSTATUSEX()
    mem_info.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
    
    if kernel32.GlobalMemoryStatusEx(ctypes.byref(mem_info)) == 0:
        error_code = ctypes.get_last_error()
        raise OSError(f"GlobalMemoryStatusEx failed with error code {error_code}")

    total_memory = mem_info.ullTotalPhys
    free_memory = mem_info.ullAvailPhys

    return total_memory, free_memory

# Convert bytes to human-readable format (e.g., MB or GB)
def bytes_to_readable(bytes):
    if bytes < 1024:
        return f"{bytes} B"
    elif bytes < 1024 ** 2:
        return f"{bytes / 1024:.2f} KB"
    elif bytes < 1024 ** 3:
        return f"{bytes / (1024 ** 2):.2f} MB"
    else:
        return f"{bytes / (1024 ** 3):.2f} GB"