import ctypes
import platform

#https://systemweakness.com/how-to-use-the-win32api-with-python3-3adde999211b
def estimate_ram():
    """
    This function estimates the total and free physical memory available on a Windows based system.
    
    Returns: 
    ----------
        A tuple containing the total and free physical memory in bytes.
        
    Raises: 
    ----------
        OSError if the function is called on a non-Windows based system or if the memory retrieval fails.
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