# cython: language_level=3

from libc.stdint cimport int64_t
import bisect

cpdef calculate_delay_hashed(int64_t from_node, int64_t to_node, int64_t current_time, object schedules_hash):
    cdef:
        list schedule_info
        int idx
        int64_t next_departure, next_arrival
        object route

    schedule_info = schedules_hash.get((from_node, to_node), [(float('inf'),)])
    
    # Static weight handling
    if isinstance(schedule_info[0], tuple) and len(schedule_info[0]) == 1:
        return schedule_info[0][0], None
    
    else:
        departure_times = [d[0] for d in schedule_info]
        idx = bisect.bisect_left(departure_times, current_time)
        
        if idx < len(schedule_info):
            next_departure, next_arrival, route = schedule_info[idx]
            # Make sure to convert the result to Python int if necessary, as Cython int64_t to Python int is automatic
            return int(next_departure - current_time + (next_arrival - next_departure)), route
        else:
            return float('inf'), None