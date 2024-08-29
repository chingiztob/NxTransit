# ruff: noqa: F401
"""
NxTransit is a Python package designed for creating a multimodal graph representation of a public transportation systems. 
It uses General Transit Feed Specification (GTFS) data to construct the graph and perform various time-dependent calculations.

Key Features:
- Multimodal Graph Creation: NxTransit can generate a graph that integrates different modes of transportation with street network.
- Time-Dependent Calculations: The package allows for the analysis of transit dynamics by considering the time-dependency of transit schedules. This includes calculating shortest paths with departure times, travel time matrices, and service frequency.
- GTFS Data Support: NxTransit uses GTFS data, a common format for public transportation schedules and associated geographic information, as the basis for graph construction and analysis.
"""
__version__ = "0.3.0"

from .loaders import feed_to_graph
from .loaders import load_stops_gdf

from .routers import time_dependent_dijkstra
from .routers import single_source_time_dependent_dijkstra

from .accessibility import calculate_od_matrix
from .accessibility import calculate_od_matrix_parallel
from .accessibility import service_area
from .accessibility import service_area_multiple_sources
from .accessibility import percent_access_service_area
from .accessibility import last_service

from .frequency import edge_frequency
from .frequency import node_frequency
from .frequency import connectivity_frequency
from .frequency import single_source_connectivity_frequency

from .functions import aggregate_to_grid
from .functions import create_centroids_dataframe
from .functions import validate_feed
from .functions import separate_travel_times

from .converters import parse_seconds_to_time
from .converters import parse_time_to_seconds

from .connectors import snap_points_to_network
