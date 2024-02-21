"""
NxTransit is a Python package designed for creating a multimodal graph representation of a public transportation systems. 
It leverages General Transit Feed Specification (GTFS) data to construct the graph and perform various time-dependent calculations.

Key Features:
- Multimodal Graph Creation: NxTransit can generate a graph that integrates different modes of transportation with street network.
- Time-Dependent Calculations: The package allows for the analysis of transit dynamics by considering the time-dependency of transit schedules. This includes calculating shortest paths with departure times, travel time matrices, and service frequency.
- GTFS Data Support: NxTransit uses GTFS data, a common format for public transportation schedules and associated geographic information, as the basis for graph construction and analysis.

This package is primarily intended for use by researchers in the field of geography, transportation and urban studies.
"""
from ._version import __version__ 

from .routers import time_dependent_dijkstra, single_source_time_dependent_dijkstra

from .loaders import feed_to_graph

from .functions import *

from .converters import parse_seconds_to_time, parse_time_to_seconds

from .connectors import snap_points_to_network
