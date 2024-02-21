.. NxTransit documentation master file, created by
   sphinx-quickstart on Wed Feb 21 22:46:02 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

NxTransit |version|
===================

**NxTransit** is a Python package designed for creating and analyzing a multimodal graph representation of a city's transit system. It leverages General Transit Feed Specification (GTFS) data to construct the graph and perform various time-dependent calculations.

Key Features
------------

- **Multimodal Graph Creation**
  NxTransit can generate a graph that integrates different modes of transportation within a city's street network, providing a comprehensive view of the available transit options.

- **Time-Dependent Calculations**
  The package supports the analysis of transit dynamics by considering the time-dependency of transit schedules. This functionality enables users to perform a variety of calculations, such as:

  - Calculating shortest paths that account for specific departure times.
  - Generating travel time matrices to understand travel durations between multiple points within the network.
  - Frequency-based analysis to understand the availability of transit services at different times of the day.
  - and more to come...


- **GTFS Data Support**
  NxTransit uses GTFS data, a widely-adopted format for public transportation schedules and associated geographic information, as the foundation for graph construction and analysis. This ensures that the package can be applied to a broad range of urban settings and transit systems.

This package is primarily intended for researchers and professionals in the fields of geography, transportation, urban studies, and related disciplines. By providing tools for detailed multimodal transit analysis, NxTransit aims to facilitate the development of more efficient and accessible urban transportation systems.


.. toctree::
   :maxdepth: 2
   :caption: Contents:

   NxTransit

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
