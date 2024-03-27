.. NxTransit documentation master file, created by
   sphinx-quickstart on Wed Feb 21 22:46:02 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. meta::
    :description: NxTransit integrates GTFS data with OSM to create a time-dependent, multimodal urban transit graph for advanced transportation analysis.
    :keywords: GTFS, transit, transportation, multimodal, graph, analysis, urban, public transportation, OSM, OpenStreetMap
    :google-site-verification: JPVWxkXKsSBRvQqXp1MfUV7TaLwUa6PlJPXV4KDEujU

NxTransit |version|
===================

**NxTransit**  is a Python package designed for creating and analyzing a multimodal graph representation of a city's transit system. It uses General Transit Feed Specification (GTFS) data to construct the graph and perform various time-dependent calculations.

.. note::
   NxTransit is being developed as part of the author's bachelor graduation paper.
   Currently it is not finished and not ready for production use.

Key Features
------------

- **Multimodal Graph Creation**
  NxTransit integrates transit data with OpenStreetMap (OSM) data to create a multimodal graph representing urban transit system.

- **Time-Dependent Calculations**
  The package enables the detailed analysis of transit systems by incorporating time-dependent nature of public transportation. This includes:

  - GTFS feed validation.
  - Calculating shortest paths with specific departure times.
  - Generating travel time matrices to evaluate travel durations between multiple network points.
  - Calculation of service areas and "typical" service areas.
  - Frequency analysis.
  - More features are planned for future updates.


- **GTFS Data Support**
  NxTransit uses GTFS data, a widely-adopted format for public transportation schedules and associated geographic information, as the foundation for graph construction and analysis. This ensures that the package can be applied to a broad range of urban settings and transit systems.

Installation
------------
.. code-block:: python

    pip install nxtransit

Repository
----------
You can find the source NxTransit Repository on `GitHub`_.

.. _GitHub: https://github.com/chingiztob/NxTransit

Documentation
-------------
.. toctree::
   :maxdepth: 1

   NxTransit
   internals
   examples_basic
   examples_h3

Indices and tables
------------------
* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`