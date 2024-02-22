# NxTransit

**NxTransit** is a Python package designed for creating and analyzing a multimodal graph representation of a city's transit system. It uses General Transit Feed Specification (GTFS) data to construct the graph and perform various time-dependent calculations.

See the [documentation](https://nxtransit.readthedocs.io/en/latest/) for more information.

## Key Features

### Multimodal Graph Creation
NxTransit can generate a graph that integrates different modes of transportation within a city's street network. This offers a comprehensive view of the transit options available.

### Time-Dependent Calculations
The package enables the analysis of transit dynamics by considering the time-dependency of transit schedules. Users can perform calculations such as:

- Calculating shortest paths with specific departure times.
- Generating travel time matrices to evaluate travel durations between multiple network points.
- Service areas calculation.
- Frequency analysis.
- More features are planned for future updates.

### GTFS Data Support
NxTransit utilizes GTFS data, a common format for public transportation schedules and geographic information. This allows the package to be used across a variety of urban environments and transit systems.

The package is primarily aimed at researchers in geography, transportation, urban studies, and similar fields.

### Installation
```bash
pip install --index-url https://test.pypi.org/simple/ NxTransit_beta
```
### License
Package is open source and licensed under the MIT license. OpenStreetMap's open data [license](https://www.openstreetmap.org/copyright/) requires that derivative works provide proper attribution. This package heavily depends on [OSMnx](https://geoffboeing.com/publications/osmnx-complex-street-networks/) by Geoff Boeing, which is also licensed under the MIT license.
