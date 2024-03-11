# NxTransit

[![tests](https://github.com/chingiztob/NxTransit/actions/workflows/basic_tests.yml/badge.svg?event=push)](https://github.com/chingiztob/NxTransit/actions/workflows/basic_tests.yml)
[![Documentation Status](https://readthedocs.org/projects/nxtransit/badge/?version=latest)](https://nxtransit.readthedocs.io/en/latest/?badge=latest)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/NxTransit)

**NxTransit** is a Python package designed for creating and analyzing a multimodal graph representation of a city's transit system. It uses General Transit Feed Specification (GTFS) data to construct the graph and perform various time-dependent calculations.

See the [documentation](https://nxtransit.readthedocs.io/en/latest/) for more information.

## Key Features

### Multimodal Graph Creation
NxTransit integrates transit data with OpenStreetMap (OSM) data to create a multimodal graph representing urban transit system.

### Time-Dependent Calculations
The package enables the detailed analysis of transit systems by incorporating time-dependent nature of public transportation. This includes:

- GTFS feed validation.
- Calculating shortest paths with specific departure times.
- Generating travel time matrices to evaluate travel durations between multiple network points.
- Calculation of service areas and "typical" service areas.
- Frequency analysis.
- More features are planned for future updates.

### GTFS Data Support
NxTransit utilizes GTFS data, a common format for public transportation schedules and geographic information.

### Installation
```bash
pip install NxTransit
```
### License
Package is open source and licensed under the MIT license. OpenStreetMap's open data [license](https://www.openstreetmap.org/copyright/) requires that derivative works provide proper attribution. This package heavily depends on [OSMnx](https://geoffboeing.com/publications/osmnx-complex-street-networks/) by Geoff Boeing, which is also licensed under the MIT license.
