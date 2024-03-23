import geopandas as gpd
import networkx as nx
from shapely.geometry import Point

import nxtransit as nt


def test_last_service_with_sorted_schedules():
    
    graph = nx.Graph()
    graph.add_node('A', type='transit')
    graph.add_node('B', type='transit')
    graph.add_edge('A', 'B', sorted_schedules=[(10, 20, 'route_1'), (30, 40, 'route_2')])

    nt.last_service(graph)

    assert graph.nodes['A']['last_service'] == 40
    assert graph.nodes['B']['last_service'] == 40


def test_last_service_without_sorted_schedules():
    
    graph = nx.Graph()
    graph.add_node('A', type='transit')
    graph.add_node('B', type='transit')
    graph.add_edge('A', 'B', weight=10)

    nt.last_service(graph)

    assert graph.nodes['A']['last_service'] is None
    assert graph.nodes['B']['last_service'] is None


def test_last_service_with_mixed_edges():
    
    graph = nx.Graph()
    graph.add_node('A', type='transit')
    graph.add_node('B', type='transit')
    graph.add_node('C', type='transit')
    graph.add_edge('A', 'B', sorted_schedules=[(10, 20, 'route_1'), (30, 40, 'route_2')])
    graph.add_edge('B', 'C', weight=10)

    nt.last_service(graph)

    assert graph.nodes['A']['last_service'] == 40
    assert graph.nodes['B']['last_service'] == 40
    assert graph.nodes['C']['last_service'] is None


def test_determine_utm_zone_north_hemisphere():
    # Create a GeoDataFrame with a centroid in the northern hemisphere
    gdf = gpd.GeoDataFrame(geometry=[Point(0, 30)], crs="EPSG:4326")
    utm_zone = nt.determine_utm_zone(gdf)
    assert utm_zone == "EPSG:32631"
    
    
def test_determine_utm_zone_equator():
    # Create a GeoDataFrame with a centroid on the equator
    gdf = gpd.GeoDataFrame(geometry=[Point(0, 0)], crs="EPSG:4326")
    utm_zone = nt.determine_utm_zone(gdf)
    assert utm_zone == "EPSG:32631"


def test_determine_utm_zone_south_hemisphere():
    # Create a GeoDataFrame with a centroid in the southern hemisphere
    gdf = gpd.GeoDataFrame(geometry=[Point(0, -30)], crs="EPSG:4326")
    utm_zone = nt.determine_utm_zone(gdf)
    assert utm_zone == "EPSG:32731"


def test_determine_utm_zone():
    gdf = gpd.GeoDataFrame(geometry=[Point(35, 55)], crs="EPSG:4326")
    utm_zone = nt.determine_utm_zone(gdf)
    assert utm_zone == "EPSG:32636"