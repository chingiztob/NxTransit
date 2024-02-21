from nxtransit.functions import last_service
import networkx as nx

def test_last_service_with_sorted_schedules():
    
    graph = nx.Graph()
    graph.add_node('A', type='transit')
    graph.add_node('B', type='transit')
    graph.add_edge('A', 'B', sorted_schedules=[(10, 20, 'route_1'), (30, 40, 'route_2')])

    last_service(graph)

    assert graph.nodes['A']['last_service'] == 40
    assert graph.nodes['B']['last_service'] == 40


def test_last_service_without_sorted_schedules():
    
    graph = nx.Graph()
    graph.add_node('A', type='transit')
    graph.add_node('B', type='transit')
    graph.add_edge('A', 'B', weight=10)

    last_service(graph)

    assert graph.nodes['A']['last_service'] is None
    assert graph.nodes['B']['last_service'] is None


def test_last_service_with_mixed_edges():
    
    graph = nx.Graph()
    graph.add_node('A', type='transit')
    graph.add_node('B', type='transit')
    graph.add_node('C', type='transit')
    graph.add_edge('A', 'B', sorted_schedules=[(10, 20, 'route_1'), (30, 40, 'route_2')])
    graph.add_edge('B', 'C', weight=10)

    last_service(graph)

    assert graph.nodes['A']['last_service'] == 40
    assert graph.nodes['B']['last_service'] == 40
    assert graph.nodes['C']['last_service'] is None
