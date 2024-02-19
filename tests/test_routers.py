import pytest
from transit.routers import _calculate_delay_sorted, _calculate_delay_hashed


@pytest.fixture(scope='module')
def graph():
    return {
        'A': {
            'B': {
                'sorted_schedules': [
                    (10, 20, 'route_1'),
                    (30, 40, 'route_2'),
                    (50, 60, 'route_3'),
                ]
            }
        }
    }


@pytest.fixture(scope='module')
def hash():
    return {('A', 'B'): [(10, 20, 'route_1'), 
                         (30, 40, 'route_2'), 
                         (50, 60, 'route_3')
                         ],
            ('B', 'C'): [(10,)]
            }


def test_next_departure_exists(graph):
    from_node = 'A'
    to_node = 'B'
    current_time = 25
    expected_delay = 15
    expected_route = 'route_2'

    delay, route = _calculate_delay_sorted(graph, from_node, to_node, current_time)

    assert delay == expected_delay
    assert route == expected_route


def test_next_departure_does_not_exist(graph):
    from_node = 'A'
    to_node = 'B'
    current_time = 70
    expected_delay = float('inf')
    expected_route = None

    delay, route = _calculate_delay_sorted(graph, from_node, to_node, current_time)

    assert delay == expected_delay
    assert route == expected_route


def test_edge_not_time_dependent():
    graph = {
        'A': {
            'B': {
                'weight': 10
            }
        }
    }
    from_node = 'A'
    to_node = 'B'
    current_time = 25
    expected_delay = 10
    expected_route = None

    delay, route = _calculate_delay_sorted(graph, from_node, to_node, current_time)

    assert delay == expected_delay
    assert route == expected_route


def test_next_departure_exists_hashed(hash):
    from_node = 'A'
    to_node = 'B'
    current_time = 25
    expected_delay = 15
    expected_route = 'route_2'

    delay, route = _calculate_delay_hashed(from_node, to_node, current_time, hash)

    assert delay == expected_delay
    assert route == expected_route


def test_next_departure_does_not_exist_hashed(hash): 
    from_node = 'A'
    to_node = 'B'
    current_time = 70
    expected_delay = float('inf')
    expected_route = None

    delay, route = _calculate_delay_hashed(from_node, to_node, current_time, hash)

    assert delay == expected_delay
    assert route == expected_route


def test_edge_not_time_dependent_hashed(hash):
    from_node = 'B'
    to_node = 'C'
    current_time = 25
    expected_delay = 10
    expected_route = None

    delay, route = _calculate_delay_hashed(from_node, to_node, current_time, hash)

    assert delay == expected_delay
    assert route == expected_route
