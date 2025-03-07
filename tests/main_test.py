from pathlib import Path

import pytest
import nxtransit as tr


# Zheleznogorsk feed by github.com/gammapopolam
GTFSpath = Path("tests/test_data/Zhelez")
graphml_path = Path("tests/test_data/Zhelez.graphml")
departure_time_input = "12:00:00"  # Departure time in HH:MM:SS format
day = 'monday'  # day of the week in lower case


@pytest.fixture(scope="module")
def graph():
    G = tr.feed_to_graph(
        GTFSpath,
        departure_time_input,
        day,
        duration_seconds=3600*3,
        read_shapes=False,
        multiprocessing=True
    )

    return G


@pytest.mark.parametrize("read_shapes,multiprocessing, load_graphml", [
    (False, False, False),
    (True, True, False),
    (False, True, True),
])
def test_loading_feed(read_shapes, multiprocessing, load_graphml):
    G = tr.feed_to_graph(GTFSpath, departure_time_input, day,
                                duration_seconds=3600*3,
                                read_shapes=read_shapes,
                                multiprocessing=multiprocessing,
                                load_graphml=load_graphml,
                                input_graph_path=graphml_path)
    assert G is not None


def test_validation():
    tr.validate_feed(GTFSpath)

    assert True


def test_routing(graph):

    source = 1327821472
    target = 3578901323
    start_time = tr.parse_time_to_seconds("12:00:00")

    path, arrival_time, travel_time, used_routes = tr.time_dependent_dijkstra(
        graph, source, target, start_time, track_used_routes=True
        )

    assert int(arrival_time) == 45926
    assert int(travel_time) == 2726
    assert path is not None
    assert used_routes is not None


def test_single_source(graph):
    source = 1327821472
    target = 3578901323
    start_time = tr.parse_time_to_seconds("12:00:00")

    arrival_times, predecessors, travel_times = tr.single_source_time_dependent_dijkstra(
        graph, source, start_time)

    assert int(arrival_times[target]) == 45926
    assert int(travel_times[target]) == 2726


def test_connectivity_frequency(graph):
    source = 1327821311
    target = 2034401710
    start_time = tr.parse_time_to_seconds("12:00:00")
    end_time = start_time + 3600*2

    connectivity = tr.connectivity_frequency(graph, source, target, start_time, end_time, 60)

    assert connectivity is not None
    assert connectivity == 1200
