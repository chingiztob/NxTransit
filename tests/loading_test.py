import pytest
import transit as tr

# Zheleznogorsk feed by github.com/gammapopolam
GTFSpath = r"tests/test_data/Zhelez" # Path to GTFS feed
departure_time_input = "12:00:00" # Departure time in HH:MM:SS format
day = 'monday' # day of the week in lower case

@pytest.fixture(scope="module")
def graph():
    G, _ = tr.feed_to_graph(GTFSpath, 
                                 departure_time_input, 
                                 day, 
                                 duration_seconds=3600*3, 
                                 read_shapes = False,
                                 multiprocessing=True)
   
    return G
    
@pytest.mark.parametrize("read_shapes,multiprocessing", [
    (False, False),
    (True, True),
])
def test_loading_feed(read_shapes, multiprocessing):
    G, stops = tr.feed_to_graph(GTFSpath, departure_time_input, day,
                                duration_seconds=3600*3, 
                                read_shapes=read_shapes,
                                multiprocessing=multiprocessing)
    assert G is not None
    assert stops is not None

def test_validation():
    tr.validate_feed(GTFSpath)
    
    assert True
    
def test_routing(graph):

    source = 1327821472  
    target = 3578901323  
    start_time = tr.parse_time_to_seconds("12:00:00")
    
    path, arrival_time, travel_time, used_routes = tr.time_dependent_dijkstra(graph, source, target, start_time, track_used_routes=True)
    
    assert int(arrival_time) == 45923 
    assert int(travel_time) == 2723
    assert path is not None
    assert used_routes is not None

def test_single_source(graph):
    source = 1327821472
    target = 3578901323   
    start_time = tr.parse_time_to_seconds("12:00:00")
    
    arrival_times, predecessors, travel_times = tr.single_source_time_dependent_dijkstra(graph, source, 
                                                                                         start_time)
    
    assert int(arrival_times[target]) == 45923
    assert int(travel_times[target]) == 2723
    
def test_connectivity_frequency(graph):
    source = 1327821311 
    target = 2034401710    
    start_time = tr.parse_time_to_seconds("12:00:00")
    end_time = start_time + 3600*2
    
    connectivity = tr.connectivity_frequency(graph, source, target, start_time, end_time, 60)
    
    assert connectivity is not None
    assert connectivity == 1200
     
if __name__ == "__main__":
    # test loading of feed
    test_loading_feed()
    test_validation()

    # test routing
    test_routing()
    test_single_source()
    
    # Other tests
    test_connectivity_frequency()
    
    #if this are all successful, then the all other tests should be successful as well
    