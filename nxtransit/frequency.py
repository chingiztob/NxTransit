"""Tools for calculating various frequency of transit service metrics in a network."""
from functools import partial
from statistics import mean

import numpy as np
import scipy.signal

from .routers import single_source_time_dependent_dijkstra, time_dependent_dijkstra


def edge_frequency(graph, start_time, end_time):
    """
    Calculates the frequency of edges in a graph 
    based on the schedules between start_time and end_time.

    Parameters
    ----------
    graph : networkx.DiGraph
        The graph containing the edges and schedules.
    start_time : int
        The start time in seconds from midnight.
    end_time : int
        The end time in seconds from midnight.

    Returns
    -------
    None
    """

    for edge in graph.edges(data=True):
        if "schedules" in edge[2]:
            trips = edge[2]["sorted_schedules"]
            seq = [
                (trips[i + 1][0] - trips[i][0])
                for i in range(len(trips) - 1)
                if start_time <= trips[i][0] <= end_time
            ]  # list containing the headways between consecutive trips along the edge

            if len(seq) > 0:
                frequency = mean(seq)
            else:
                frequency = None

            edge[2]["frequency"] = frequency


def node_frequency(graph, start_time, end_time):
    """
    Calculates the frequency of departures at nodes in a graph 
    based on the schedules of adjacent edges between start_time and end_time.

    Parameters
    ----------
    graph : networkx.DiGraph
        The graph containing the nodes and adjacent edges with schedules.
    start_time : int
        The start time in seconds from midnight.
    end_time : int
        The end time in seconds from midnight.

    Returns
    -------
    None
    """

    for node_view in graph.nodes(data=True):
        node = node_view[0]
        all_times = []

        if node_view[1]["type"] == "transit":
            # Iterate through all edges adjacent to the current node
            for edge in graph.edges(node, data=True):
                if "schedules" in edge[2]:
                    for schedule in edge[2]["schedules"]:
                        departure_time = schedule[0]
                        if start_time <= departure_time <= end_time:
                            all_times.append(departure_time)

            all_times.sort()
            # Calculate the headways between consecutive departures (arrivals?)
            headways = [
                (all_times[i + 1] - all_times[i]) for i in range(len(all_times) - 1)
            ]

            if len(headways) > 0:
                frequency = mean(headways)
            else:
                frequency = None

            graph.nodes[node]["frequency"] = frequency


def connectivity_frequency(graph, source, target, start_time, end_time, sampling_interval=60):
    """
    Calculates the connectivity frequency between a source and target node in a graph
    over a specified time period.

    Parameters
    ----------
    graph : networkx.DiGraph
        The graph object representing the network.
    source : hashable
        The source node.
    target : hashable
        The target node.
    start_time : int or float
        The start time of the analysis period.
    end_time : int or float
        The end time of the analysis period.
    sampling_interval : int or float
        The time interval at which to sample the connectivity.

    Returns
    -------
    float
        The mean interval between peaks in the connectivity.
    """
    travel_parameters = [graph, source, target]
    func = partial(time_dependent_dijkstra, *travel_parameters)
    data = [
        (start_time, func(start_time)[1])
        for start_time in range(start_time, end_time, sampling_interval)
    ]

    time_values = np.array([float(item[1]) for item in data])
    time_seconds = np.array([float(item[0]) for item in data])

    # Compute the first and second derivatives of the travel times
    first_derivative = np.gradient(time_values)
    second_derivative = np.gradient(first_derivative)

    # Find the extrema in the second derivative
    peaks, _ = scipy.signal.find_peaks(second_derivative)
    # Calculate the intervals between the peaks
    intervals = np.diff(time_seconds[peaks])

    # Calculate the mean interval between the peaks
    mean_interval_between_peaks = np.mean(intervals)

    return mean_interval_between_peaks


def single_source_connectivity_frequency(
    graph,
    source,
    start_time,
    end_time,
    sampling_interval=60,
    hashtable=None,
    algorithm="sorted",
):
    """
    Calculates the mean interval between peaks in travel times from a source node to all other nodes
    in a graph over a specified time period.

    Parameters
    ----------
    graph : networkx.DiGraph
        The graph object representing the network.
    source : hashable
        The source node.
    start_time : int or float
        The start time of the analysis period.
    end_time : int or float
        The end time of the analysis period.
    sampling_interval : int or float
        The time interval at which to sample the connectivity.
    algorithm : str, optional
        The algorithm to use for the shortest path calculation. Options are 'sorted' and 'hashed'.
    hashtable : dict, optional
        Hashtable to use for the 'hashed' algorithm.

    Returns
    -------
    dict
        A dictionary mapping each node to the mean interval between peaks in travel times.
    """
    node_to_intervals = {}  # Dictionary to hold intervals for each node

    # Iterate over each sampling interval
    for current_time in range(start_time, end_time, sampling_interval):
        _, _, travel_times = single_source_time_dependent_dijkstra(
            graph, source, current_time, hashtable=hashtable, algorithm=algorithm
        )

        # Update travel times for each node
        for node, travel_time in travel_times.items():
            # Add the current time to the list of travel times for the node
            if node not in node_to_intervals:
                node_to_intervals[node] = []
            node_to_intervals[node].append(travel_time)

    # Calculate mean interval between peaks for each node
    # Node_to_interval contains the travel times list for each node
    for node, times in node_to_intervals.items():
        if len(times) > 1:
            time_array = np.array(times)

            # Calculate the first and second derivatives of the travel times
            # to find the peaks
            derivative = np.gradient(time_array)
            second_derivative = np.gradient(derivative)
            peaks, _ = scipy.signal.find_peaks(np.abs(second_derivative))

            # If there are peaks, calculate the intervals between them
            if len(peaks) > 1:
                intervals = np.diff(peaks) * sampling_interval
                mean_interval = np.mean(intervals)
                node_to_intervals[node] = mean_interval
            # If there are no peaks, set the interval to NaN
            # (pedestrian-only paths)
            else:
                node_to_intervals[node] = np.nan
        else:
            node_to_intervals[node] = np.nan

    return node_to_intervals
