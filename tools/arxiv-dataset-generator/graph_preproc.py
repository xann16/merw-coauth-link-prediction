from scipy.sparse import lil_matrix
from scipy.sparse.csgraph import connected_components
import calendar


# GENERAL UTILS

def get_label_counts(labels):
    counts = {}
    for label in labels:
        key = int(label)
        if key in counts:
            counts[key] += 1
        else:
            counts[key] = 1
    return counts


def get_maxcc_label(labels):
    max_label = -1
    max_count = 0
    counts = get_label_counts(labels)
    for label in counts:
        if counts[label] > max_count:
            max_count = counts[label]
            max_label = label
    return max_label


def extract_maxcc(graph):
    _, labels = connected_components(graph, directed=False)
    max_label = get_maxcc_label(labels)
    graph = graph[:, labels == max_label]
    graph = graph[labels == max_label]
    return graph


def extract_maxcc_and_add_edges(graph, late_edges):
    _, labels = connected_components(graph, directed=False)
    max_label = get_maxcc_label(labels)
    append_ts_edges(graph, late_edges)
    graph = graph[:, labels == max_label]
    graph = graph[labels == max_label]
    return graph


# TESTING OF OUTPUT

def is_square(graph):
    return graph.shape[0] == graph.shape[1]


def is_connected(graph):
    cc_count = connected_components(graph,
                                    directed=False,
                                    return_labels=False)
    return cc_count == 1


def is_symmetric(graph):
    for i, j in zip(graph.nonzero()[0], graph.nonzero()[1]):
        if not graph[j, i] > 0:
            return False
    return True


def check_output(graph):
    if not is_square(graph):
        raise BaseException("Extracted graph adjacency matrix is not square.")
    if not is_connected(graph):
        raise BaseException("Failed to extract maximal closed component.")
    if not is_symmetric(graph):
        raise BaseException("Extracted graph's adjacency matrix is not" +
                            " symmetric.")


# SIMPLE GRAPH UTILS (i.e  no chrono labels)

def edges_to_simple_graph(size, edges):
    graph = lil_matrix((size, size), dtype="uint8")
    for v1, v2 in edges:
        graph[v1, v2] = 1
        graph[v2, v1] = 1
    return graph


def simple_graph_to_edge_data(graph):
    check_output(graph)
    edge_set = set()
    for i, j in zip(graph.nonzero()[0], graph.nonzero()[1]):
        if graph[i, j] > 0:
            if i < j:
                edge_set.add((i, j))
            else:
                edge_set.add((j, i))
    return graph.shape[0], len(edge_set), list(edge_set)


# GRAPHS WITH CHRONO DATA

def ts_edges_to_utc_graph(size, ts_edges):
    graph = lil_matrix((size, size), dtype="uint64")
    for ts, (v1, v2) in ts_edges:
        utc_ts = calendar.timegm(ts.utctimetuple())
        if graph[v1, v2] == 0 or utc_ts < graph[v1, v2]:
            graph[v1, v2] = utc_ts
            graph[v2, v1] = utc_ts
    return graph


def append_ts_edges(graph, new_edges):
    for ts, (v1, v2) in new_edges:
        utc_ts = calendar.timegm(ts.utctimetuple())
        if graph[v1, v2] == 0 or utc_ts < graph[v1, v2]:
            graph[v1, v2] = utc_ts
            graph[v2, v1] = utc_ts


def get_uts_ts(utc_edge):
    return utc_edge[0]


def utc_graph_to_chrono_ordered_edge_data(graph):
    check_output(graph)
    utc_edges_set = set()
    for i, j in zip(graph.nonzero()[0], graph.nonzero()[1]):
        utc_ts = graph[i, j]
        if utc_ts > 0:
            if i < j:
                utc_edges_set.add((utc_ts, i, j))
            else:
                utc_edges_set.add((utc_ts, j, i))
    edges = list(utc_edges_set)
    edges.sort(key=get_uts_ts)
    return graph.shape[0], len(edges), edges


# PREPROCESSING METHODS

def preprocess_simple_graph(vx_count, edges):
    graph = extract_maxcc(edges_to_simple_graph(vx_count, edges))
    return simple_graph_to_edge_data(graph)


def preprocess_chrono_graph(vx_count, ts_edges):
    graph = extract_maxcc(ts_edges_to_utc_graph(vx_count, ts_edges))
    return utc_graph_to_chrono_ordered_edge_data(graph)


def preprocess_chrono_graph_oldedges(vx_count, ts_edges_tuple):
    ts_edges, ts_edges2 = ts_edges_tuple
    graph = extract_maxcc_and_add_edges(ts_edges_to_utc_graph(vx_count, ts_edges), ts_edges2)
    return utc_graph_to_chrono_ordered_edge_data(graph)
