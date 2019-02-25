from scipy.sparse import lil_matrix
from scipy.sparse.csgraph import connected_components


def edges_to_simple_graph(size, edges):
    graph = lil_matrix((size, size), dtype="uint8")
    for v1, v2 in edges:
        graph[v1, v2] = 1
    return graph


def simple_graph_to_edges(graph):
    if graph.shape[0] != graph.shape[1]:
        raise BaseException("Processed graph dimension mismatch")
    edges = []
    v1s, v2s = graph.nonzero()
    for offset in range(0, len(v1s)):
        i, j = v1s[offset], v2s[offset]
        if graph[i, j] > 0:
            edges.append((i, j))
    return graph.shape[0], int(len(edges) / 2), edges


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


def is_connected(graph):
    cc_count = connected_components(graph,
                                    directed=False,
                                    return_labels=False)
    return cc_count == 1


def extract_maxcc_simple(size, edges):
    graph = edges_to_simple_graph(size, edges)
    _, labels = connected_components(graph, directed=False)
    max_label = get_maxcc_label(labels)
    graph = graph[:, labels == max_label]
    graph = graph[labels == max_label]
    if not is_connected(graph):
        raise BaseException("Failed to extract maximal closed component.")
    return simple_graph_to_edges(graph)


def extract_maximal_connected_component(vx_count, edges):
    return extract_maxcc_simple(vx_count, edges)
