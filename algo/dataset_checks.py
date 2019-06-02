import utils
import numpy as np
from scipy.sparse.csgraph import connected_components
from scipy.sparse import csr_matrix

# DISJOINTNESS OF TRAINING AND TEST SETS

def are_training_and_test_sets_disjoint(training_edges, test_edges):
    trns = utils.get_edges_set(training_edges),
    tsts = utils.get_edges_set(test_edges)

    for edge in trns:
        if edge in tsts:
            return False
    return True


# GRAPH IS UNDIRECTED (SYMMETRY OF REPRESENTATION)

def is_symmetric_s(sparse_mx):
    for i, j in zip(sparse_mx.nonzero()[0], sparse_mx.nonzero()[1]):
        if not sparse_mx[j, i] > 0:
            return False
    return True


def is_symmetric_d(dense_mx):
    return np.allclose(dense_mx, dense_mx.T, atol=1e-8)


def is_symmetric_el(edge_list):
    return True


def is_symmetric(data, mode):
    if mode == 'edge_list':
        return is_symmetric_el(data)
    elif mode == 'adjacency_matrix_d':
        return is_symmetric_d(data)
    else:
        return is_symmetric_s(data)


# GRAPH HAS NO LOOPS (REPRESENTATION HAS EMPTY DIAGONAL)

def has_no_loops_s(sparse_mx):
    for i, j in zip(sparse_mx.nonzero()[0], sparse_mx.nonzero()[1]):
        if i == j:
            return False
    return True


def has_no_loops_d(dense_mx):
    for i in range(0, dense_mx.shape[0]):
        if dense_mx[i, i] != 0:
            return False
        return True


def has_no_loops_el(edge_list):
    for i, j in edge_list:
        if i == j:
            return False
    return True


def has_no_loops(data, mode):
    if mode == 'edge_list':
        return has_no_loops_el(data)
    elif mode == 'adjacency_matrix_d':
        return has_no_loops_d(data)
    else:
        return has_no_loops_s(data)


# WHOLE GRAPH (BOTH TEST AND TRAINING EDGES) IS CONNECTED

def check_connected_s_el(trn_mx_sparse, test_edges):
    for i, j in test_edges:
        trn_mx_sparse[i, j] = 1
        trn_mx_sparse[j, i] = 1

    cc_count = connected_components(trn_mx_sparse,
                                    directed=False, return_labels=False)
    return cc_count == 1


def check_connected(trn_data, test_edges, ts_mode):
    data = trn_data
    if ts_mode == 'edge_list':
        ii, jj = [], []
        size = len(trn_data)
        for v1, v2 in trn_data:
            ii.append(v1)
            ii.append(v2)
            jj.append(v2)
            jj.append(v1)
        data = csr_matrix(([1] * len(ii), (ii, jj)), (size * 2, size * 2), 'd')
    elif ts_mode == 'adjacency_matrix_d':
        data = csr_matrix(trn_data)

    return check_connected_s_el(data, test_edges)


# MAIN TESTING FUNCTION

def check_dataset(training_data, test_edges, ts_mode):
    result = ""
    if not is_symmetric(training_data, ts_mode):
        result += "Training data representation is not symmetric.\n"
    if not is_symmetric(test_edges, "edge_list"):
        result += "Test data representation is not symmetric.\n"
    if not has_no_loops(training_data, ts_mode):
        result += "Training data representation has loops.\n"
    if not has_no_loops(test_edges, "edge_list"):
        result += "Test data representation has loops.\n"
    if (ts_mode == "edge_list" and
       not are_training_and_test_sets_disjoint(training_data, test_edges)):
        result += "Training and test data sets are not disjoint.\n"
    if not check_connected(training_data, test_edges, ts_mode):
        result += "Whole data set represents a graph that is not connected.\n"

    if len(result) == 0:
        result = None

    return result
