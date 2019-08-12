import json
from os import path
from scipy.sparse import lil_matrix, csr_matrix, csc_matrix
import numpy as np
from dataset_checks import check_dataset


def get_filepath(base_path, name, ds_index, ds_type_string):
    fname = '{:03}_{}.{}.csv'.format(ds_index, name, ds_type_string)
    return path.join(base_path, fname)


def get_basic_edge_list(filepath):
    edges = []
    with open(filepath, 'r', encoding='utf-8') as file:
        for line in file:
            tokens = line.split("\t", 2)
            edges.append((int(tokens[0]), int(tokens[1])))
    return edges


def get_basic_edge_set(filepath):
    edges = set()
    with open(filepath, 'r', encoding='utf-8') as file:
        for line in file:
            tokens = line.split("\t", 2)
            edges.add((int(tokens[0]), int(tokens[1])))
    return edges


FORMAT_TO_LOADER = {'basic-edge-list': get_basic_edge_list,
                    'basic-dge-set': get_basic_edge_set}


def proj_2(_, x):
    return x


def edge_list_to_sparse_lil(size, edges):
    mx = lil_matrix((size, size))
    for v1, v2 in edges:
        mx[v1, v2] = 1
        mx[v2, v1] = 1
    return mx


def edge_list_to_dense(size, edges):
    mx = np.zeros(size * size)
    mx.reshape((size, size))
    for v1, v2 in edges:
        mx[v1, v2] = 1
        mx[v2, v1] = 1
    return mx


def prep_sparse_matrix_args(edges):
    ii, jj = [], []
    for v1, v2 in edges:
        ii.append(v1)
        ii.append(v2)
        jj.append(v2)
        jj.append(v1)
    return ii, jj, [1] * len(ii)


def edge_list_to_sparse_csr(size, edges):
    rows, cols, data = prep_sparse_matrix_args(edges)
    return csr_matrix((data, (rows, cols)), (size, size), 'd')


def edge_list_to_sparse_csc(size, edges):
    rows, cols, data = prep_sparse_matrix_args(edges)
    return csc_matrix((data, (rows, cols)), (size, size), 'd')


MODE_EDGES_TO_OUTPUT = {'edge_list': proj_2,
                        'adjacency_matrix_lil': edge_list_to_sparse_lil,
                        'adjacency_matrix_csr': edge_list_to_sparse_csr,
                        'adjacency_matrix_csc': edge_list_to_sparse_csc,
                        'adjacency_matrix_d': edge_list_to_dense}


def basic_load_training_set(base_path, name, ds_index, _, format_type):
    filepath = get_filepath(base_path, name, ds_index, 'train')
    if format_type not in FORMAT_TO_LOADER:
        raise BaseException('Unsupported training set format type.')
    return FORMAT_TO_LOADER[format_type](filepath)


def basic_load_test_set(base_path, name, ds_index, _, format_type):
    filepath = get_filepath(base_path, name, ds_index, 'test')
    if format_type not in FORMAT_TO_LOADER:
        raise BaseException('Unsupported training set format type.')
    return FORMAT_TO_LOADER[format_type](filepath)


def experience_load_test_set(base_path, name, ds_index, _, format_type):
    if format_type not in FORMAT_TO_LOADER:
        raise BaseException('Unsupported training set format type.')
    testpath = get_filepath(base_path, name, ds_index, 'test')
    trainpath = get_filepath(base_path, name, ds_index, 'train')
    train_nodes = set()
    for v1, v2 in FORMAT_TO_LOADER[format_type](trainpath):
        train_nodes.add(v1)
        train_nodes.add(v2)
    return [(v1,v2) for v1,v2 in FORMAT_TO_LOADER[format_type](testpath) if v1 in train_nodes and v2 in train_nodes]


def k_cross_load_training_set(base_path, name, ds_index, ds_count,
                              format_type):
    if format_type not in FORMAT_TO_LOADER:
        raise BaseException('Unsupported training set format type.')
    edges = []
    for i in range(1, ds_count + 1):
        if i != ds_index:
            filepath = get_filepath(base_path, name, i, 'cross')
            edges += FORMAT_TO_LOADER[format_type](filepath)
    return edges


def k_cross_load_test_set(base_path, name, ds_index, _, format_type):
    filepath = get_filepath(base_path, name, ds_index, 'cross')
    if format_type not in FORMAT_TO_LOADER:
        raise BaseException('Unsupported training set format type.')
    return FORMAT_TO_LOADER[format_type](filepath)


SPLIT_METHOD_TO_LOADERS = {'random': (basic_load_training_set,
                                      basic_load_test_set),
                           'k-cross-random': (k_cross_load_training_set,
                                              k_cross_load_test_set),
                           'chrono-perc': (basic_load_training_set,
                                           basic_load_test_set),
                           'chrono-from': (basic_load_training_set,
                                           basic_load_test_set),
                           'chrono-perc-old': (basic_load_training_set,
                                               experience_load_test_set)}


class DataSet:

    name = ""
    base_path = ""
    vx_count = 0
    edge_count = 0
    training_egdes = 0
    test_edges = 0
    set_count = 0
    format_type = ""
    split_method = ""

    def __init__(self, base_path, category, ds_name):
        full_path = path.join(base_path, category, ds_name)
        meta_path = path.join(full_path, '{}_meta.json'.format(ds_name))
        with open(meta_path, 'r', encoding='utf-8') as file:
            md = json.load(file)
        self.name = ds_name
        self.base_path = full_path
        self.vx_count = md["vertices"]
        self.edge_count = md["edges"]
        self.train_size = md["training_sets_size"]
        self.test_size = md["test_sets_size"]
        self.format_type = md["format_type"]
        self.split_method = md["split_method"]
        self.set_count = md["set_count"]

    def throw_if_index_oob(self, ds_index):
        if ds_index < 1 or ds_index > self.set_count:
            raise BaseException("Given test set index is invalid " +
                                "(given: {}, min: {}, max: {}"
                                .format(ds_index, 1, self.set_count))

    def get_training_set(self, ds_index=1, mode='edge_list'):
        self.throw_if_index_oob(ds_index)
        if self.split_method not in SPLIT_METHOD_TO_LOADERS:
            raise BaseException('Unsupported dataset split type.')
        edges = \
            SPLIT_METHOD_TO_LOADERS[self.split_method][0](self.base_path,
                                                          self.name,
                                                          ds_index,
                                                          self.set_count,
                                                          self.format_type)

        if mode not in MODE_EDGES_TO_OUTPUT:
            raise BaseException('Unsupported training set output mode.')
        return MODE_EDGES_TO_OUTPUT[mode](self.vx_count, edges)

    def get_test_edges(self, ds_index=1):
        self.throw_if_index_oob(ds_index)
        if self.split_method not in SPLIT_METHOD_TO_LOADERS:
            raise BaseException('Unsupported dataset split type.')
        return \
            SPLIT_METHOD_TO_LOADERS[self.split_method][1](self.base_path,
                                                          self.name,
                                                          ds_index,
                                                          self.set_count,
                                                          self.format_type)

    def get_dataset(self, ds_index=1, ts_mode='edge_list', do_check=False):
        self.throw_if_index_oob(ds_index)

        trn_data = self.get_training_set(ds_index, ts_mode)
        tst_data = self.get_test_edges(ds_index)

        if do_check:
            errors = check_dataset(trn_data, tst_data, ts_mode)
            if errors:
                raise BaseException("Invalid form of dataset being loaded:\n" +
                                    errors)

        return (trn_data, tst_data)
