import json
from os import path
from scipy.sparse import lil_matrix


def __get_basic_edge_list(filepath):
    edges = []
    with open(filepath, 'r', encoding='utf-8') as file:
        for line in file:
            tokens = line.split("\t", 2)
            edges.append(int(tokens[0]), int(tokens[1]))
    return edges


__FORMAT_TO_LOADER = {'basic_edge_list': __get_basic_edge_list}


def __proj_2(_, x):
    return x


def __edge_list_to_sparse_lil(size, edges):
    mx = lil_matrix((size, size), dtype='uint8')
    for v1, v2 in edges:
        mx[v1, v2] = 1
    return mx


__MODE_EDGES_TO_OUTPUT = {'edge_list': __proj_2,
                          'adjacency_matrix_lil': __edge_list_to_sparse_lil}


class DataSet:

    name = ""
    base_path = ""
    vx_count = 0
    edge_count = 0
    training_egdes = 0
    test_edges = 0
    is_maxcc = False
    set_count = 0
    format_type = ""

    def __init__(self, base_path, category, ds_name):
        full_path = path.join(base_path, category, ds_name)
        meta_path = path.join(base_path, '{}_meta.json'.format(ds_name))
        with open(meta_path, 'r', encoding='utf-8') as file:
            md = json.load(file)
        self.name = ds_name
        self.base_path = full_path
        self.vx_count = md["vertices"]
        self.edge_count = md["edges"]
        self.train_size = md["training_sets_size"]
        self.test_size = md["test_sets_size"]
        self.is_maxcc = md["is_maxcc"]
        self.format_type = md["format_type"]
        self.set_count = md["set_count"]

    def __get_filepath(self, ds_index, ds_type_string):
        fname = '{:03}_{}.{}.csv'.format(ds_index, self.name, ds_type_string)
        return path.join(self.base_path, fname)

    def __throw_if_index_oob(self, ds_index):
        if ds_index < 1 or ds_index > self.set_count:
            raise BaseException("Given test set index is invalid " +
                                "(given: {}, min: {}, max: {}"
                                .format(ds_index, 1, self.set_count))

    def get_training_set(self, ds_index=1, mode='edge_list'):
        self.__throw_if_index_oob(ds_index)
        filepath = self.__get_filepath(ds_index, 'train')
        if self.format_type not in __FORMAT_TO_LOADER:
            raise BaseException('Unsupported training set format type.')
        edges = __FORMAT_TO_LOADER[self.format_type](filepath)

        if mode not in __MODE_EDGES_TO_OUTPUT:
            raise BaseException('Unsupported training set output mode.')
        return __MODE_EDGES_TO_OUTPUT[mode](self.vx_count, edges)

    def get_test_edges(self, ds_index=1):
        self.__throw_if_index_oob(ds_index)
        filepath = self.__get_filepath(ds_index, 'test')
        if self.format_type not in __FORMAT_TO_LOADER:
            raise BaseException('Unsupported training set format type.')
        return __FORMAT_TO_LOADER[self.format_type](filepath)

    def get_dataset(self, ds_index=1, ts_mode='edge_list'):
        self.__throw_if_index_oob(ds_index)
        return (self.get_training_set(ds_index, ts_mode),
                self.get_test_edges(ds_index))
