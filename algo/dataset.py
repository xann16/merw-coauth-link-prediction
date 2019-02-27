import json
from os import path
from scipy.sparse import lil_matrix


def __get_filepath(base_path, name, ds_index, ds_type_string):
    fname = '{:03}_{}.{}.csv'.format(ds_index, name, ds_type_string)
    return path.join(base_path, fname)


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


def __basic_load_training_set(base_path, name, ds_index, _, format_type):
    filepath = __get_filepath(base_path, name, ds_index, 'train')
    if format_type not in __FORMAT_TO_LOADER:
        raise BaseException('Unsupported training set format type.')
    return __FORMAT_TO_LOADER[format_type](filepath)


def __basic_load_test_set(base_path, name, ds_index, _, format_type):
    filepath = __get_filepath(base_path, name, ds_index, 'test')
    if format_type not in __FORMAT_TO_LOADER:
        raise BaseException('Unsupported training set format type.')
    return __FORMAT_TO_LOADER[format_type](filepath)


def __k_cross_load_training_set(base_path, name, ds_index, ds_count,
                                format_type):
    if format_type not in __FORMAT_TO_LOADER:
        raise BaseException('Unsupported training set format type.')
    edges = []
    for i in range(1, ds_count + 1):
        if i != ds_index:
            filepath = __get_filepath(base_path, name, ds_index, 'train')
            edges.append(__FORMAT_TO_LOADER[format_type](filepath))
    return


def __k_cross_load_test_set(base_path, name, ds_index, _, format_type):
    filepath = __get_filepath(base_path, name, ds_index, 'cross')
    if format_type not in __FORMAT_TO_LOADER:
        raise BaseException('Unsupported training set format type.')
    return __FORMAT_TO_LOADER[format_type](filepath)


__SPLIT_METHOD_TO_LOADERS = {'random': (__basic_load_training_set,
                                        __basic_load_test_set),
                             'k-cross-random': (__k_cross_load_training_set,
                                                __k_cross_load_test_set)}


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
    split_method = ""

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
        self.split_method = md["split_method"]
        self.set_count = md["set_count"]

    def __throw_if_index_oob(self, ds_index):
        if ds_index < 1 or ds_index > self.set_count:
            raise BaseException("Given test set index is invalid " +
                                "(given: {}, min: {}, max: {}"
                                .format(ds_index, 1, self.set_count))

    def get_training_set(self, ds_index=1, mode='edge_list'):
        self.__throw_if_index_oob(ds_index)
        if self.split_method not in __SPLIT_METHOD_TO_LOADERS:
            raise BaseException('Unsupported dataset split type.')
        edges = \
            __SPLIT_METHOD_TO_LOADERS[self.split_method][0](self.base_path,
                                                            self.name,
                                                            ds_index,
                                                            self.set_count,
                                                            self.format_type)

        if mode not in __MODE_EDGES_TO_OUTPUT:
            raise BaseException('Unsupported training set output mode.')
        return __MODE_EDGES_TO_OUTPUT[mode](self.vx_count, edges)

    def get_test_edges(self, ds_index=1):
        self.__throw_if_index_oob(ds_index)
        if self.split_method not in __SPLIT_METHOD_TO_LOADERS:
            raise BaseException('Unsupported dataset split type.')
        return \
            __SPLIT_METHOD_TO_LOADERS[self.split_method][1](self.base_path,
                                                            self.name,
                                                            ds_index,
                                                            self.set_count,
                                                            self.format_type)

    def get_dataset(self, ds_index=1, ts_mode='edge_list'):
        self.__throw_if_index_oob(ds_index)
        return (self.get_training_set(ds_index, ts_mode),
                self.get_test_edges(ds_index))
