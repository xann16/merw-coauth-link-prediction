import sys
import random
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

#        dstTrainPath = '{}.train.csv'.format(dstName)
#        dstTrainPath = '{}.test.csv'.format(dstName)


def get_max_auth_id(srcPath):
    with open(srcPath, 'r', encoding='utf-8') as edges_file:
        maxId = 0
        for line in edges_file:
            if len(line) > 0 and line[0] != '#':
                tokens = line.split(';')
                id1 = int(tokens[1])
                id2 = int(tokens[2])
                if id1 > maxId:
                    maxId = id1
                if id2 > maxId:
                    maxId = id2
        return maxId


def load_graph_data(srcPath, size):
    graph = csr_matrix((size, size))
    with open(srcPath, 'r', encoding='utf-8') as edges_file:
        for line in edges_file:
            if len(line) > 0 and line[0] != '#':
                tokens = line.split(';')
                id1 = int(tokens[1])
                id2 = int(tokens[2])
                graph[id1, id2] = 1
                graph[id2, id1] = 1
    return graph


def get_maximal_conn_comp_label(labels):
    counts = {}

    for lbl in labels:
        key = int(lbl)
        if key in counts:
            counts[key] += 1
        else:
            counts[key] = 1

    maxLabel = -1
    maxCount = 0

    for lbl in counts:
        if counts[lbl] > maxCount:
            maxCount = counts[lbl]
            maxLabel = lbl

    return maxLabel


def graph_to_edge_list(graph):
    res = []
    for i in range(graph.shape[0]):
        for j in range(graph.shape[1]):
            if graph[i, j] > 0:
                res.append((i, j))
    return res


def write_edge_list(dstPath, edges):
    with open(dstPath, 'w', encoding='utf-8') as ds_file:
        for i, j in edges:
            ds_file.write('{};{}\n'.format(i, j))


def prepare_simple_dataset(srcPath, dstName):
    n = get_max_auth_id(srcPath) + 1
    graph = load_graph_data(srcPath, n)
    nComps, labels = connected_components(graph, directed=False)
    maxLabel = get_maximal_conn_comp_label(labels)
    res_graph = graph[:, labels == maxLabel]
    res_graph2 = res_graph[labels == maxLabel]
    edge_list = graph_to_edge_list(res_graph2)
    nEdges = len(edge_list)
    testSetSize = int(nEdges * 0.1)
    random.shuffle(edge_list)

    write_edge_list('{}.test.csv'.format(dstName), edge_list[:testSetSize])
    write_edge_list('{}.train.csv'.format(dstName), edge_list[testSetSize:])


if __name__ == '__main__':
    if len(sys.argv) < 2 or len(sys.argv) > 3:
        raise BaseException('Invalid number of command line arguments.')
    else:
        srcPath = sys.argv[1]
        dstName = sys.argv[2]

        print('Preparing randomized 90/10 (training/test) dataset based on ' +
              'edges from: {}'.format(srcPath))
        prepare_simple_dataset(srcPath, dstName)
        print('Data set successfully prepared.')
