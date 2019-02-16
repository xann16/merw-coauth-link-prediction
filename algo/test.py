import compute_merw as merw
import numpy as num


def print_similarity_matrix(S):
    for row in S:
        print('[', end='')
        for value in row:
            print('{:6.3f} '.format(value), end='')
        print(']')


def print_adiacency_matrix(A):
    for row in A:
        print('[', end='')
        for val in row:
            if val == 1:
                print('X ', end='')
            else:
                print('. ', end='')
        print(']')


def unit_test(graph):
    A = merw.graph_to_matrix(graph)
    print('\n>>>> GRAF: ')
    print_adiacency_matrix(A)
    Rsim, eps = merw.compute_basic_simrank(graph, .9, 0, 1000)
    print('Zwykły SimRank')
    print_similarity_matrix(Rsim)
    print('Dokładność:', eps)
    Rmerw, eps = merw.compute_merw_simrank(graph, 1., 0, 1000)
    print('MERW SimRank')
    print_similarity_matrix(Rmerw)
    print('Dokładność:', eps)
    print('Samo MERW')
    P, val, vekt, dist = merw.compute_merw(A)
    print('Rozkład stacjonarny: ', dist)
    print_similarity_matrix(P)


if __name__ == '__main__':  # Odrobina testów
    graph1 = [[1, 2], [0, 2], [0, 1, 3, 4], [2, 4],
              [2, 3, 5], [4, 6], [5, 7, 8], [6, 8],
              [6, 7, 9, 10], [8,10], [8,9]]
    graph2 = [[1, 2, 3, 4, 5], [0, 2], [0, 1, 3], [0, 2, 4], [0, 3, 5], [0, 4],
              [7, 8, 9, 10], [6], [6], [6], [6]]
    unit_test(graph1)
    for g in merw.get_all_components(graph2):
        unit_test(g)
