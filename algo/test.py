import compute_merw as merw
import scipy.sparse as sparse
import numpy as num


def print_similarity_matrix(S):
    if not sparse.issparse(S):
        S = sparse.csr_matrix(S)
    n, m = S.get_shape()
    for row in range(n):
        print('[', end='')
        for col in range(m):
            print('{:6.3f} '.format(S[row, col]), end='')
        print(']')


def print_adiacency_matrix(A):
    n, m = A.get_shape()
    for row in range(n):
        print('[', end='')
        for col in range(m):
            if A[row, col] == 1:
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
    print('Samo GRW')
    P, dist = merw.compute_grw(A)
    print('Rozkład stacjonarny: ', dist)
    print_similarity_matrix(P)


def test_pdistance(graph):
    A = merw.graph_to_matrix(graph)
    Pgrw, vekt = merw.compute_grw(A)
    # print(vekt * Pgrw)
    Pmerw, val, vekt, dist = merw.compute_merw(A)
    R, eps = merw.compute_P_distance(Pgrw)
    print("\nP-distance GRW")
    print_similarity_matrix(R)
    print(' Dokładność:', eps)
    R, eps = merw.compute_P_distance(Pmerw)
    print("P-distance MERW")
    print_similarity_matrix(R)
    print(' Dokładność:', eps)


def test_pdistance_alpha(graph):
    A = merw.graph_to_matrix(graph)
    Pgrw, vekt = merw.compute_grw(A)
    for a in [0.1, 0.2, 0.3, 0.4, 0.5 ,0.6, 0.7, 0.8, 0.9, 0.999]:
        print("\nP-distance GRW: Alpha=", a)
        R, eps = merw.compute_P_distance(Pgrw, alpha=a, precision=1e-10)
        diag = sparse.linalg.inv(sparse.diags([R.diagonal()], [0], format='csc'))
        # każdy wiersz "normujemy" do wyrazu na przekątnej
        print_similarity_matrix(diag * R)
        print(' Dokładność:', eps)


if __name__ == '__main__':  # Odrobina testów
    graph1 = [[1, 2], [0, 2], [0, 1, 3, 4], [2, 4],
              [2, 3, 5], [4, 6], [5, 7, 8], [6, 8],
              [6, 7, 9, 10], [8,10], [8,9]]
    # Tego nie chce mi się rysować.
    graph2 = [[1, 2, 3, 4, 5], [0, 2], [0, 1, 3], [0, 2, 4], [0, 3, 5], [0, 4],
              [7, 8, 9], [6], [6], [6, 10], [9]]
    #
    #  1 - 2 - 3 - 4 - 5    7 - 6 - 8
    #    \  \  |  /  /          |
    #      --- 0 ---            9 - 10
    #
    unit_test(graph1)
    # for g in merw.get_all_components(graph2):
    #    unit_test(g)
    test_pdistance_alpha(graph1)
