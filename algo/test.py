import compute_merw as merw
import scipy.sparse as sparse
import metrics
import dataset


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


def __unit_test(graph):
    A = merw.graph_to_matrix(graph)
    n = len(graph)
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
    P, vekt, val, dist = merw.compute_merw(A)
    print('Rozkład stacjonarny: ', dist)
    print_similarity_matrix(P)
    print('Eigen-wieghted:')
    for v in range(n):
        for w in graph[v]:
            if v < w:
                print('(', v+1, w+1, ')', vekt[v]*vekt[w]/val)
    print('Samo GRW')
    P, dist = merw.compute_grw(A)
    print('Rozkład stacjonarny: ', dist)
    print_similarity_matrix(P)


def __test_pdistance(graph):
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


def __test_pdistance_alpha(graph):
    A = merw.graph_to_matrix(graph)
    Pgrw, vekt = merw.compute_grw(A)
    for a in [0.1, 0.2, 0.3, 0.4, 0.5 ,0.6, 0.7, 0.8, 0.9, 0.999]:
        print("\nP-distance GRW: Alpha=", a)
        R = merw.compute_P_distance(Pgrw, alpha=a)
        diag = sparse.linalg.inv(sparse.diags([R.diagonal()], [0], format='csc'))
        # każdy wiersz "normujemy" do wyrazu na przekątnej
        print_similarity_matrix(diag * R)
        #print(' Dokładność:', eps)
        #R = merw.compute_P_distance_exact(Pgrw, alpha=a)
        #diag = sparse.linalg.inv(sparse.diags([R.diagonal()], [0], format='csc'))
        #print_similarity_matrix(diag * R)

def __experiment_plain():
    graph1 = [[1, 2], [0, 2], [0, 1, 3, 4], [2, 4],
              [2, 3, 5], [4, 6], [5, 7, 8], [6, 8],
              [6, 7, 9, 10], [8, 10], [8, 9]]
    # Tego nie chce mi się rysować.
    graph2 = [[1, 2, 3, 4, 5], [0, 2], [0, 1, 3], [0, 2, 4], [0, 3, 5], [0, 4],
              [7, 8, 9], [6], [6], [6, 10], [9]]
    graph_o = [[1], [0, 2, 3], [1, 3], [1, 2, 4], [3]]
    #
    #  1 - 2 - 3 - 4 - 5    7 - 6 - 8
    #    \  \  |  /  /          |
    #      --- 0 ---            9 - 10
    #
    __unit_test(graph1)
    __unit_test(graph_o)
    # for g in merw.get_all_components(graph2):
    #    unit_test(g)
    #__test_pdistance_alpha(graph1)


def __experiment_01(data_set, skipSimRank=False, set_no=1, a=0.5, aucn=2000):
    data = dataset.DataSet('../datasets/', 'gr-qc', data_set)
    matrix = sparse.csc_matrix(
        data.get_training_set(mode='adjacency_matrix_lil', ds_index=set_no), dtype='f')
    training = metrics.get_edges_set(data.get_training_set())
    test = metrics.get_edges_set(data.get_test_edges())

    print('Zestaw',set_no,' N=', data.vx_count)
    print('Obliczanie: macierzy przejścia MERW...', end=' ')
    Pmerw, vekt, eval, stat = merw.compute_merw_matrix(matrix, method=merw.power_method)
    #print(vekt)
    #print(Pmerw.get_shape()[0])
    print('macierzy "odległości"...')
    p_dist_merw = merw.compute_P_distance(Pmerw, alpha=a)
    print('Obliczanie: macierzy przejścia GRW... ', end=' ')
    Pgrw, sd = merw.compute_grw(matrix)
    print('macierzy "odległości"...')
    p_dist_grw = merw.compute_P_distance(Pgrw, alpha=a)
    print('   Skuteczność PD (AUC {}):'.format(aucn),
          metrics.auc(data.vx_count, training, test, p_dist_grw, aucn))
    print(' Skuteczność MEPD (AUC {}):'.format(aucn),
          metrics.auc(data.vx_count, training, test, p_dist_merw, aucn))

    if skipSimRank:
        return
    graph = merw.matrix_to_graph(matrix)
    #print(graph)
    print('SimRank...')
    sr, eps = merw.compute_basic_simrank(graph, a, maxiter=30)
    print('Skuteczność (AUC 1000):',
          metrics.auc(data.vx_count, training, test, sr, 1000),
          ' Dokładność:', eps)
    print('MERW SimRank...')
    sr, eps = merw.compute_merw_simrank_ofmatrix(matrix, a, maxiter=30)
    print('Skuteczność (AUC 1000):',
          metrics.auc(data.vx_count, training, test, sr, 1000),
          ' Dokładność:',eps)


def __test_mat_merw():
    graph = [[1, 2], [0, 2], [0, 1, 3, 4], [2, 4],
              [2, 3, 5], [4, 6], [5, 7, 8], [6, 8],
              [6, 7, 9, 10], [8, 10], [8, 9]]
    A = merw.graph_to_matrix(graph)
    Q, v, val, dist = merw.compute_merw_matrix(A, method=merw.scipy_method)
    print('SciPy ', val, v)
    #print(Q*dist)
    print(dist)
    print(dist*Q)
    v, val = merw.power_method(A)
    print('Moje ', val, v)
    print(v*A)
    print(val*v)


if __name__ == '__main__':  # Odrobina testów
    #__experiment_plain()
    for i in range(1, 10):
        __experiment_01('eg1k', skipSimRank=True, set_no=i)
    #__test_mat_merw()
    #__experiment_01('basic-test')
