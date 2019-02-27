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
    #
    #  1 - 2 - 3 - 4 - 5    7 - 6 - 8
    #    \  \  |  /  /          |
    #      --- 0 ---            9 - 10
    #
    __unit_test(graph1)
    # for g in merw.get_all_components(graph2):
    #    unit_test(g)
    __test_pdistance_alpha(graph1)


def __experiment_01(data_set, skipSimRank=False, set_no=1):
    data = dataset.DataSet('../datasets/', 'gr-qc', data_set)
    matrix = sparse.csc_matrix(
        data.get_training_set(mode='adjacency_matrix_lil', ds_index=set_no), dtype='f')

    training = data.get_training_set()
    test = data.get_test_edges()
    print('Zestaw',set_no,' N=', data.vx_count)
    print('Obliczanie: macierzy przejścia MERW...', end=' ')
    Pmerw, vekt, eval, stat = merw.compute_merw_matrix(matrix)
    #print(Pmerw.get_shape()[0])
    print('macierzy "odległości"...')
    p_dist_merw = merw.compute_P_distance(Pmerw, alpha=.83)
    print('Obliczanie macierzy przejścia GRW...', end=' ')
    Pgrw, sd = merw.compute_grw(matrix)
    print('macierzy "odległości"...')
    p_dist_grw = merw.compute_P_distance(Pgrw, alpha=.99)
    print('   Skuteczność PD (AUC 1000):', \
          metrics.auc(data.vx_count, training, test, p_dist_grw, 1000))
    #print('Skuteczność (AUC 2000):', \
    #     metrics.auc(data.vx_count, training, test, p_dist_grw, 2000))
    # print('Skuteczność (AUC 600):', \
    #      metrics.auc(data.vx_count, training, test, p_dist_merw, 600))
    print(' Skuteczność MEPD (AUC 1000):', \
          metrics.auc(data.vx_count, training, test, p_dist_merw, 1000))
    if skipSimRank:
        return
    graph = merw.matrix_to_graph(matrix)
    #print(graph)
    print('SimRank...')
    sr, eps = merw.compute_basic_simrank(graph, 0.9, maxiter=30)
    print('Skuteczność (AUC 400):', \
          metrics.auc(data.vx_count, training, test, sr, 400))
    print('Skuteczność (AUC 1000):', \
          metrics.auc(data.vx_count, training, test, sr, 1000))
    print('MERW SimRank...')
    sr, eps = merw.compute_merw_simrank_ofmatrix(matrix, 1.0, maxiter=30)
    print('Skuteczność (AUC 400):', \
          metrics.auc(data.vx_count, training, test, sr, 400))
    print('Skuteczność (AUC 1000):', \
          metrics.auc(data.vx_count, training, test, sr, 1000))

def __test_mat_merw():
    graph = [[1, 2], [0, 2], [0, 1, 3, 4], [2, 4],
              [2, 3, 5], [4, 6], [5, 7, 8], [6, 8],
              [6, 7, 9, 10], [8, 10], [8, 9]]
    A = merw.graph_to_matrix(graph)
    Q, v, val, dist = merw.compute_merw_matrix(A)
    print((Q*dist).toarray())
    print(dist)
    print((dist*Q).toarray())


if __name__ == '__main__':  # Odrobina testów
    #__experiment_plain()
    for i in range(1, 10):
        __experiment_01('eg1k', skipSimRank=True, set_no=i)
    #__test_mat_merw()
    __experiment_01('basic-test',set_no=3)
