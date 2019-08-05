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


def test_dataset_symmetry(data_set, set_no=1):
    data = dataset.DataSet('../datasets/', 'math.FA', data_set)
    matrix = sparse.csc_matrix(
        data.get_training_set(mode='adjacency_matrix_lil', ds_index=set_no), dtype='d')
    print('DATASET ', data_set)
    for i in range(data.vx_count):
        for j in range(i):
            if matrix[i, j] != matrix[j, i]:
                print("ERROR! ({},{})".format(i, j), end=' ')
    print(' OK')


def edges_to_deg(edges, n):
    degs = [0]*n
    for x,y in edges:
        degs[x] += 1
        degs[y] += 1
    return degs


def graph_stats(graph, degs, size):
    maxdeg = 0
    mindeg = len(graph)-1
    counts = [0]*len(graph)
    for v in graph:
        maxdeg = max(len(v), maxdeg)
        mindeg = min(len(v), mindeg)
        counts[len(v)] += 1
    print(" Degrees: min={} (x{}), max={} (x{})".format(mindeg, counts[mindeg], maxdeg, counts[maxdeg]))
    added = 0
    for i in range(len(graph)):
        if len(graph[i]) == 0:
            added += degs[i]
            #print("   +{}".format(degs[i]))
    print(" [!] \"New\" edges {}, ({:3.2f}%)".format(added, 50*added/size))


DATASETS = [ ('chrono_93_96_97', 'gr-qc', 'RQ1'), ('chrono_95_97_98', 'gr-qc', 'RQ2'),
                    ('random', 'gr-qc', 'RQR'), ('random20', 'gr-qc', 'RQR2'),
                    ('chrono_94_09_10', 'math.FA', 'FA1'), ('chrono_96_08_11', 'math.FA','FA2'),
                    ('chrono_97_09_11', 'math.FA', 'FA3'), ('random', 'math.FA', 'FAR'),
                    ('chrono_94_10_12', 'math.AT', 'AT1'), ('chrono_00_11_13', 'math.AT', 'AT2'),
                    ('random', 'math.AT', 'ATR')]


def __TeX_datainfo():
    for dset, cat, name in DATASETS:
        data = dataset.DataSet('../datasets/', cat, dset)
        print("\t{:4} & {:4} & {:4} & {:4} \\\\ \\hline".format(name, data.vx_count, data.train_size, data.test_size))
    print()

def __test_datasets():
    for dset, cat in [('std_gr-qc', 'import'), ('eg1k_rnd_std','gr-qc'), ('eg1k_chr_frm', 'gr-qc'), ('eg1k_chr_frm1995', 'gr-qc'),
            ('eg1k_chr_frm', 'math.GN'),  # ('eg1k_chr_prc', 'math.GN'), ('eg1k_rnd_kcv', 'math.GN'),('eg1k_chr_prc', 'gr-qc'),
            ('chrono_94_09_10', 'math.FA'), ('chrono_96_08_11', 'math.FA'), ('chrono_96_09_11', 'math.FA'),
            ('eg1k_rnd_std', 'math.FA')]:
        print("{} > {}".format(cat,dset))
        data = dataset.DataSet('../datasets/', cat, dset)
        for dsi in range(1,data.set_count+1):
            test_ed = data.get_test_edges(ds_index=dsi)
            print("#{} Vertices:".format(dsi), data.vx_count, "(Edges: {}/{})".format(data.train_size, data.test_size))
            matrix = data.get_training_set(mode='adjacency_matrix_csc', ds_index=dsi)
            graph = merw.matrix_to_graph(matrix)
            graph_stats(graph, edges_to_deg(test_ed, data.vx_count), len(test_ed))
        print()


def __experiment_01(data_set, skipSimRank=False, a=0.5, aucn=None, simrank_iter=10):
    dset, category, name= data_set
    print('\n"{}" ({})'.format(dset, category))
    data = dataset.DataSet('../datasets/', category, dset)
    print("\t{:4} & {:4} & {:4} & {:4} \\\\ \\hline".format(name, data.vx_count, data.train_size, data.test_size))
    if not aucn:
        aucn = 4 * data.train_size
    for set_no in range(1, data.set_count+1):
        matrix = sparse.csc_matrix(
            data.get_training_set(mode='adjacency_matrix_csc', ds_index=set_no), dtype='d')
        training = data.get_training_set(ds_index=set_no) #metrics.get_edges_set(data.get_training_set())
        test = data.get_test_edges(ds_index=set_no) #metrics.get_edges_set(data.get_test_edges())
        if data.set_count > 1:
            print('>> Zestaw', set_no)
        print('  PD...', end=' ')
        Pgrw, sd = merw.compute_grw(matrix)
        p_dist_grw = merw.compute_P_distance(Pgrw, alpha=a)
        aucPD = metrics.auc(data.vx_count, training, test, p_dist_grw, aucn)
        print('MEPD...', end=' ')
        Pmerw, vekt, eval, stat = merw.compute_merw_matrix(matrix)
        p_dist_merw = merw.compute_P_distance(Pmerw, alpha=a)
        aucMEPD = metrics.auc(data.vx_count, training, test, p_dist_merw, aucn)
        print('PDM...', end=' ')
        ep_dist_grw = merw.compute_P_distance(Pgrw, alpha=a)
        aucPDM = metrics.auc(data.vx_count, training, test, ep_dist_grw, aucn)
        print('MEPDM...', end=' ')
        ep_dist_merw = merw.compute_P_distance(Pmerw, alpha=a)
        aucMEPDM = metrics.auc(data.vx_count, training, test, ep_dist_merw, aucn)
        if skipSimRank:
            endstr=''
        else:
            graph = merw.matrix_to_graph(matrix)
            #print(graph)
            print('SR...', end=' ')
            sr, eps = merw.general_simrank(graph, Pgrw, alpha=a, iterations=simrank_iter)
            aucSR = metrics.auc(data.vx_count, training, test, sr, aucn)
            print('({}) MESR...'.foramt(eps), end=' ')
            sr, eps = merw.general_simrank(graph, Pmerw, alpha=a, iterations=simrank_iter)
            print('({})'.foramt(eps), end='')
            aucMESR = metrics.auc(data.vx_count, training, test, sr, aucn)
            endstr = "& {:.3f} & {:.3f}".format(aucSR, aucMESR)
        print('\n\t{:.3f} & {:.3f} & {:.3f} & {:.3f} {}\\\\ \\hline'.format(aucPD, aucMEPD, aucPDM, aucMEPDM, endstr))
        del sr, eps, data, endstr, aucPD, aucMEPD, aucPDM, aucMEPDM, aucSR, aucMESR


def __experiment_02(data_set, set_no=1, aucn=2000, category='math.GN'):
    print('Kategoria: ', category)
    data = dataset.DataSet('../datasets/', category, data_set)
    matrix = sparse.csc_matrix(
        data.get_training_set(mode='adjacency_matrix_lil', ds_index=set_no), dtype='d')
    training = data.get_training_set() #metrics.get_edges_set(data.get_training_set())
    test = data.get_test_edges() #metrics.get_edges_set(data.get_test_edges())
    print('Rozmiar grafu=',data.vx_count)

    print('Obliczanie MERW i GRW...')
    Pgrw, sd = merw.compute_grw(matrix)
    Pmerw, vekt, eval, stat = merw.compute_merw_matrix(matrix)
    for a in [.1, .5, .9]:
        print('alfa=', a)
        p_dist = merw.compute_P_distance(Pgrw, alpha=a)
        print('   Skuteczność PD (AUC {}):'.format(aucn),
              metrics.auc(data.vx_count, training, test, p_dist, aucn))
        p_dist = merw.compute_P_distance(Pmerw, alpha=a)
        print(' Skuteczność MEPD (AUC {}):'.format(aucn),
              metrics.auc(data.vx_count, training, test, p_dist, aucn))
        p_dist = merw.compute_P_distance(Pgrw, alpha=a)
        print('  Skuteczność PDM (AUC {}):'.format(aucn),
              metrics.auc(data.vx_count, training, test, p_dist, aucn))
        p_dist = merw.compute_P_distance(Pmerw, alpha=a)
        print('Skuteczność MEPDM (AUC {}):'.format(aucn),
              metrics.auc(data.vx_count, training, test, p_dist, aucn))


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

#__TeX_datainfo()
#__test_datasets()
#exit(0)


if __name__ == '__main__':  # Odrobina testów
    #__experiment_plain()
    #__experiment_01('basic-test-bis', aucn=5000)
    #for i in range(1, 10):
        #test_dataset_symmetry('basic-test', set_no=i)
        #__experiment_01('basic-test', skipSimRank=True, set_no=i, aucn=5000)
    #__test_mat_merw()
    #__experiment_01('std_gr-qc', aucn=1000, category='import', skipSimRank=True)
    #__experiment_02('eg1k_rnd_std', aucn=1000, category='gr-qc')
    #__experiment_02('eg1k_chr_frm', aucn=1000, category='gr-qc')
    #__experiment_02('chrono_96_08_11', aucn=1000, category='math.FA')

    for data_set in DATASETS:
        __experiment_01(data_set, skipSimRank=True)
