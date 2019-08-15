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


DATASETS = [ ('gr-qc', 'RQ1'), ('gr-qc', 'RQ2'), ('gr-qc', 'RQR1'), ('gr-qc', 'RQR2'),
             ('math.FA', 'FA1'), ('math.FA', 'FA2'), ('math.FA', 'FA3'), ('math.FA', 'FAR1'), ('math.FA', 'FAR2'),
             ('math.AT', 'AT1'), ('math.AT', 'AT2'), ('math.AT', 'ATR1'), ('math.AT', 'ATR2')]


def __test_datasets():
    for dset, cat, _ in DATASETS:
        print("{} > {}".format(cat,dset))
        data = dataset.DataSet('../datasets/', cat, dset)
        for dsi in range(1,data.set_count+1):
            test_ed = data.get_test_edges(ds_index=dsi)
            print("#{} Vertices:".format(dsi), data.vx_count, "(Edges: {}/{})".format(data.train_size, data.test_size))
            matrix = data.get_training_set(mode='adjacency_matrix_csc', ds_index=dsi)
            graph = merw.matrix_to_graph(matrix)
            graph_stats(graph, edges_to_deg(test_ed, data.vx_count), len(test_ed))
        print()


def __experiment_01(data_set, skipSimRank=False, a=0.5, aucn=None, simrank_iter=10,
                    metric=metrics.auc):
    category, name = data_set
    # print('\n"{}" ({})'.format(name, category))
    data = dataset.DataSet('../datasets/', category, name)
    #print("\n{:4} & {:4} & {:4} & {:4} &  \\\\ \\hline".format(name, data.vx_count, data.train_size, data.test_size))
    if aucn is None:
        aucn = 7 * data.train_size
    aucPD = aucMEPD = aucPDM = aucMEPDM = aucSR = aucMESR = 0
    for set_no in range(1, data.set_count+1):
        matrix = sparse.csc_matrix(
            data.get_training_set(mode='adjacency_matrix_csc', ds_index=set_no), dtype='d')
        training = set(data.get_training_set(ds_index=set_no)) #metrics.get_edges_set(data.get_training_set())
        test = set(data.get_test_edges(ds_index=set_no)) #metrics.get_edges_set(data.get_test_edges())
        if data.set_count > 1:
            print(' %> #{}'.format(set_no), end='')
        else:
            print(' %', end='')
        print(' PD.', end='', flush=True)
        Pgrw, sd = merw.compute_grw(matrix)
        print('.', end='', flush=True)
        p_dist_grw = merw.P_distance(Pgrw, alpha=a)
        print('.', end=' ', flush=True)
        aucPD += metric(data.vx_count, training, test, p_dist_grw, aucn)
        print('MEPD.', end='', flush=True)
        Pmerw, vekt, eval, stat = merw.compute_merw_matrix(matrix)
        print('.', end='', flush=True)
        p_dist_merw = merw.P_distance(Pmerw, alpha=a)
        print('.', end=' ', flush=True)
        aucMEPD += metric(data.vx_count, training, test, p_dist_merw, aucn)
        print('PDM.', end='', flush=True)
        ep_dist_grw = merw.exp_P_distance(Pgrw, alpha=a)
        print('.', end=' ', flush=True)
        aucPDM += metric(data.vx_count, training, test, ep_dist_grw, aucn)
        print('MEPDM.', end='', flush=True)
        ep_dist_merw = merw.exp_P_distance(Pmerw, alpha=a)
        print('.', end=' ', flush=True)
        aucMEPDM += metric(data.vx_count, training, test, ep_dist_merw, aucn)
        if not skipSimRank:
            print('SR', end='')
            graph = merw.matrix_to_graph(matrix)
            sr, eps = merw.basic_simrank(graph, alpha=a, iterations=simrank_iter)
            aucSR += metric(data.vx_count, training, test, sr, aucn)
            print(' ({:.4f}) MESR'.format(eps), end='')
            sr, eps = merw.merw_simrank(graph, eigenvector=(vekt, eval), alpha=a, iterations=simrank_iter)
            print(' ({:.4f})'.format(eps))
            aucMESR += metric(data.vx_count, training, test, sr, aucn)
    if skipSimRank:
        endstr = ''
    else:
        endstr = "& {:.3f} & {:.3f}".format(aucSR/data.set_count, aucMESR/data.set_count)
    print('{:4} #& {:.3f} & {:.3f} & {:.3f} & {:.3f} {}\\\\ \\hline'
          .format(name, aucPD/data.set_count, aucMEPD/data.set_count,
          aucPDM/data.set_count, aucMEPDM/data.set_count, endstr))
    if not skipSimRank:
        del sr, eps, graph
    del data, endstr, aucPD, aucMEPD, aucPDM, aucMEPDM, aucSR, aucMESR


def __experiment_02(data_set, skipSimRank=False, a=0.5, aucn=2000, simrank_iter=10):
    category, name = data_set
    data = dataset.DataSet('../datasets/', category, name)
    aucPD = aucMEPD = aucPDM = aucMEPDM = aucSR = aucMESR = 0
    for set_no in range(1, data.set_count+1):
        matrix = sparse.csc_matrix(
            data.get_training_set(mode='adjacency_matrix_csc', ds_index=set_no), dtype='d')
        training = set(data.get_training_set(ds_index=set_no)) #metrics.get_edges_set(data.get_training_set())
        test = set(data.get_test_edges(ds_index=set_no)) #metrics.get_edges_set(data.get_test_edges())
        if data.set_count > 1:
            print('   %> #{}'.format(set_no), end='', flush=True)
        else:
            print('   %     ', end='', flush=True)
        absent = metrics.prepare_all_edges_list(data.vx_count, training, test)
        print(' PD.', end='', flush=True)
        Pgrw, sd = merw.compute_grw(matrix)
        print('.', end='', flush=True)
        p_dist_grw = merw.P_distance(Pgrw, alpha=a)
        print('.', end=' ', flush=True)
        aucPD += metrics.auc_semi_total(data.vx_count, training, test, p_dist_grw,
                                        count=aucn, absent_edges=absent)
        print('MEPD.', end='', flush=True)
        Pmerw, vekt, eval, stat = merw.compute_merw_matrix(matrix)
        print('.', end='', flush=True)
        p_dist_merw = merw.P_distance(Pmerw, alpha=a)
        print('.', end=' ', flush=True)
        aucMEPD += metrics.auc_semi_total(data.vx_count, training, test, p_dist_merw,
                                          count=aucn, absent_edges=absent)
        print('PDM.', end='', flush=True)
        ep_dist_grw = merw.exp_P_distance(Pgrw, alpha=a)
        print('.', end=' ', flush=True)
        aucPDM += metrics.auc_semi_total(data.vx_count, training, test, ep_dist_grw,
                                         count=aucn, absent_edges=absent)
        print('MEPDM.', end='', flush=True)
        ep_dist_merw = merw.exp_P_distance(Pmerw, alpha=a)
        print('.', end=' ', flush=True)
        aucMEPDM += metrics.auc_semi_total(data.vx_count, training, test, ep_dist_merw,
                           count=aucn, absent_edges=absent)
        if not skipSimRank:
            print('SR', end='')
            graph = merw.matrix_to_graph(matrix)
            sr, eps = merw.basic_simrank(graph, alpha=a, iterations=simrank_iter)
            aucSR += metrics.auc_semi_total(data.vx_count, training, test, sr,
                                            count=aucn, absent_edges=absent)
            print(' ({:.4f}) MESR'.format(eps), end='')
            sr, eps = merw.merw_simrank(graph, eigenvector=(vekt, eval), alpha=a, iterations=simrank_iter)
            print(' ({:.4f})'.format(eps))
            aucMESR += metrics.auc_semi_total(data.vx_count, training, test, sr,
                                              count=aucn, absent_edges=absent)
    if skipSimRank:
        endstr = ''
    else:
        endstr = "& {:.3f} & {:.3f}".format(aucSR/data.set_count, aucMESR/data.set_count)
    print('{:4} #& {:.3f} & {:.3f} & {:.3f} & {:.3f} {}\\\\ \\hline'
          .format(name, aucPD/data.set_count, aucMEPD/data.set_count,
          aucPDM/data.set_count, aucMEPDM/data.set_count, endstr))
    if not skipSimRank:
        del sr, eps, graph
    del data, endstr, aucPD, aucMEPD, aucPDM, aucMEPDM, aucSR, aucMESR


def __simrank_test(data_set, a=0.5, set_no=1, aucn=2000, simrank_iter=6):
    category, name = data_set
    data = dataset.DataSet('../datasets/', category, name)
    if data.set_count == 1:
        return
    #print('%{} -----------'.format(name))
    #print("\t{:4} & {:4} & {:4} & {:4} \\\\ \\hline".format(name, data.vx_count, data.train_size, data.test_size))
    aucSR = aucMESR = aucMESR2 = aucSR2 = 0
    for set_no in range(1, data.set_count+1):
        matrix = sparse.csc_matrix(
            data.get_training_set(mode='adjacency_matrix_csc', ds_index=set_no), dtype='d')
        training = set(data.get_training_set(ds_index=set_no))  # metrics.get_edges_set(data.get_training_set())
        test = set(data.get_test_edges(ds_index=set_no))  # metrics.get_edges_set(data.get_test_edges())
        graph = merw.matrix_to_graph(matrix)
        all_edges = metrics.prepare_all_edges_list(data.vx_count, training, test)
        print(' %#{} SR'.format(set_no), end='')
        sr, eps = merw.basic_simrank(graph, a, iterations=simrank_iter)
        aucSR += metrics.auc(data.vx_count, training, test, sr, aucn)
        aucSR2 += metrics.auc_semi_total(data.vx_count, training, test, sr, aucn, absent_edges=all_edges)
        print(' ({:.5f}) MESR'.format(eps), end='')
        mesr, eps = merw.merw_simrank(graph, matrix=matrix, alpha=a, iterations=simrank_iter)
        print(' ({:.5f})'.format(eps))
        aucMESR += metrics.auc(data.vx_count, training, test, mesr, aucn)
        aucMESR2 += metrics.auc_semi_total(data.vx_count, training, test, mesr, aucn, absent_edges=all_edges)
    print(" {:4} & {:.3f} & {:.3f} \\\\ \\hline ".format(name, aucSR/data.set_count, aucMESR/data.set_count), end='')
    print(" {:4} & {:.3f} & {:.3f} \\\\ \\hline".format(name, aucSR2/data.set_count, aucMESR2/data.set_count))


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


def __test_auc(data_set, category):
    print('Testing AUC metric...')
    data = dataset.DataSet('../datasets/', category, data_set)
    print("\t{:4} & {:4} & {:4} & {:4}".format(data_set, data.vx_count, data.train_size, data.test_size),
          end='')
    matrix = data.get_training_set(mode='adjacency_matrix_csc', ds_index=1)
    training = set(data.get_training_set())  # metrics.get_edges_set(data.get_training_set())
    test = set(data.get_test_edges())
    Pgrw, sd = merw.compute_grw(matrix)
    p_dist = merw.P_distance(Pgrw, alpha=.5)
    all_edges = metrics.prepare_all_edges_list(data.vx_count, training, test)
    for aucn in [500, 1000, 2000, 3000]:
        print('\n AUC {:4} :'.format(aucn), end='')
        for _ in range(4):
            print(' {:.3f} '.format(metrics.auc_semi_total(data.vx_count, training, test,
                                                           p_dist, aucn, absent_edges=all_edges)), end='', flush=True)
    print('\n [{:.3f}]'.format(metrics.auc_total_fast(data.vx_count, training, test, p_dist)))


if __name__ == '__main__':  # Odrobina testów
    #__experiment_plain()
    #__experiment_01('basic-test-bis', aucn=5000)
    #for i in range(1, 10):
        #test_dataset_symmetry('basic-test', set_no=i)
        #__experiment_01('basic-test', skipSimRank=True, set_no=i, aucn=5000)
    #__test_mat_merw()
    #__simrank_test(DATASETS[2], set_no=1, simrank_iter=[2, 3, 5, 10])
    #__simrank_test(DATASETS[7], set_no=2, simrank_iter=[2, 3, 5, 10])
    #exit(0)

    #for cat, ds in DATASETS[:2]:
    #    __test_auc(ds, cat)
    #exit(0)

    for data_set in DATASETS:
        __simrank_test(data_set)

    #    __experiment_02(data_set, skipSimRank=False, simrank_iter=3, aucn=2000)
    print('\n==================')
    for category, name in DATASETS:
        data = dataset.DataSet('../datasets/', category, name)
        print("{:4} #& {:4} & {:4} & {:4} &  \\\\ \\hline".format(name, data.vx_count,
                                                                  data.train_size, data.test_size))
    exit(0)
    for cat, ds in DATASETS[:2]:
        __test_auc(ds, cat)



