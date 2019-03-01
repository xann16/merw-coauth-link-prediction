import dataset
from dataset import DataSet
import compute_merw as rw
import metrics as mtr
import kernel_methods as kern
import numpy as np
import scipy.sparse.linalg as sla
from scipy.sparse import csc_matrix, csr_matrix


def get_small_scores():
    return np.array([[0.0, 0.4, 0.5, 0.6, 0.1],
                     [0.4, 0.0, 0.3, 0.7, 0.2],
                     [0.5, 0.3, 0.0, 0.5, 0.3],
                     [0.6, 0.7, 0.5, 0.0, 0.6],
                     [0.1, 0.2, 0.3, 0.6, 0.0]])


def get_small_adjmx():
    return np.array([[0, 0, 1, 0, 1],
                     [0, 0, 1, 1, 1],
                     [1, 1, 0, 0, 1],
                     [0, 1, 0, 0, 1],
                     [1, 1, 1, 1, 0]])


def get_art_adjmx():
    return np.array([[0, 1, 0, 0, 0],
                     [1, 0, 1, 1, 0],
                     [0, 1, 0, 1, 0],
                     [0, 1, 1, 0, 1],
                     [0, 0, 0, 1, 0]])


def test_small_basic():
    ds = DataSet('../datasets/', 'test', 'small-basic')
    print('DS: {}; iterations: {}'.format(ds.name, ds.set_count))
    for i in range(1, ds.set_count + 1):
        print("ITER #{}".format(i))

        trn, tst = ds.get_dataset(i)
        print('\tTRAIN: {}'.format(trn))
        print('\tTEST:  {}'.format(tst))

        trns, tsts = mtr.get_edges_set(trn), mtr.get_edges_set(tst)
        scores = get_small_scores()

        auc_res_tot = mtr.auc(ds.vx_count, trns, tsts, scores)
        auc_res_010 = mtr.auc(ds.vx_count, trns, tsts, scores, 10)
        auc_res_100 = mtr.auc(ds.vx_count, trns, tsts, scores, 100)
        auc_res_01k = mtr.auc(ds.vx_count, trns, tsts, scores, 1000)
#        auc_res_10k = mtr.auc(ds.vx_count, trns, tsts, scores, 10000)
#        auc_res_1ck = mtr.auc(ds.vx_count, trns, tsts, scores, 100000)
#        auc_res_01m = mtr.auc(ds.vx_count, trns, tsts, scores, 1000000)
        prc_res_002 = mtr.precision(ds.vx_count, trns, tsts, scores, 2)

        print('\tMETRICS:')
        print('\t\t-> AUC___TOTAL: {:.04}'.format(auc_res_tot))  # exp: 0.67
        print('\t\t-> AUC______10: {:.04}'.format(auc_res_010))
        print('\t\t-> AUC_____100: {:.04}'.format(auc_res_100))
        print('\t\t-> AUC____1000: {:.04}'.format(auc_res_01k))
#        print('\t\t-> AUC___10000: {:.04}'.format(auc_res_10k))
#        print('\t\t-> AUC__100000: {:.04}'.format(auc_res_1ck))
#        print('\t\t-> AUC_1000000: {:.04}'.format(auc_res_01m))
        print('\t\t-> PRECISON__2: {:.04}'.format(prc_res_002))  # exp: 0.50

    print()


def test_small_cross():
    ds = DataSet('../datasets/', 'test', 'small-cross')
    print('DS: {}; iterations: {}'.format(ds.name, ds.set_count))
    for i in range(1, ds.set_count + 1):
        print("ITER #{}".format(i))
        trn, tst = ds.get_dataset(i)
        print('\tTRAIN: {}'.format(trn))
        print('\tTEST:  {}'.format(tst))

        trns, tsts = mtr.get_edges_set(trn), mtr.get_edges_set(tst)
        scores = get_small_scores()

        auc_res_tot = mtr.auc(ds.vx_count, trns, tsts, scores)
        auc_res_010 = mtr.auc(ds.vx_count, trns, tsts, scores, 10)
        auc_res_100 = mtr.auc(ds.vx_count, trns, tsts, scores, 100)
        auc_res_01k = mtr.auc(ds.vx_count, trns, tsts, scores, 1000)
#        auc_res_10k = mtr.auc(ds.vx_count, trns, tsts, scores, 10000)
#        auc_res_1ck = mtr.auc(ds.vx_count, trns, tsts, scores, 100000)
#        auc_res_01m = mtr.auc(ds.vx_count, trns, tsts, scores, 1000000)
        prc_res_002 = mtr.precision(ds.vx_count, trns, tsts, scores, 2)

        print('\tMETRICS:')
        print('\t\t-> AUC___TOT: {:.04}'.format(auc_res_tot))  # expected: 0.67
        print('\t\t-> AUC____10: {:.04}'.format(auc_res_010))
        print('\t\t-> AUC___100: {:.04}'.format(auc_res_100))
        print('\t\t-> AUC____1K: {:.04}'.format(auc_res_01k))
#        print('\t\t-> AUC___10K: {:.04}'.format(auc_res_10k))
#        print('\t\t-> AUC__100K: {:.04}'.format(auc_res_1ck))
#        print('\t\t-> AUC____1M: {:.04}'.format(auc_res_01m))
        print('\t\t-> PREC____2: {:.04}'.format(prc_res_002))  # expected: 0.50

    print()


def print_sparse_as_dense(S):
    for i in range(S.get_shape()[0]):
        print('[', end=' ')
        for j in range(S.get_shape()[1]):
            print('{:7.4f}'.format(S[i, j]), end=' ')
        print(' ]')


def walks_survey(A):
    print('A (adjacency matrix):')
    print(A)

    print()
    print('-------------------------------------')
    print('GRW:')
    P_grw, pi_grw = rw.compute_grw(csc_matrix(A))

    print()
    print('P (GRW transition matrix):')
    print_sparse_as_dense(P_grw)

    print()
    print('pi (GRW stationary distribution):')
    print(pi_grw)

    L_grw = kern.general_laplacian(P_grw, pi_grw)
    print()
    print('L (GRW general laplacian):')
    print_sparse_as_dense(L_grw)

    LL = kern.laplacian(csr_matrix(A, (A.shape[0], A.shape[1]), 'd'))
    print()
    print('LL (GRW laplacian):')
    print_sparse_as_dense(LL)

    LL_sym = kern.symmetric_normalized_laplacian(
        csr_matrix(A, (A.shape[0], A.shape[1]), 'd'))
    print()
    print('L (GRW symmetric normalized laplacian):')
    print_sparse_as_dense(LL_sym)

    print()
    print('-------------------------------------')
    print('MERW:')
    P_merw, v_merw, lambda_merw, pi_merw = \
        rw.compute_merw(csr_matrix(A, (A.shape[0], A.shape[1]), 'd'))
    v_merw *= -1

    l, v = sla.eigsh(csr_matrix(A, (A.shape[0], A.shape[1]), 'd'), 1,
                     which='LA')
    lambda_max = l[0]
    v_max = v[:, 0]

    print()
    print('P (MERW transition matrix):')
    print_sparse_as_dense(P_merw)

    print()
    print('pi (MERW stationary distribution):')
    print(pi_merw)

    print()
    print('lambda (max eigenvalue):')
    print(lambda_merw)
    print(lambda_max)

    print()
    print('v (max eigenvector):')
    print(v_merw)
    print(v_max)

    W = kern.compute_eigen_weighted_graph(
        csr_matrix(A, (A.shape[0], A.shape[1]), 'd'), lambda_merw, v_merw)

    print()
    print('W (eigen-weighted graph):')
    print_sparse_as_dense(W)

    L_merw = kern.general_laplacian(P_merw, pi_merw)
    print()
    print('L (MERW general laplacian):')
    print_sparse_as_dense(L_merw)

    L = kern.mecl(
        csr_matrix(A, (A.shape[0], A.shape[1]), 'd'), lambda_merw, v_merw)
    print()
    print('L (maximal entropy combinatorial laplacian):')
    print_sparse_as_dense(L)

    L_sym = kern.mecl(
        csr_matrix(A, (A.shape[0], A.shape[1]), 'd'), lambda_merw, v_merw,
        type='sym')
    print()
    print('L_sym (symmetric normalized maximal entropy laplacian):')
    print_sparse_as_dense(L_sym)

    L_asym = kern.mecl(
        csr_matrix(A, (A.shape[0], A.shape[1]), 'd'), lambda_merw, v_merw,
        type='asym')
    print()
    print('L_rw (asymmetric normalized maximal entropy laplacian):')
    print_sparse_as_dense(L_asym)

    CK = kern.commute_time_kernel(LL, 3)
    print()
    print('CK:')
    print_sparse_as_dense(CK)

    NCK = kern.commute_time_kernel(LL_sym, 3)
    print()
    print('NCK:')
    print_sparse_as_dense(NCK)

    MECK = kern.commute_time_kernel(L, 3)
    print()
    print('MECK:')
    print_sparse_as_dense(MECK)

    NMECK = kern.commute_time_kernel(L_sym, 3)
    print()
    print('NMECK:')
    print_sparse_as_dense(NMECK)

    DK = kern.heat_diffusion_kernel(LL)
    print()
    print('DK:')
    print_sparse_as_dense(DK)

    NDK = kern.heat_diffusion_kernel(LL_sym, 3)
    print()
    print('NDK:')
    print_sparse_as_dense(NDK)

    MEDK = kern.heat_diffusion_kernel(L)
    print()
    print('MEDK:')
    print_sparse_as_dense(MEDK)

    NMEDK = kern.heat_diffusion_kernel(L_sym, 3)
    print()
    print('NMEDK:')
    print_sparse_as_dense(NMEDK)

    RK = kern.regularized_laplacian_kernel(LL)
    print()
    print('RK:')
    print_sparse_as_dense(RK)

    NRK = kern.regularized_laplacian_kernel(LL_sym)
    print()
    print('NRK:')
    print_sparse_as_dense(NRK)

    MERK = kern.regularized_laplacian_kernel(L)
    print()
    print('MERK:')
    print_sparse_as_dense(MERK)

    NMERK = kern.regularized_laplacian_kernel(L_sym)
    print()
    print('NMERK:')
    print_sparse_as_dense(NMERK)

    MENK = kern.neumann_kernel(
        csr_matrix(A, (A.shape[0], A.shape[1]), 'd'), lambda_merw, v_merw)
    print()
    print('MENK:')
    print_sparse_as_dense(MENK)

    NNK = kern.traditional_normalized_neumann_kernel(
        csr_matrix(A, (A.shape[0], A.shape[1]), 'd'))
    print()
    print('NNK:')
    print_sparse_as_dense(NNK)

    NMENK = kern.normalized_neumann_kernel(
        csr_matrix(A, (A.shape[0], A.shape[1]), 'd'), lambda_merw, v_merw)
    print()
    print('NMENK:')
    print_sparse_as_dense(NMENK)


def dk_tests_1k():
    ds = DataSet('../datasets/', 'gr-qc', 'eg1k')
    trn, tst = ds.get_dataset()
    trns, tsts = mtr.get_edges_set(trn), mtr.get_edges_set(tst)
    A = csr_matrix(dataset.edge_list_to_sparse_lil(ds.vx_count, trn),
                   (ds.vx_count, ds.vx_count), 'd')

    ls, vs = sla.eigsh(A, 1, which='LA')
    l_max = ls[0]
    v_max = vs[:, 0]

    # print("Values of AUC (1000 samples) and precision (K=30) " +
    #       "for heat diffusion kernel variants:")

    print("Values of AUC (1000 samples) for heat diffusion kernel variants:")

    auc_sampl = 1000
    # prc_k = 30

    # DK
    DK = kern.heat_diffusion_kernel(kern.laplacian(A))

    auc = mtr.auc(ds.vx_count, trns, tsts, DK, auc_sampl)
    # prc = mtr.precision(ds.vx_count, trns, tsts, DK, prc_k)
    print("   DK - AUC:  {:.4f}".format(auc))
    # print("   DK - PREC: {:.4f}".format(prc))

    # NDK
    NDK = kern.heat_diffusion_kernel(kern.symmetric_normalized_laplacian(A))

    auc = mtr.auc(ds.vx_count, trns, tsts, NDK, auc_sampl)
    # prc = mtr.precision(ds.vx_count, trns, tsts, NDK, prc_k)
    print("  NDK - AUC:  {:.4f}".format(auc))
    # print("  NDK - PREC: {:.4f}".format(prc))

    # MEDK
    MEDK = kern.heat_diffusion_kernel(kern.mecl(A, l_max, v_max))

    auc = mtr.auc(ds.vx_count, trns, tsts, MEDK, auc_sampl)
    # prc = mtr.precision(ds.vx_count, trns, tsts, MEDK, prc_k)
    print(" MEDK - AUC: {:.4f}".format(auc))
    # print(" MEDK - PREC: {:.4f}".format(prc))

    # NMEDK
    NMEDK = kern.heat_diffusion_kernel(kern.mecl(A, l_max, v_max, type='sym'))

    auc = mtr.auc(ds.vx_count, trns, tsts, NMEDK, auc_sampl)
    # prc = mtr.precision(ds.vx_count, trns, tsts, NMEDK, prc_k)
    print("NMEDK - AUC: {:.4f}".format(auc))
    # print("NMEDK - PREC: {:.4f}".format(prc))


if __name__ == '__main__':
    # print('TEST #1 - SMALL BASIC:')
    # test_small_basic()

    # rint('TEST #2 - SMALL CROSS:')
    # test_small_cross()

    # print('WALKS SURVEY - small graph:\n')
    # walks_survey(get_small_adjmx())

    # print('WALKS SURVEY - article graph:\n')
    # walks_survey(get_art_adjmx())

    dk_tests_1k()
