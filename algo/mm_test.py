from dataset import DataSet
import metrics as mtr
import numpy as np


def get_small_scores():
    return np.array([[0.0, 0.4, 0.5, 0.6, 0.1],
                     [0.4, 0.0, 0.3, 0.7, 0.2],
                     [0.5, 0.3, 0.0, 0.5, 0.3],
                     [0.6, 0.7, 0.5, 0.0, 0.6],
                     [0.1, 0.2, 0.3, 0.6, 0.0]])


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
        auc_res_10k = mtr.auc(ds.vx_count, trns, tsts, scores, 10000)
        auc_res_1ck = mtr.auc(ds.vx_count, trns, tsts, scores, 100000)
        auc_res_01m = mtr.auc(ds.vx_count, trns, tsts, scores, 1000000)
        prc_res_002 = mtr.precision(ds.vx_count, trns, tsts, scores, 2)

        print('\tMETRICS:')
        print('\t\t-> AUC___TOTAL: {:.04}'.format(auc_res_tot))  # exp: 0.67
        print('\t\t-> AUC______10: {:.04}'.format(auc_res_010))
        print('\t\t-> AUC_____100: {:.04}'.format(auc_res_100))
        print('\t\t-> AUC____1000: {:.04}'.format(auc_res_01k))
        print('\t\t-> AUC___10000: {:.04}'.format(auc_res_10k))
        print('\t\t-> AUC__100000: {:.04}'.format(auc_res_1ck))
        print('\t\t-> AUC_1000000: {:.04}'.format(auc_res_01m))
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
#        auc_res_010 = mtr.auc(ds.vx_count, trns, tsts, scores, 10)
#        auc_res_100 = mtr.auc(ds.vx_count, trns, tsts, scores, 100)
#        auc_res_01k = mtr.auc(ds.vx_count, trns, tsts, scores, 1000)
#        auc_res_10k = mtr.auc(ds.vx_count, trns, tsts, scores, 10000)
#        auc_res_1ck = mtr.auc(ds.vx_count, trns, tsts, scores, 100000)
#        auc_res_01m = mtr.auc(ds.vx_count, trns, tsts, scores, 1000000)
        prc_res_002 = mtr.precision(ds.vx_count, trns, tsts, scores, 2)

        print('\tMETRICS:')
        print('\t\t-> AUC___TOT: {:.04}'.format(auc_res_tot))  # expected: 0.67
#        print('\t\t-> AUC____10: {:.04}'.format(auc_res_010))
#        print('\t\t-> AUC___100: {:.04}'.format(auc_res_100))
#        print('\t\t-> AUC____1K: {:.04}'.format(auc_res_01k))
#        print('\t\t-> AUC___10K: {:.04}'.format(auc_res_10k))
#        print('\t\t-> AUC__100K: {:.04}'.format(auc_res_1ck))
#        print('\t\t-> AUC____1M: {:.04}'.format(auc_res_01m))
        print('\t\t-> PREC____2: {:.04}'.format(prc_res_002))  # expected: 0.50

    print()


if __name__ == '__main__':
    print('TEST #1 - SMALL BASIC:')
    test_small_basic()

    #print('TEST #2 - SMALL CROSS:')
    #test_small_cross()
