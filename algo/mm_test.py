from dataset import DataSet
#import metrics
#import numpy as np


def test_small_basic():
    ds = DataSet('../datasets/', 'test', 'small-basic')
    print('DS: {}; iterations: {}'.format(ds.name, ds.set_count))
    for i in range(1, ds.set_count + 1):
        print("ITER #{}".format(i))
        trn, tst = ds.get_dataset(i)
        print('\tTRAIN: {}'.format(trn))
        print('\tTEST:  {}'.format(tst))

    print()


def test_small_cross():
    ds = DataSet('../datasets/', 'test', 'small-cross')
    print('DS: {}; iterations: {}'.format(ds.name, ds.set_count))
    for i in range(1, ds.set_count + 1):
        print("ITER #{}".format(i))
        trn, tst = ds.get_dataset(i)
        print('\tTRAIN: {}'.format(trn))
        print('\tTEST:  {}'.format(tst))

    print()


if __name__ == '__main__':
    print('TEST #1 - SMALL BASIC:')
    test_small_basic()

    print('TEST #2 - SMALL CROSS:')
    test_small_cross()
