from scipy.sparse.linalg import svds, expm, inv
import scipy.sparse as sp
from scipy.sparse import csr_matrix, csc_matrix
import numpy as np


def compute_eigen_weighted_graph(adj_matrix, lambda_max, v_max):
    res = adj_matrix.copy()
    for i, j in zip(adj_matrix.nonzero()[0], adj_matrix.nonzero()[1]):
        res[i, j] = (v_max[i] * v_max[j]) / lambda_max
    return res


def general_laplacian(P, pi):
    n = P.get_shape()[0]
    return sp.diags([pi], [0], shape=(n, n), format='csr') * \
        (sp.eye(n, format='csr') - P)


def laplacian(A):
    n = A.get_shape()[0]
    return sp.diags([A.sum(axis=0)], [0], shape=(n, n), format='csr') - A


def symmetric_normalized_laplacian(A):
    n = A.get_shape()[0]
    D_invsqrt = sp.diags([np.power(A.sum(axis=0), np.full(n, -0.5))], [0],
                         shape=(n, n), format='csr')
    return sp.eye(n, format='csr') - (D_invsqrt * A * D_invsqrt)


def maximal_entropy_combinatorial_laplacian(A, lambda_max, v_max):
    n = A.get_shape()[0]
    D_v = sp.diags([v_max], [0], shape=(n, n), format='csr')
    D_v_sqr = sp.diags([v_max * v_max], [0], shape=(n, n), format='csr')
    return D_v_sqr - ((D_v * A * D_v) / lambda_max)


def symmetric_normalized_maximal_entropy_laplacian(A, lambda_max, _):
    n = A.get_shape()[0]
    return sp.eye(n, format='csr') - (A / lambda_max)


def asymmetric_normalized_maximal_entropy_laplacian(A, lambda_max, v_max):
    n = A.get_shape()[0]
    D_v = sp.diags([v_max], [0], shape=(n, n), format='csr')
    D_v_inv = sp.diags([np.power(v_max, np.full(n, -1))], [0],
                       shape=(n, n), format='csr')
    return sp.eye(n, format='csr') - ((D_v_inv * A * D_v) / lambda_max)


def mecl(A, lambda_max, v_max, type='std'):
    if type == 'sym':
        return symmetric_normalized_maximal_entropy_laplacian(
            A, lambda_max, v_max)
    elif type == 'asym':
        return asymmetric_normalized_maximal_entropy_laplacian(
            A, lambda_max, v_max)
    else:
        return maximal_entropy_combinatorial_laplacian(
            A, lambda_max, v_max)


def pinv(A, k=6):
    U, Sigma, V = svds(A, k)
    UU = csr_matrix(U, (U.shape[0], U.shape[1]), 'd')
    D_sigma = sp.diags([np.power(Sigma, np.full(k, -1))], [0],
                       shape=(k, k), format='csr')
    VV = csr_matrix(V, (V.shape[0], V.shape[1]), 'd')
    return UU * D_sigma * VV


def commute_time_kernel(L, k=6):
    return pinv(L, k)


def heat_diffusion_kernel(L, alpha=0.5):
    return csr_matrix(expm((-alpha) * csc_matrix(L)))


def regularized_laplacian_kernel(L, alpha=0.5):
    n = L.get_shape()[0]
    return csr_matrix(inv(sp.eye(n, format='csc') + (alpha * csc_matrix(L))))


def neumann_kernel(A, lambda_max, v_max, alpha=0.5):
    n = A.get_shape()[0]
    D_v = sp.diags([v_max], [0], shape=(n, n), format='csc')
    return csr_matrix(inv(sp.eye(n, format='csc') -
                          ((alpha * D_v * csc_matrix(A) * D_v) / lambda_max)))


def normalized_neumann_kernel(A, lambda_max, _, alpha=0.5):
    n = A.get_shape()[0]
    return csr_matrix(inv(sp.eye(n, format='csc') -
                          ((alpha * csc_matrix(A)) / lambda_max)))


def traditional_normalized_neumann_kernel(A, alpha=0.5):
    n = A.get_shape()[0]
    return csr_matrix(inv(sp.eye(n, format='csc') - (alpha * csc_matrix(A))))
