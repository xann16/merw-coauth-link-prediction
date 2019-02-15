import numpy as num
import scipy.linalg as alg


def to_adiacency_row(neighbours, n):
    row = num.zeros(n)
    row[neighbours]=1
    return row


def graph_to_matrix(graph):
    return num.array([to_adiacency_row(vertex, len(graph)) for vertex in graph])


def compute_merw(A: num.array):
    w,v = alg.eig(A)
    maxeigi = 0
    for i in range(1, len(w)):
        if w[maxeigi] < w[i]:
            maxeigi = i
    evalue = w[maxeigi]
    evector = v[:, maxeigi]
    n, = A.shape
    P = num.zeros(A.shape)
    for row in range(n):
        denom = evalue * evector[row]
        for col in range(n):
            P[row][col] = A[row][col] * evector[col] / denom
    return P, evector, evalue, [evector[i]*evector[i] for i in range(n)]


def compute_merw_simrank(graph, alpha, precision=1e-5, maxiter=100):
    n = len(graph)
    R = num.identity(n)
    P, v, val, = compute_merw(graph_to_matrix(graph))
    denom = [[v[x]*v[y] for x in range(n)] for y in range(n)]
    alpha = alpha / val / val
    for iteration in range(maxiter):
        S = num.zeros((n, n))
        for y in range(n):
            for x in range(n):
                for a in graph[x]:
                    for b in graph[y]:
                        S[x][y] += R[a][b] / denom[a][b]
                S[x][y] *= alpha * denom[x][y]
        eps = alg.norm(R - S)
        if eps < precision:
            return S, eps
        R = S
    return R, eps


def compute_basic_simrank(graph, alpha, precision=1e-5, maxiter=100):
    n = len(graph)
    R = num.identity(n)
    for iteration in range(maxiter):
        S = num.zeros((n, n))
        for y in range(n):
            for x in range(n):
                for a in graph[x]:
                    for b in graph[y]:
                        S[x][y] += R[a][b]
                    S[x][y] /= len(graph[y])
                S[x][y] *= alpha / len(graph[x])
        eps = alg.norm(R - S)
        if eps < precision:
            return S, eps
        R = S
    return R, eps