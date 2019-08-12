import numpy as num
import scipy.sparse.linalg as alg
import scipy.linalg as algnorm
import scipy.sparse as smat
import random
# Operacje grafowe - może wydzielić ?


def to_adiacency_row(neighbours, n):
    row = num.zeros(n)
    row[neighbours] = 1
    return row


def graph_to_matrix(graph):  # Tworzy macierz rzadką opisująca graf
    n = len(graph)
    rows = []
    cols = []
    for i in range(n):
        rows.extend([i]*len(graph[i]))
        cols.extend(graph[i])
    data = [1.0 for v in rows]
    matrix = smat.csr_matrix((data, (rows, cols)), (n, n), 'd')
    # for i in range(n):
    #    matrix[i, graph[i]] = 1
    return matrix
    # return num.array([to_adiacency_row(vertex, len(graph)) for vertex in graph])


def matrix_to_graph(A):
    rows, cols = A.nonzero()
    n = A.get_shape()[0]
    graph = [[]] * n
    vert = [0] * n
    for (row, col) in zip(rows, cols):
        if graph[row].count(col) == 0:
            if len(graph[row]) == 0:
                graph[row] = [col]
            else:
                graph[row].append(col)
        if graph[col].count(row) == 0:
            if len(graph[col]) == 0:
                graph[col] = [row]
            else:
                graph[col].append(row)
    return graph

# Obliczanie MERW i SimRanków


# Oblicza macierz przejścia MERW, wektor własny, wartość własną i rozkład stacjonarny
def compute_merw(A):  # Archaiczne liczenie MERW
    n = A.get_shape()[0]
    w, v = alg.eigsh(A, 1, )  # Macierz jest symetryczna
    evalue = w[0]
    evector = v[:, 0]
    evector = evector / algnorm.norm(evector)
    P = smat.lil_matrix((n, n))
    for row in range(n):
        denom = evalue * evector[row]
        for col in range(n):
            if A[row, col] != 0:
                P[row, col] = A[row, col] * evector[col] / denom
    return P, evector, evalue, [evector[i]*evector[i] for i in range(n)]


def power_method(A, precision=1e-11):
    n = A.get_shape()[0]
    v0 = num.array([random.random()+.1 for i in range(n)])
    eps = 1
    iter = 0
    while iter < 100 and eps > precision or iter < 20:
        v1 = v0*A
        iter += 1
        eval = 0
        eps = 0
        for i in range(n):
            if v0[i] == 0:
                continue
            div = v1[i]/v0[i]
            eval = max(eval, div)
            eps += eval - div
        v0 = v1/eval
    return v0/algnorm.norm(v0), eval, iter


def scipy_method(A):
    w, v = alg.eigsh(A, k=1, which='LA')  # Macierz jest symetryczna
    evalue = w[0]
    evector = v[:, 0]
    if evector[0] < 0:
        evector *= -1
    return evector / algnorm.norm(evector), evalue, 1


def _inv(x, y):
    if x != 0:
        return 1/(x*y)
    else:
        return 0.0


# Oblicza macierz przejścia MERW, wektor własny, wartość własną i rozkład stacjonarny
def compute_merw_matrix(A, method=power_method):
    n = A.get_shape()[0]
    evector, evalue, iter = method(A)
    #print('({} itr.)'.format(iter), end='')
    mat1 = smat.diags([evector], [0], shape=(n, n), format='csc')
    mat2 = smat.diags([[_inv(v, evalue) for v in evector]],  # Coś jakby odwrotność macierzy diagonalnej
                      [0], shape=(n, n), format='csc')
    return mat2*A*mat1, evector, evalue, [v*v for v in evector]


# Wyznacza rozkład prawdopodobieństwa i rozkład stacjonarny dla klasycznego błądzenia losowego
def compute_grw(A):
    n = A.get_shape()[0]
    degrees = smat.diags(A.sum(axis=0), [0], shape=(n, n), format='csc').power(-1)
    P = degrees * A
    vals, stationary = alg.eigs(P.transpose(), k=1, sigma=0.9999999)
    inorm = 1/num.sum(stationary[:, 0]).real
    return P, [x.real * inorm for x in stationary[:, 0]]


# Wyznacza miarę MERW-SimRank dla danego grafu.
# Wartości charakteryzujące MERW można podać jako parametry.
def merw_simrank(graph, eigenvector=None, matrix=None, alpha=0.5, iterations=100):
    n = len(graph)
    if eigenvector is None:
        if matrix is None:
            matrix = graph_to_matrix(graph)
        _, v, val, _ = compute_merw_matrix(matrix)
    else:
        v, val = eigenvector
    R = num.identity(n)
    S = num.zeros((n, n))
    denom = [[v[x] * v[y] for x in range(n)] for y in range(n)]
    alpha = alpha / val / val
    for _ in range(iterations):
        for y in range(n):
            S[y, y] = 1.0
            for x in range(y):
                S[x, y] = 0.0
                if denom[x][y] != 0:   # To mmoże nie zachodzić, jeśli graf nie jest spójny
                    for a in graph[x]:
                        for b in graph[y]:
                            if denom[a][b] != 0:
                                S[x, y] += R[a, b] / denom[a][b]
                            #else:
                            #    print('!', end='', flush=True)
                    S[x, y] *= alpha * denom[x][y]
                #else:
                #    print('!', end='', flush=True)
                S[y, x] = S[x, y]
        t = R
        R = S
        S = t
        print(".", end='', flush=True)
    return R, algnorm.norm(R - S)


# Generyczna (mało wydajna) metoda obliczająca miarę SimRank dla dowolnej
# probabilistycznej macierzy przejścia, wykonując zadaną
# liczbę iteracji wzoru rekurencyjnego.
def general_simrank(graph, transm, alpha=0.5, iterations=8):
    n = len(graph)
    R = num.identity(n)
    S = num.zeros((n, n))
    for _ in range(iterations):
        print(".", end='', flush=True)
        for y in range(n):
            S[y, y] = 1.0
            if len(graph[y]) > 0:
                for x in range(y):  # x < y
                    val = 0.0
                    for b in graph[y]:
                        for a in graph[x]:
                            val += R[a, b] * transm[x, a] * transm[y, b]
                    S[y, x] = S[x, y] = val * alpha
        t = R
        R = S
        S = t
    return R, algnorm.norm(R - S)


# Wyznacza klasyczną miarę podobieństwa wierzchołków SimRank, wykonując zadaną
# liczbę iteracji wzoru rekurencyjnego.
def basic_simrank(graph, alpha, iterations=20):
    n = len(graph)
    R = num.identity(n)
    S = num.zeros((n, n))
    for _ in range(iterations):
        for y in range(n):
            S[y, y] = 1.0
            if len(graph[y])>0:
                for x in range(y):
                    S[x, y] = 0.0
                    if len(graph[x]) > 0:
                        for a in graph[x]:
                            for b in graph[y]:
                                S[x, y] += R[a, b]
                        S[x, y] *= alpha / (len(graph[x])*len(graph[y]))
                    S[y, x] = S[x, y]
        t = R
        R = S
        S = t
        print(".", end='', flush=True)
    return R, algnorm.norm(R - S)


def merw_simrank_ofmatrix(matrix, alpha, precision=1e-5, maxiter=20, method=power_method):
    graph = matrix_to_graph(matrix)
    n = len(graph)
    P, v, val, sdist = compute_merw_matrix(matrix, method=method)
    R = num.identity(n)
    S = num.zeros((n, n))
    denom = [[v[x]*v[y] for x in range(n)] for y in range(n)]
    alpha = alpha / val / val
    for iteration in range(maxiter):
        #S.fill(0)  # S = num.zeros((n, n))
        for y in range(n):
            S[y, y] = 1.0
            for x in range(y):
                S[x, y] = 0.0
                if denom[x][y] != 0:   # To może nie zachodzić, jeśli graf nie jest spójny
                    for a in graph[x]:
                        for b in graph[y]:
                            S[x, y] += R[a, b] / denom[a][b]
                    S[x, y] *= alpha * denom[x][y]
                S[y, x] = S[x, y]
        t = R
        R = S
        S = t
    del t, P, v, val, sdist
    return R, algnorm.norm(R - S)


def P_distance(P, alpha=0.8):
    D = smat.identity(P.get_shape()[0], format='csc')
    D -= P * alpha
    return alg.inv(D)


def exp_P_distance(P, alpha=0.8):
    return alg.expm(P * alpha)

