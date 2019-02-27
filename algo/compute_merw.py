import numpy as num
import scipy.sparse.linalg as alg
import scipy.linalg as algnorm
import scipy.sparse as smat
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
                graph[row]=[col]
            else:
                graph[row].append(col)
        if graph[col].count(row) == 0:
            if len(graph[col]) == 0:
                graph[col] = [row]
            else:
                graph[col].append(row)
    return graph


def __dfs(graph, v, visited):
    for w in graph[v]:
        if visited[w] == 0:
            visited[w] = visited[v]
            __dfs(graph, w, visited)


def extract_connected_component(graph, vertex):  # Być może zbędna funkcja
    n = len(graph)
    member = [0 for v in graph]
    member[vertex] = 1
    __dfs(graph, vertex, member)
    number = [0 for v in graph]
    offset = 0
    j = 0
    component = []
    for i in range(n):
        if member[i]:
            number[i] = i - offset
            component.append(graph[i])
        else:
            number[i] = -1
            offset -= 1
    for v in component:
        for i in range(len(v)):
            v[i] = number[v[i]]
    return component


def get_all_components(graph):
    n = len(graph)
    member = [0 for v in graph]
    vertex = 0
    comp_id = 1
    while vertex < n and member[vertex] == 0:
        member[vertex] = comp_id
        comp_id += 1
        __dfs(graph, vertex, member)
        while vertex < n and member[vertex] > 0:
            vertex += 1
    components = []
    number = [0 for v in graph]
    index = [0 for c in range(1, comp_id)]
    for i in range(n):
        comp = member[i]-1
        if index[comp] == 0:
            components.append([graph[i]])
        else:
            components[comp].append(graph[i])
        number[i] = index[comp]
        index[comp] += 1
    for component in components:
        for v in component:
            for i in range(len(v)):
                v[i] = number[v[i]]
    return components

# Obliczanie MERW i SimRanków


def compute_merw(A, k=10): # Archaiczne liczenie MERW
    n = A.get_shape()[0]
    k = min(k, n-1)
    w, v = alg.eigsh(A, k)  # Macierz jest symetryczna
    maxeigi = 0
    for i in range(1, len(w)):
        if w[maxeigi] < w[i]:
            maxeigi = i
    evalue = w[maxeigi]
    evector = v[:, maxeigi]
    evector = evector / algnorm.norm(evector)
    P = smat.lil_matrix((n, n))
    for row in range(n):
        denom = evalue * evector[row]
        for col in range(n):
            if A[row, col] != 0:
                P[row, col] = A[row, col] * evector[col] / denom
    return P, evector, evalue, [evector[i]*evector[i] for i in range(n)]


def compute_merw_matrix(A, k=5):
    n = A.get_shape()[0]
    k = min(k, n - 1)
    w, v = alg.eigsh(A, k)  # Macierz jest symetryczna
    maxeigi = 0
    for i in range(1, len(w)):
        if w[maxeigi] < w[i]:
            maxeigi = i
    evalue = w[maxeigi]
    evector = v[:, maxeigi]
    evector = evector / algnorm.norm(evector)
    mat1 = smat.diags([evector],[0], shape=(n, n), format='csc')
    mat2 = alg.inv(mat1)
    return mat2*A*mat1*(1/evalue), evector, evalue, [v*v for v in evector]


def compute_grw(A):  # Wyznacza rozkład prawdopodobieństwa i rozkład stacjonarny dla zwykłego błądzenia
    n = A.get_shape()[0]
    degrees = smat.diags(A.sum(axis=0), [0], shape=(n, n), format='csr').power(-1)
    P = degrees * A
    vals, stationary = alg.eigs(P.transpose(), k=1, sigma=0.9999999)
    inorm = 1/num.sum(stationary[:, 0]).real
    return P, [x.real * inorm for x in stationary[:, 0]]


def compute_merw_simrank(graph, alpha, precision=1e-5, maxiter=100):
    n = len(graph)
    R = num.identity(n)
    P, v, val, sdist = compute_merw_matrix(graph_to_matrix(graph))
    denom = [[v[x]*v[y] for x in range(n)] for y in range(n)]
    alpha = alpha / val / val
    for iteration in range(maxiter):
        S = num.zeros((n, n))
        for y in range(n):
            for x in range(n):
                if x == y:
                    S[x, y] = 1.0
                elif denom[x][y] != 0:   # To mmoże nie zachodzić, jeśli graf nie jest spójny
                    for a in graph[x]:
                        for b in graph[y]:
                            S[x, y] += R[a, b] / denom[a][b]
                    S[x, y] *= alpha * denom[x][y]
                else:
                    S[x, y] = 0.0
        eps = algnorm.norm(R - S)
        if eps < precision:
            return R, eps
        R = S
    return R, eps


def compute_basic_simrank(graph, alpha, precision=1e-5, maxiter=100):
    n = len(graph)
    R = num.identity(n)
    S = num.zeros((n, n))
    for iteration in range(maxiter):
        for y in range(n):
            S[y, y] = 1.0
            for x in range(y):
                S[x, y] = 0.0
                for a in graph[x]:
                    for b in graph[y]:
                        S[x, y] += R[a, b]
                S[x, y] *= alpha / (len(graph[x])*len(graph[y]))
                S[y, x] = S[x, y]
        #eps = algnorm.norm(R - S)
        #if eps < precision:
        #    return R, eps
        R = S
    return R, algnorm.norm(R - S)


def compute_merw_simrank_ofmatrix(matrix, alpha, precision=1e-5, maxiter=100):
    graph = matrix_to_graph(matrix)
    n = len(graph)
    P, v, val, sdist = compute_merw_matrix(matrix)
    R = num.identity(n)
    S = num.zeros((n, n))
    denom = [[v[x]*v[y] for x in range(n)] for y in range(n)]
    alpha = alpha / val / val
    for iteration in range(maxiter):
        #S.fill(0)  #S = num.zeros((n, n))
        for y in range(n):
            S[y, y] = 1.0
            for x in range(y):
                #if denom[x][y] != 0:   # To mmoże nie zachodzić, jeśli graf nie jest spójny
                    S[x, y] = 0.0
                    for a in graph[x]:
                        for b in graph[y]:
                            S[x, y] += R[a, b] / denom[a][b]
                    S[x, y] *= alpha * denom[x][y]
                    S[y, x] = S[x, y]
                #else:
                #    S[x, y] = 0.0
        #eps = algnorm.norm(R - S)
        #if eps < precision:
        #    return R, eps
        R = S
    return R, algnorm.norm(R - S)


def compute_P_distance_iterative(P, alpha=0.8, maxiter=100, precision=1e-6):  # Archaiczna i niedokładna
    if alpha <=0 or alpha>1:
        raise ValueError()
    D = powr = P*alpha
    result = smat.identity(P.get_shape()[0], format='csr') + D
    for i in range(maxiter):
        powr *= D
        result = result + powr
        eps = alg.norm(powr)
        if eps < precision:
            return result, eps
    return result, eps


def compute_P_distance(P, alpha=0.8):
    D = smat.identity(P.get_shape()[0], format='csc')
    D -= P * alpha
    return alg.inv(D)
