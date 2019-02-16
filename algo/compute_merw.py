import numpy as num
import scipy.linalg as alg


def to_adiacency_row(neighbours, n):
    row = num.zeros(n)
    row[neighbours]=1
    return row


def graph_to_matrix(graph):
    return num.array([to_adiacency_row(vertex, len(graph)) for vertex in graph])


def dfs(graph, v, visited):
    for w in graph[v]:
        if visited[w] == 0:
            visited[w] = visited[v]
            dfs(graph, w, visited)


def extract_connected_component(graph, vertex):
    n = len(graph)
    member = [0 for v in graph]
    member[vertex] = 1
    dfs(graph, vertex, member)
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
        dfs(graph, vertex, member)
        while vertex < n and member[vertex] > 0:
            vertex += 1
    components = []
    for comp in range(1, comp_id):
        number = [0 for v in graph]
        offset = 0
        j = 0
        components.append([])
        for i in range(n):
            if member[i] == comp:
                number[i] = i + offset
                components[comp-1].append(graph[i])
            else:
                number[i] = -1
                offset -= 1
        for v in components[comp-1]:
            for i in range(len(v)):
                v[i] = number[v[i]]
    return components


def compute_merw(A: num.array):
    w, v = alg.eig(A)
    maxeigi = 0
    for i in range(1, len(w)):
        if w[maxeigi] < w[i]:
            maxeigi = i
    evalue = w[maxeigi]
    evector = v[:, maxeigi]
    n, m = A.shape
    P = num.zeros(A.shape)
    for row in range(n):
        denom = evalue * evector[row]
        for col in range(n):
            P[row][col] = A[row][col] * evector[col] / denom
    return P, evector, evalue, [evector[i]*evector[i] for i in range(n)]


def compute_merw_simrank(graph, alpha, precision=1e-5, maxiter=100):
    n = len(graph)
    R = num.identity(n)
    P, v, val, sdist = compute_merw(graph_to_matrix(graph))
    denom = [[v[x]*v[y] for x in range(n)] for y in range(n)]
    alpha = alpha / val / val
    for iteration in range(maxiter):
        S = num.zeros((n, n))
        for y in range(n):
            for x in range(n):
                if x == y:
                    S[x][y] = 1.0
                elif denom[x][y] != 0:   # To mmoże nie zachodzić, jeśli graf nie jest spójny
                    for a in graph[x]:
                        for b in graph[y]:
                            S[x][y] += R[a][b] / denom[a][b]
                    S[x][y] *= alpha * denom[x][y]
                else:
                    S[x][y] = 0.0
        eps = alg.norm(R - S)
        if eps < precision:
            return R, eps
        R = S
    return R, eps


def compute_basic_simrank(graph, alpha, precision=1e-5, maxiter=100):
    n = len(graph)
    R = num.identity(n)
    for iteration in range(maxiter):
        S = num.zeros((n, n))
        for y in range(n):
            for x in range(n):
                if x == y:
                    S[x][y] = 1.0
                else:
                    for a in graph[x]:
                        for b in graph[y]:
                            S[x][y] += R[a][b]
                    S[x][y] *= alpha / len(graph[x]) / len(graph[y])
        eps = alg.norm(R - S)
        if eps < precision:
            return R, eps
        R = S
    return R, eps

