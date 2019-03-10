def get_edges_set(edges):
    res = set()
    for i, j in edges:
        if i < j:
            res.add((i, j))
        else:
            res.add((j, i))
    return res
