def load_basic_edge_list(src_path):
    edges = []
    verts = set()
    with open(src_path, 'r', encoding='utf-8') as file:
        for line in file:
            if len(line) > 0 and line[0] != '#':
                tokens = line.split("\t", 2)
                vx1 = int(tokens[0].strip())
                vx2 = int(tokens[1].strip())
                edges.append((vx1, vx2))
                verts.add(vx1)
                verts.add(vx2)

    v2v = {}
    vx_index = 0
    for vx in verts:
        v2v[vx] = vx_index
        vx_index += 1

    edge_list = []
    for i, j in edges:
        v1, v2 = v2v[i], v2v[j]
        edge_list.append((v1, v2))
        edge_list.append((v2, v1))

    return len(verts), edge_list


FORMAT_IMPORT_FUNCS = {"edge-list-basic": load_basic_edge_list}
