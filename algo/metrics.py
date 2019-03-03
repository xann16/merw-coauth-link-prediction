import random


def get_edges_set(edges):
    res = set()
    for i, j in edges:
        if i < j:
            res.add((i, j))
        else:
            res.add((j, i))
    return res


def pick_rnd_missing(test_edges):
    return random.sample(test_edges, 1)[0]


def pick_rnd_notrain(vx_count, train_edges, test_edges):
    while True:
        i = min(int(random.uniform(0, vx_count)), vx_count - 1)
        j = min(int(random.uniform(0, vx_count)), vx_count - 1)
        if i != j and not is_in_edges_2(train_edges, test_edges, i, j):
            if i < j:
                return (i, j)
            return (j, i)


def get_score(scores, coords):
    return scores[coords[0], coords[1]]


def auc_prob(vx_count, train_edges, test_edges, scores, samples):
    score = 0
    for _ in range(0, samples):
        missing_score = get_score(scores, pick_rnd_missing(test_edges))
        notrain_score = get_score(scores, pick_rnd_notrain(vx_count,
                                                           train_edges,
                                                           test_edges))
        if missing_score > notrain_score:
            score += 1
        elif missing_score == notrain_score:
            score += 0.5
    return score / samples


def is_in_edges(edges, i, j):
    if i < j:
        return (i, j) in edges
    return (j, i) in edges


def is_in_edges_2(edges1, edges2, i, j):
    if i < j:
        return (i, j) in edges1 or (i, j) in edges2
    return (j, i) in edges1 or (j, i) in edges2


def auc_total(vx_count, train_edges, test_edges, scores):
    score = 0
    samples = 0

    for i in range(0, vx_count):
        for j in range(i + 1, vx_count):
            if is_in_edges_2(train_edges, test_edges, i, j):
                continue
            notrain_score = scores[i, j]
            for mi, mj in test_edges:
                missing_score = scores[mi, mj]
                if missing_score > notrain_score:
                    score += 1
                elif missing_score == notrain_score:
                    score += 0.5
                samples += 1
    return score / samples


# Oblicza metrykę AUC
# -> vx_count - ilość wierzchołków badanego grafu
# -> train_edges - zbiór krawędzi zbioru treningowego
# -> test_edges - zbiór krawędzi zbioru testowego
# -> scores - macierz zawierająca oceny nadane krawędziom spoza zbioru tren.
# -> samples - ilość próbek (metryka ma charakter probilistyczny), lub dla
#     wartości niedodatniej (dmyślnie) bierze pod uwagę wszystkie możliwe pary
def auc(vx_count, train_edges, test_edges, scores, samples=0):
    if samples < 1:
        return auc_total(vx_count, train_edges, test_edges, scores)
    else:
        return auc_prob(vx_count, train_edges, test_edges, scores, samples)


def find_min(l):
    min_val = l[0]
    min_pos = 0
    for i in range(1, len(l)):
        if l[i] < min_val:
            min_pos = i
            min_val = l[i]
    return min_pos, min_val


def get_top_predictions(vx_count, train_edges, scores, count):
    curr_min_val = -1
    curr_min_pos = 0
    res = [(0, 0)] * count
    scs = [0] * count

    for i in range(0, vx_count):
        for j in range(i + 1, vx_count):
            if (i, j) in train_edges:
                continue
            if scores[i, j] > curr_min_val:
                res[curr_min_pos] = (i, j)
                scs[curr_min_pos] = scores[i, j]
                curr_min_pos, curr_min_val = find_min(scs)

    return res


# Oblicza metrykę Precision (parametry jak dla auc(), oprócz ostatniego)
# -> L - ilość najlepiej ocenionych krawędzi branych pod uwagę
def precision(vx_count, train_edges, test_edges, scores, L=30):
    predictions = get_top_predictions(vx_count, train_edges, scores, L)
    score = 0
    for edge in predictions[:L]:
        if edge in test_edges:
            score += 1
    return score / L
