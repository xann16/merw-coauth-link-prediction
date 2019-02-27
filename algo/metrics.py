import random


def pick_rnd_missing(test_edges):
    return random.sample(test_edges, 1)


def pick_rnd_notrain(vx_count, train_edges):
    while True:
        i = min(int(random.uniform(0, vx_count)), vx_count - 1)
        j = min(int(random.uniform(0, vx_count)), vx_count - 1)
        if i != j and (i, j) not in train_edges:
            return (i, j)


def get_score(scores, coords):
    return scores[coords[0], coords[1]]


def auc_prob(vx_count, train_edges, test_edges, scores, samples):
    score = 0
    for _ in range(0, samples):
        missing_score = get_score(scores, pick_rnd_missing(test_edges))
        notrain_score = get_score(scores, pick_rnd_notrain(vx_count,
                                                           train_edges))
        if missing_score > notrain_score:
            score += 1
        elif missing_score == notrain_score:
            score += 0.5
    return score / samples


def auc_total(vx_count, train_edges, test_edges, scores):
    score = 0
    samples = 0
    for i in range(1, vx_count):
        for j in range(i, vx_count):
            if (i, j) in train_edges:
                continue
            notrain_score = scores[i, j]
            for mi, mj in test_edges:
                missing_score = scores[i, j]
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
    for i in range(0, len(l)):
        if l[i] < min_val:
            min_pos = i
            min_val = l[i]
    return min_pos, min_val


def get_top_predictions(vx_count, train_edges, scores, count):
    curr_min_val = -1
    curr_min_pos = 0
    res = [(0, 0)] * count
    scs = [0] * count

    for i in range(1, vx_count):
        for j in range(i, vx_count):
            print('i:', i, 'j:', j, 'val', scores[i, j])
            if (i, j) in train_edges:
                continue
            if scores[i, j] > curr_min_val:
                res[curr_min_pos] = (i, j)
                scs[curr_min_pos] = scores[i, j]
                curr_min_pos, curr_min_val = find_min(scs)
            print(res)
            print(scs)
    return res


# Oblicza metrykę Precision (parametry jak dla auc(), oprócz ostatniego)
# -> L - ilość najlepiej ocenionych krawędzi branych pod uwagę
def precision(vx_count, train_edges, test_edges, scores, L=30):
    ranked_predictions = get_top_predictions(vx_count, train_edges, scores, L)
    score = 0
    for edge in ranked_predictions[:L]:
        if edge in test_edges:
            score += 1
    return score / L
