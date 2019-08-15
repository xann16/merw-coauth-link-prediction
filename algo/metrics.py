import random


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
    for _ in range(samples):
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


def auc_total0(vx_count, train_edges, test_edges, scores):
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


def first_not_greater(alist, value):
    a = 0
    b = len(alist)
    while a < b:
        c = (a + b) // 2  # c <= b
        if alist[c] > value:
            a = c + 1
        else:
            b = c
    return b


def first_less(alist, value):
    a = 0
    b = len(alist)
    while a < b:
        c = (a + b) // 2  # a <= c <= b
        if alist[c] < value:
            b = c
        else:
            a = c + 1
    return a


def prepare_all_edges_list(vx_count, train_edges, test_edges):
    alledges = []
    for i in range(vx_count):
        for j in range(i):
            if (j, i) not in train_edges and (j, i) not in test_edges:
                alledges.append((j, i))
    return alledges


# To w praktyce nie działa dość szybko by używać.
def auc_total_fast(vx_count, train_edges, test_edges, scores):
    print('+', end='', flush=True)
    absent_scores = []
    for i in range(1, vx_count):
        for j in range(i):
            if (j, i) not in train_edges and (j, i) not in test_edges:
                absent_scores.append(scores[j, i])
    print('+', end='', flush=True)
    absent_scores.sort(reverse=True)
    score = 0
    samples = 0
    series_length = len(absent_scores)
    for i, j in test_edges:
        train_score = scores[i, j]
        print('\b\b??', end='', flush=True)
        i0 = first_not_greater(absent_scores, train_score)
        i1 = first_less(absent_scores, train_score)
        print('\b\b!!', end='', flush=True)
        samples += series_length
        score += series_length - .5*(i0 + i1)
    return score / samples


# Znacznie szybsza metoda obliczania przybliżonego AUC
def auc_semi_total(vx_count, train_edges, test_edges, scores, count=1000, absent_edges=None):
    #print('++', end='')
    test_scores = [scores[i, j] for (i, j) in test_edges]
    test_scores.sort(reverse=True)
    score = 0
    samples = 0
    if absent_edges is None:
        absent_edges = prepare_all_edges_list(vx_count, train_edges, test_edges)
    for ct in range(count):
        i, j = random.choice(absent_edges)
        notrain_score = scores[i, j]
        #print('\b\b{:2}'.format(ct), end='', flush=True)
        i0 = first_not_greater(test_scores, notrain_score)
        i1 = first_less(test_scores, notrain_score)
        samples += len(test_edges)
        score += .5*(i0 + i1)
    #print('\b\b', end='')
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


def get_absent_edge_list(vx_count, train_edges):
    alledges = []
    for i in range(vx_count):
        for j in range(i):
            if (j, i) not in train_edges:
                alledges.append((j, i))
    return alledges


def get_top_predictions2(vx_count, train_edges, scores, count, absent_edges=None):
    curr_min_val = -1
    curr_min_pos = 0
    res = [(0, 0)] * count
    scs = [-1] * count
    if absent_edges is None:
        absent_edges = get_absent_edge_list(vx_count, train_edges)
    for i, j in absent_edges:
        if scores[i, j] > curr_min_val:
            res[curr_min_pos] = (i, j)
            scs[curr_min_pos] = scores[i, j]
            curr_min_pos, curr_min_val = find_min(scs)
    return res


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
def precision(vx_count, train_edges, test_edges, scores, L=30, absent_edges=None):
    if absent_edges is None:
        predictions = get_top_predictions(vx_count, train_edges, scores, L)
    else:
        predictions = get_top_predictions2(vx_count, train_edges, scores, L, absent_edges=absent_edges)
    score = 0
    for edge in predictions[:L]:
        if edge in test_edges:
            score += 1
    return score / L
