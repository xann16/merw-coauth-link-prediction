import os
import json
import shutil
from os import path
from datetime import datetime
import calendar


ARXIV_DT_ISO_FORMAT = "%Y-%m-%dT%H:%M:%SZ"


def get_yearly_file_name(category, year):
    return '{}_articles_raw_{}.json'.format(category, year)


def get_years_range(t_from, t_to):
    beg_year = t_from.year
    end_year = t_to.year
    if (t_to.month == 1 and t_to.day == 1 and t_to.hour == 0 and
            t_to.minute == 0 and t_to.second == 0):
        end_year -= 1
    return beg_year, end_year + 1


def create_dir(base_path, dir_path, overwrite=False):
    is_new = False
    full_path = path.join(base_path, dir_path)
    if path.exists(full_path):
        if overwrite:
            shutil.rmtree(full_path)
            is_new = True
    else:
        is_new = True

    if is_new:
        os.mkdir(full_path)


def parse_period(period):
    return parse_datetime(period["from"]), parse_datetime(period["to"])


def parse_datetime(string):
    return datetime.strptime(string, ARXIV_DT_ISO_FORMAT)


def now_as_string():
    return datetime.now().strftime(ARXIV_DT_ISO_FORMAT)


def str_to_utc_ts(string):
    dt = parse_datetime(string)
    return calendar.timegm(dt.utctimetuple())


def append_clique_edges(edges, vertices, include_ts=False, ts=None):
    n = len(vertices)
    for i in range(0, n):
        for j in range(i + 1, n):
            if include_ts:
                edges.append((ts, (vertices[i], vertices[j])))
                edges.append((ts, (vertices[j], vertices[i])))
            else:
                edges.append((vertices[i], vertices[j]))
                edges.append((vertices[j], vertices[i]))


def write_edges_to_file(edges, filepath):
    with open(filepath, 'w', encoding='utf-8') as file:
        for v1, v2 in edges:
            file.write('{}\t{}\n'.format(v1, v2))


def write_to_json(data, filepath):
    with open(filepath, 'w', encoding='utf-8') as file:
        json.dump(data, file, indent=2)


def load_from_json(filepath):
    with open(filepath, 'r', encoding='utf-8') as file:
        return json.load(file)


def triples_to_rear_pairs(triples):
    pairs = []
    for x, y, z in triples:
        pairs.append((y, z))
    return pairs


def find_utc_edges_split_index(utc_edges, split_ts):
    for i in range(len(utc_edges)):
        if utc_edges[i][0] > split_ts:
            return i
    return len(utc_edges)


def process_author_name(name, disable=False):
    if disable:
        return name
    res = ''
    tokens = name.split(' ')
    for token in tokens[:-1]:
        res += token[0]
        res += '.'
    res += tokens[-1]
    return res
