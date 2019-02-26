import os
import json
import shutil
from os import path
from datetime import datetime

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


def append_clique_edges(edges, vertices):
    n = len(vertices)
    for i in range(0, n):
        for j in range(i + 1, n):
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
