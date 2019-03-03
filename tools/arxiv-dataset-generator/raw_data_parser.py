from os import path
import dsgen_utils as util


def load_articles_from_cache(category, year, cache_path):
    filepath = path.join(cache_path,
                         util.get_yearly_file_name(category, year))
    return util.load_from_json(filepath)


def parse_article_entry(edges, author_ids, curr_auth_id,
                        article, t_from, t_to, include_ts=False):
    authors = article["authors"]
    publ = util.parse_datetime(article["published"])
    if len(authors) < 2 or publ < t_from or publ > t_to:
        return 0
    res, ids = update_author_ids(author_ids, curr_auth_id, authors)
    util.append_clique_edges(edges, ids, include_ts, publ)
    return res


def update_author_ids(author_ids, curr_auth_id, authors):
    ids = []
    new_auth_ids = 0
    for author in authors:
        author = util.process_author_name(author)
        if author not in author_ids:
            author_ids[author] = curr_auth_id
            curr_auth_id += 1
            new_auth_ids += 1
        ids.append(author_ids[author])
    return new_auth_ids, ids


def load_edge_list(category, cache_path, t_from, t_to, include_ts=False):
    edges = []
    author_ids = {}
    current_id = 0

    y_beg, y_end = util.get_years_range(t_from, t_to)
    for year in range(y_beg, y_end):
        articles = load_articles_from_cache(category, year, cache_path)
        for article in articles:
            new_author_count = parse_article_entry(edges,
                                                   author_ids,
                                                   current_id,
                                                   article,
                                                   t_from,
                                                   t_to,
                                                   include_ts)
            current_id += new_author_count

    return current_id, edges
