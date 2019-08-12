import random
from os import path
import arxiv_api_client as arxiv
import raw_data_parser as parser
import graph_preproc as preproc
import dsgen_utils as util
import import_dataset as dsimp


def get_edges_for_random(category, cache_path, t_from, t_to, t_mid):
    return parser.load_edge_list(category, cache_path, t_from, t_to)


def get_edges_for_chrono(category, cache_path, t_from, t_to, t_mid):
    return parser.load_edge_list(category, cache_path, t_from, t_to,
                                 include_ts=True)


def get_edges_for_chrono_from(category, cache_path, t_from, t_to, t_mid):
    return parser.load_edge_list_not_to_new_nodes(category, cache_path, t_from, t_mid, t_to)


def prepare_random_datesets(edges, settings, vx_count, edge_count, base_path):
    ds_name = settings["name"]
    ds_count = settings["series_count"]
    test_frac = settings["test_perc"] / 100
    test_edges_count = int(edge_count * test_frac)

    for ds_index in range(1, ds_count + 1):
        test_path = path.join(base_path, '{:03}_{}.test.csv'
                                         .format(ds_index, ds_name))
        train_path = path.join(base_path, '{:03}_{}.train.csv'
                                          .format(ds_index, ds_name))
        random.shuffle(edges)
        util.write_edges_to_file(edges[:test_edges_count], test_path)
        util.write_edges_to_file(edges[test_edges_count:], train_path)

    metadata = {"name": ds_name,
                "vertices": vx_count,
                "edges": edge_count,
                "set_count": ds_count,
                "format_type": "basic-edge-list",
                "split_method": settings["split_method"],
                "training_sets_size": edge_count - test_edges_count,
                "test_sets_size": test_edges_count,
                "created": util.now_as_string()}
    meta_path = path.join(base_path, '{}_meta.json'.format(ds_name))
    util.write_to_json(metadata, meta_path)

    print('Data files for "{}" dataset succesfully created '.format(ds_name) +
          '({} vertices, {} edges).'.format(vx_count, edge_count))
    print('For details, see: {}'.format(meta_path))


def prepare_k_cross_random_datesets(edges, settings, vx_count, edge_count,
                                    base_path):
    ds_name = settings["name"]
    k = settings["k_subset_count"]
    k_frac = 1 / k
    k_size = 0

    random.shuffle(edges)
    for i in range(1, k + 1):
        full_path = path.join(base_path, '{:03}_{}.cross.csv'
                                         .format(i, ds_name))
        start = int(((i - 1) * k_frac) * edge_count)
        end = int((i * k_frac) * edge_count)
        k_size = end - start
        util.write_edges_to_file(edges[start:end], full_path)

    metadata = {"name": ds_name,
                "vertices": vx_count,
                "edges": edge_count,
                "set_count": k,
                "format_type": "basic-edge-list",
                "split_method": settings["split_method"],
                "training_sets_size": k_size * (k - 1),
                "test_sets_size": k_size,
                "created": util.now_as_string()}
    meta_path = path.join(base_path, '{}_meta.json'.format(ds_name))
    util.write_to_json(metadata, meta_path)

    print('Data files for "{}" dataset succesfully created '.format(ds_name) +
          '({} vertices, {} edges).'.format(vx_count, edge_count))
    print('For details, see: {}'.format(meta_path))


def prepare_chrono_perc_dataset(ts_edges, settings, vx_count, edge_count,
                                base_path):
    ds_name = settings["name"]
    test_frac = settings["test_perc"] / 100
    test_edges_count = int(edge_count * test_frac)
    train_edges_count = edge_count - test_edges_count
    test_path = path.join(base_path, '{:03}_{}.test.csv'.format(1, ds_name))
    train_path = path.join(base_path, '{:03}_{}.train.csv'.format(1, ds_name))

    edges = util.triples_to_rear_pairs(ts_edges)

    util.write_edges_to_file(edges[:train_edges_count], train_path)
    util.write_edges_to_file(edges[train_edges_count:], test_path)

    metadata = {"name": ds_name,
                "vertices": vx_count,
                "edges": edge_count,
                "set_count": 1,
                "format_type": "basic-edge-list",
                "split_method": settings["split_method"],
                "training_sets_size": train_edges_count,
                "test_sets_size": test_edges_count,
                "created": util.now_as_string()}
    meta_path = path.join(base_path, '{}_meta.json'.format(ds_name))
    util.write_to_json(metadata, meta_path)

    print('Data files for "{}" dataset succesfully created '.format(ds_name) +
          '({} vertices, {} edges).'.format(vx_count, edge_count))
    print('For details, see: {}'.format(meta_path))


def prepare_chrono_from_dataset(ts_edges, settings, vx_count, edge_count,
                                base_path):
    ds_name = settings["name"]
    test_path = path.join(base_path, '{:03}_{}.test.csv'.format(1, ds_name))
    train_path = path.join(base_path, '{:03}_{}.train.csv'.format(1, ds_name))

    split_ts = util.str_to_utc_ts(settings["test_from"])
    split_index = util.find_utc_edges_split_index(ts_edges, split_ts)
    edges = util.triples_to_rear_pairs(ts_edges)

    util.write_edges_to_file(edges[:split_index], train_path)
    util.write_edges_to_file(edges[split_index:], test_path)

    metadata = {"name": ds_name,
                "vertices": vx_count,
                "edges": edge_count,
                "set_count": 1,
                "format_type": "basic-edge-list",
                "split_method": settings["split_method"],
                "training_sets_size": split_index,
                "test_sets_size": edge_count - split_index,
                "created": util.now_as_string()}
    meta_path = path.join(base_path, '{}_meta.json'.format(ds_name))
    util.write_to_json(metadata, meta_path)

    print('Data files for "{}" dataset succesfully created '.format(ds_name) +
          '({} vertices, {} edges, {} test e.).'.format(vx_count, edge_count, edge_count - split_index))
    print('For details, see: {}'.format(meta_path))


SPLIT_METHOD_FUNCS = {"random":
                      (get_edges_for_random, preproc.preprocess_simple_graph,
                       prepare_random_datesets),
                      "k-cross-random":
                      (get_edges_for_random, preproc.preprocess_simple_graph,
                       prepare_k_cross_random_datesets),
                      "chrono-perc":
                      (get_edges_for_chrono, preproc.preprocess_chrono_graph,
                       prepare_chrono_perc_dataset),
                      "chrono-from":
                      (get_edges_for_chrono_from, preproc.preprocess_chrono_graph_oldedges,
                       prepare_chrono_from_dataset),
                      "chrono-perc-old":
                      (get_edges_for_chrono, preproc.preprocess_chrono_graph,
                       prepare_chrono_perc_dataset)
                      }


def process_dataset(category, settings, base_path, import_src, import_fmt):
    if settings["disable"]:
        print('Processing dataset "{}" skipped.'.format(settings["name"]))
        return

    output_path = path.join(base_path, category, settings["name"])
    if settings["disable_overwrite"] and path.exists(output_path):
        print('Processing dataset "{}" skipped (already exists)'
              .format(settings["name"]))
        return

    sm = settings["split_method"]

    # DATASET FROM ARXIV DATA
    if category != "import":
        cache_path = path.join(base_path, '.arxiv-cache', category)
        t_from, t_to = util.parse_period(settings["period"])
        arxiv.update_cache(category, cache_path, t_from, t_to)
        t_mid = t_to
        if settings["split_method"] == "chrono-from":
            t_mid = util.parse_datetime(settings["test_from"])
        # LOAD EDGE LIST
        vx_count, edges = \
            SPLIT_METHOD_FUNCS[sm][0](category, cache_path, t_from, t_to, t_mid)
    else:
        src_path = path.join(base_path, '.ext-cache', import_src)
        # LOAD EDGE LIST
        vx_count, edges = \
            dsimp.FORMAT_IMPORT_FUNCS[import_fmt](src_path)

    # PREPROCESS GRAPH: FLATTEN, EXTRACT MCC, etc.
    print("Preprocessing: extracting mcc from main graph ({} vertices)."
          .format(vx_count))
    vx_count, edge_count, edges = SPLIT_METHOD_FUNCS[sm][1](vx_count, edges)

    SPLIT_METHOD_FUNCS[sm][2](edges, settings, vx_count, edge_count,
                              output_path)
