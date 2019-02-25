import random
from os import path
from datetime import datetime
import arxiv_api_client as arxiv
import raw_data_parser as parser
import graph_preproc as preproc
import dsgen_utils as util


def get_edges_for_random(category, cache_path, t_from, t_to):
    return parser.load_simple_edge_list(category, cache_path, t_from, t_to)


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
                "format_type": "basic_edge_list",
                "is_maxcc": not settings["disable_maxcc_extraction"],
                "training_sets_size": edge_count - test_edges_count,
                "test_sets_size": test_edges_count,
                "created": util.now_as_string()}
    meta_path = path.join(base_path, '{}_meta.json'.format(ds_name))
    util.write_to_json(metadata, meta_path)

    print('Data files for "{}" dataset succesfully created '.format(ds_name) +
          '({} vertices, {} edges).'.format(vx_count, edge_count))
    print('For details, see: {}'.format(meta_path))


SPLIT_METHOD_FUNCS = {"random":
                      (get_edges_for_random, prepare_random_datesets)}


def process_dataset(category, settings, base_path):
    if not settings["disable"]:
        cache_path = path.join(base_path, '.arxiv-cache', category)
        output_path = path.join(base_path, category, settings["name"])
        t_from, t_to = util.parse_period(settings["period"])

        arxiv.update_cache(category, cache_path, t_from, t_to)

        vx_count, edge_count, edges = \
            SPLIT_METHOD_FUNCS[settings["split_method"]][0](category,
                                                            cache_path,
                                                            t_from,
                                                            t_to)

        if not settings["disable_maxcc_extraction"]:
            vx_count, edge_count, edges = \
                preproc.extract_maximal_connected_component(vx_count, edges)

        SPLIT_METHOD_FUNCS[settings["split_method"]][1](edges,
                                                        settings,
                                                        vx_count,
                                                        edge_count,
                                                        output_path)
