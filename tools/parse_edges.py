import json
import sys
import os


def is_src_file(category, srcDir, filename):
    if not os.path.isfile(os.path.join(srcDir, filename)):
        return False
    if filename.startswith(category) and filename.endswith(".json"):
        return True
    return False


def process_src_file(srcDir, filename):
    year = int(filename[-9:-5])
    filepath = os.path.join(srcDir, filename)
    return year, filepath


def get_src_filenames(category, srcDir):
    if not os.path.isdir(srcDir):
        raise BaseException("Invalid source directory")
    else:
        return [process_src_file(srcDir, fn) for fn in os.listdir(srcDir)
                if is_src_file(category, srcDir, fn)]


def load_author_ids_file(category, srcDir, auth_ids):
    authid_filename = "{}_author_ids.txt".format(category)
    authid_path = os.path.join(srcDir, authid_filename)
    with open(authid_path, 'r', encoding='utf-8') as au_file:
        for line in au_file:
            if len(line) > 0 and line[0] != '#':
                tokens = line.split(';')
                auth_ids[tokens[1].rstrip()] = int(tokens[0])


def preproc_article_data_file(srcPath):
    with open(srcPath, 'r', encoding='utf-8') as art_file:
        art_data = json.load(art_file)
        nEdges = 0
        for art in art_data:
            nAuth = len(art["authors"])
            if nAuth > 1:
                nEdges += (nAuth * (nAuth - 1)) / 2
        return len(art_data), int(nEdges)


def get_authid_pairs(auth_ids, authors):
    ids = [auth_ids[name] for name in authors]
    pairs = []
    for id1 in range(len(ids)):
        for id2 in range(id1 + 1, len(ids)):
            pairs.append((ids[id1], ids[id2]))
    return pairs


def parse_article_data_file(artIx, nYear, category, year, srcPath, dstDir,
                            auth_ids):
    nArt, nEdge = preproc_article_data_file(srcPath)
    print("[{:2}/{:2}] Parsing {} ({} articles, {} edges expected)..."
          .format(artIx, nYear, srcPath, nArt, nEdge))
    index = 0
    edges_filename = "{}_edges_{}.edges.csv".format(category, year)
    edges_filepath = os.path.join(dstDir, edges_filename)

    with open(srcPath, 'r', encoding='utf-8') as art_file:
        art_data = json.load(art_file)
        with open(edges_filepath, 'w', encoding='utf-8') as edges_file:
            edges_file.write('# {}\n'.format(nEdge))
            edges_file.write('# edge_id;author1_id;' +
                             'author2_id;published_date\n')
            for art in art_data:
                pub_date = art["published"]
                id_pairs = get_authid_pairs(auth_ids, art["authors"])
                for id1, id2 in id_pairs:
                    edges_file.write('{};{};{};{}\n'
                                     .format(index, id1, id2, pub_date))
                    index += 1

    print("DONE. {} edge entries written to: {}".format(index, edges_filepath))
    return index


def parse_edges_data(category, srcDir, dstDir):
    print('Loading list of authors and preassigned ids.')
    auth_ids = {}
    load_author_ids_file(category, srcDir, auth_ids)
    print('Obtained dictionary of {} author name-id pairs.'
          .format(len(auth_ids)))

    print('Parsing raw article data files...')
    nEdge = 0
    index = 1
    art_files = get_src_filenames(category, srcDir)
    for year, filepath in art_files:
        nEdge += parse_article_data_file(index, len(art_files), category, year,
                                         filepath, dstDir, auth_ids)
        index += 1
    return nEdge


if __name__ == '__main__':
    if len(sys.argv) < 2 or len(sys.argv) > 4:
        raise BaseException('Invalid number of command line arguments.')
    else:
        category = sys.argv[1]
        srcDir = "."
        dstDir = "."

        if len(sys.argv) > 2:
            srcDir = sys.argv[2]
            if len(sys.argv) > 3:
                dstDir = sys.argv[3]

        print('Parsing raw data (\"{}\") to coauthorsip edges data.'
              .format(category))
        nEdge = parse_edges_data(category, srcDir, dstDir)
        print('Edge data successfully processed and saved (total of {} edges).'
              .format(nEdge))
