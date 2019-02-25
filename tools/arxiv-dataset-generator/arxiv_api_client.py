import time
import json
import httplib2
import dsgen_utils as util
from os import path
from lxml import etree


http_client = httplib2.Http()
ARXIV_QUERY_INTERVAL = 3
ARXIV_DATA_CHUNK_SIZE = 100
ARXIV_QUERY_MAX_REPEATS = 10


def update_cache(category, basePath, t_from, t_to):
    y_beg, y_end = util.get_years_range(t_from, t_to)
    for year in range(y_beg, y_end):
        update_yearly_data(category, year,
                           path.join(basePath,
                                     util.get_yearly_file_name(category,
                                                               year)))


def update_yearly_data(category, year, filepath):
    if not is_cached(filepath):
        print('Raw data for {}:{}. Cache not available.'
              .format(category, year))

        total = fetch_article_count(category, year)
        articles = []
        for offset in range(0, total, ARXIV_DATA_CHUNK_SIZE):
            print_progress_info(category, year, total, offset)
            chunk_size = ARXIV_DATA_CHUNK_SIZE
            if offset + chunk_size > total:
                chunk_size = total - offset
            data = fetch_data_chunk(category, year, offset, chunk_size)
            parse_data_chunk(articles, data)
            wait()
        util.write_to_json(articles, filepath)
        print('Raw data for {}:{}. Cache updated ({} of {} articles).'
              .format(category, year, len(articles), total))
    else:
        print('Raw data for {}:{}. Cache present.'
              .format(category, year))


def print_progress_info(category, year, total, current):
    perc = int((current / total) * 100)
    print('[{:2}%] Raw data for {}:{}. Downloaded {} of {}...'
          .format(perc, category, year, current, total),
          end='\r')


def is_cached(filepath):
    if path.exists(filepath) and path.isfile(filepath):
        return True
    return False


def fetch_article_count(category, year):
    feed = fetch_raw_query_data(category, year, 0, 0)
    cnt = feed.find('{http://a9.com/-/spec/opensearch/1.1/}totalResults').text
    return int(cnt)


def wait():
    time.sleep(ARXIV_QUERY_INTERVAL)


def fetch_raw_query_data(category, year, offset, count):
    response, content = http_client.request(
        'http://export.arxiv.org/api/query' +
        '?search_query=cat:{}'.format(category) +
        '+AND+submittedDate:' +
        '[{}01010000+TO+{}01010000]'.format(year, year + 1) +
        '&start={}'.format(offset) +
        '&max_results={}'.format(count) +
        '&sortBy=lastUpdatedDate&sortOrder=ascending')

    if response.status != 200:
        raise BaseException('http client reponse error (status: {})'
                            .format(response.status))
    else:
        return etree.fromstring(content)


def fetch_data_chunk(category, year, offset, count):
    for repeat in range(0, ARXIV_QUERY_MAX_REPEATS):
        feed = fetch_raw_query_data(category, year, offset, count)
        entries = feed.findall('{http://www.w3.org/2005/Atom}entry')
        if len(entries) == count:
            break
        if (repeat + 1) == ARXIV_QUERY_MAX_REPEATS:
            raise BaseException('Failed to get entries after {} trials ({}:{})'
                                .format(repeat + 1, category, year))
    return entries


def parse_data_chunk(articles, entries):
    for elem in entries:
        title = elem.find('{http://www.w3.org/2005/Atom}title').text
        title = title.replace(";", "").strip()
        publ_date = elem.find('{http://www.w3.org/2005/Atom}published').text
        authors = []
        for auth_elem in elem.findall('{http://www.w3.org/2005/Atom}author'):
            author = auth_elem.find('{http://www.w3.org/2005/Atom}name').text
            author = author.replace(";", "").strip()
            if len(author) > 0:
                authors.append(author)
        article = {"title": title,
                   "published": publ_date,
                   "authors": authors}
        articles.append(article)
