import time
import json
import sys
import httplib2
from lxml import etree

httpClient = httplib2.Http(".cache")


def get_query_data(category, offset, numEntries):
    response, content = httpClient.request(
        'http://export.arxiv.org/api/query' +
        '?search_query=cat:{}'.format(category) +
        '&start={}'.format(offset) +
        '&max_results={}'.format(numEntries) +
        '&sortBy=lastUpdatedDate&sortOrder=ascending')

    if response.status != 200:
        raise BaseException('Http reponse error')
    else:
        return etree.fromstring(content)


def get_category_article_count(category):
    feed = get_query_data(category, 0, 0)
    countStr = feed.find('{http://a9.com/-/spec/opensearch/1.1/}totalResults')
    return int(countStr.text)


def get_parsed_data_chunk(articles, authors, category, offset, numEntries):
    feed = get_query_data(category, offset, numEntries)
    for elem in feed.findall('{http://www.w3.org/2005/Atom}entry'):
        title = elem.find('{http://www.w3.org/2005/Atom}title').text
        publDate = elem.find('{http://www.w3.org/2005/Atom}published').text
        localAuthors = []
        for authElem in elem.findall('{http://www.w3.org/2005/Atom}author'):
            author = authElem.find('{http://www.w3.org/2005/Atom}name').text
            localAuthors.append(author)
            authors.add(author)
        articles.append({"title": title,
                         "published": publDate,
                         "authors": localAuthors})


def get_parsed_data_for_category(articles, authors, category, chunkSize=100,
                                 queryIntervalInSecs=3):
    totalCount = get_category_article_count(category)
    time.sleep(queryIntervalInSecs)
    totalCount = 35
    chunkSize = 10
    print('Attempting to download data on {} articles in category \"{}\".'
          .format(totalCount, category))
    for offset in range(0, totalCount, chunkSize):
        print('Completed: {}/{}'.format(offset, totalCount), end='\r')
        get_parsed_data_chunk(articles, authors, category, offset, chunkSize)
        time.sleep(queryIntervalInSecs)
    print('Completed: {}/{}'.format(totalCount, totalCount))


def write_articles_data(articles, filename):
    print('Writing parsed data of {} articles to: {}.'
          .format(len(articles), filename))
    with open(filename, 'w', encoding='utf-8') as artFile:
        json.dump(articles, artFile, indent=2)


def write_authors_data(authors, filename):
    print('Writing ids of {} authors to: {}.'.format(len(authors), filename))
    with open(filename, 'w', encoding='utf-8') as auFile:
        index = 0
        auFile.write('# {}\n'.format(len(authors)))
        for auth in authors:
            auFile.write('{}\t{}\n'.format(index, auth))
            index += 1


def get_arxiv_data(category):
    authors = set()
    articles = []

    get_parsed_data_for_category(articles, authors, 'gr-qc')
    write_articles_data(articles, '{}_articles.json')
    write_authors_data(articles, '{}_author_ids.txt')


if __name__ == '__main__':
    if len(sys.argv) != 2:
        raise BaseException('Invalid command line arguments.')
    else:
        category = sys.argv[1]
        print('Download data from arXiv category \"{}\".'.format(category))
        get_arxiv_data(category)
        print('Data successfully downloaded and saved.')
