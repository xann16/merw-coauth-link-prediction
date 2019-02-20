import time
import json
import sys
import os
import httplib2
import datetime
from lxml import etree

httpClient = httplib2.Http()


def get_query_data(category, offset, numEntries, year=None):
    yearPart = ''
    if year:
        yearPart = '+AND+submittedDate:'
        yearPart += '[{}01010000+TO+{}01010000]'.format(year, year + 1)

    response, content = httpClient.request(
        'http://export.arxiv.org/api/query' +
        '?search_query=cat:{}'.format(category) +
        yearPart +
        '&start={}'.format(offset) +
        '&max_results={}'.format(numEntries) +
        '&sortBy=lastUpdatedDate&sortOrder=ascending')

    if response.status != 200:
        raise BaseException('Http reponse error')
    else:
        return etree.fromstring(content)


def get_category_count_and_years(category):
    feed = get_query_data(category, 0, 1)
    countStr = feed.find('{http://a9.com/-/spec/opensearch/1.1/}totalResults')
    elem = feed.find('{http://www.w3.org/2005/Atom}entry')
    publDate = elem.find('{http://www.w3.org/2005/Atom}published').text
    startYear = int(publDate[:4])
    endYear = datetime.datetime.now().year

    return (int(countStr.text), startYear, endYear)


def get_year_count(category, year):
    feed = get_query_data(category, 0, 0, year=year)
    countStr = feed.find('{http://a9.com/-/spec/opensearch/1.1/}totalResults')

    return int(countStr.text)


def get_parsed_data_chunk(articles, authors, category, offset, numEntries,
                          year, maxRepeats=10):
    for repeat in range(0, maxRepeats):
        feed = get_query_data(category, offset, numEntries, year=year)
        entries = feed.findall('{http://www.w3.org/2005/Atom}entry')
        if len(entries) == numEntries:
            break
        if (repeat + 1) == maxRepeats:
            raise BaseException('Failed to get entries after {} trials'
                                .format(repeat + 1))

    for elem in entries:
        title = elem.find('{http://www.w3.org/2005/Atom}title').text
        publDate = elem.find('{http://www.w3.org/2005/Atom}published').text
        localAuthors = []
        for authElem in elem.findall('{http://www.w3.org/2005/Atom}author'):
            author = authElem.find('{http://www.w3.org/2005/Atom}name').text
            localAuthors.append(author)
            authors.add(author)
        prevCount = len(articles)
        newArt = {"title": title,
                  "published": publDate,
                  "authors": localAuthors}
        articles.append(newArt)
        if prevCount == len(articles):
            print('APPEND ERROR: {}'.format(newArt))


def print_progress_info(year, totalCount, currentCount):
    perc = int((currentCount / totalCount) * 100)
    print(' [{}%] Data for year {}: downloaded {} of {}'
          .format(perc, year, currentCount, totalCount),
          end='\r')


def get_parsed_data_for_category(authors, category, chunkSize=500,
                                 queryIntervalInSecs=3, year=None):
    if year:
        begYear = year
        endYear = year
        totalCount = get_year_count(category, year)
    else:
        totalCount, begYear, endYear = get_category_count_and_years(category)

    time.sleep(queryIntervalInSecs)
    print('Attempting to download data on {} articles in category \"{}\"'
          .format(totalCount, category) +
          ' in years {}-{}.'.format(begYear, endYear))
    articleCount = 0

    for year in range(begYear, endYear + 1):
        articles = []
        yearCount = get_year_count(category, year)
        remaining = yearCount
        for offset in range(0, yearCount, chunkSize):
            print_progress_info(year, yearCount, offset)
            time.sleep(queryIntervalInSecs)
            currChunkSize = chunkSize
            if offset + chunkSize > yearCount:
                currChunkSize = yearCount - offset
            get_parsed_data_chunk(articles, authors, category, offset,
                                  currChunkSize, year=year)
        print('Downloaded article data for year {} ({}/{} entries).'
              .format(year, len(articles), yearCount),
              end=' ')
        filename = '{}_articles_{}.json'.format(category, year)
        write_articles_data(articles, filename)
        print('Article data written to file: {}.'.format(filename))
        articleCount += len(articles)
    print('Downloaded data on {} articles out of expected {} from {} category'
          .format(articleCount, totalCount, category))


def write_articles_data(articles, filename):
    with open(filename, 'w', encoding='utf-8') as artFile:
        json.dump(articles, artFile, indent=2)


def write_authors_data(authors, filename):
    print('Writing ids of {} authors to: {}.'.format(len(authors), filename))
    with open(filename, 'w', encoding='utf-8') as auFile:
        index = 0
        auFile.write('# {}\n'.format(len(authors)))
        auFile.write('# author_id;author_name\n'.format(len(authors)))
        for auth in authors:
            auFile.write('{};{}\n'.format(index, auth))
            index += 1


def preload_authors_set(authors, filename):
    if not os.path.exists(filename):
        return
    print('Preloading current author list from {}.'.format(filename))
    with open(filename, 'r', encoding='utf-8') as auFile:
        for line in auFile:
            if len(line) > 0 and line[0] != '#':
                authors.add(line.split(';')[1].rstrip())


def get_arxiv_data(category, year=None):
    authors = set()
    auFilename = '{}_author_ids.txt'.format(category)
    if year:
        preload_authors_set(authors, auFilename)
    get_parsed_data_for_category(authors, 'gr-qc', year=year)
    write_authors_data(authors, auFilename)


if __name__ == '__main__':
    if len(sys.argv) < 2 or len(sys.argv) > 3:
        raise BaseException('Invalid command line arguments.')
    else:
        category = sys.argv[1]
        year = None
        if len(sys.argv) == 3:
            year = int(sys.argv[2])

        print('Download data from arXiv category \"{}\".'.format(category))
        get_arxiv_data(category, year)
        print('Data successfully downloaded and saved.')
