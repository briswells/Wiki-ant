import requests
from bs4 import BeautifulSoup
import re
import networkx as nx
import multiprocessing

def get_trunk(link):
    result = re.search(r"\/wiki\/(.+)", link)
    try:
        return result.group(1)
    except:
        print("Error extracting link trunk: {}".format(link))
        return False

def add_link(link_dict, link):
    trunk = get_trunk(link)
    if trunk not in link_dict.keys():
        link_dict[trunk] =  "https://en.wikipedia.org" + link

def main():
    G = nx.DiGraph()
    G.add_node('Philosophy')
    link_dict = {}
    add_link(link_dict, '/wiki/Philosophy')
    crawler_pages = ['Philosophy']
    counter = 0
    while len(crawler_pages):
        page = crawler_pages.pop(0)
        # print('popped: {} of degree {}'.format(page, G.out_degree(page)))
        if G.out_degree(page) == 0:
            if counter % 1000 == 0:
                print('Processed {} Pages'.format(counter))
            # print("Processing page: {}".format(page))
            URL = link_dict[page]
            try:
                r = requests.get(URL)
            except:
                print("unable to open connection: {}".format(URL))
                continue
            soup = BeautifulSoup(r.content, 'html5lib') # If this line causes an error, run 'pip install html5lib' or install html5lib
            all_links = soup.select('p a[href]')
            for item in all_links:
                try:
                    re.match("\[.+\]",  str(item.contents[0]))
                    if not re.match("\[.+\]",  str(item.contents[0])) and '#' not in item['href'] and 'index.php' not in item['href'] and '/wiki/' in item['href'] and 'https://' not in item['href']:
                        trunk = get_trunk(item['href'])
                        if trunk != False:
                            crawler_pages.append(trunk)
                            add_link(link_dict, item['href'])
                            G.add_edge(page, get_trunk(item['href']))
                except:
                    print("Error Processing {}".format(item))
            counter+=1
    nx.write_adjlist(G, "wikipedia.adjlist")
if __name__ == '__main__':
    main()
