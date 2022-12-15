import requests
from bs4 import BeautifulSoup
import re
import networkx as nx
from multiprocessing Process, Queue, Lock

# custom manager to support custom classes
class GraphManager(BaseManager):
    # nothing
    pass

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

def process_page(i, crawler_pages, link_dict, return_dict, lock, G):
    G = nx.DiGraph()
    counter = 0
    while True:
        try:
            page = queue.get(timeout=0.2)
        except Empty:
            continue
        if item is None:
            break
        # print('popped: {} of degree {}'.format(page, G.out_degree(page)))
        if page not in G:
            G.add_node(page)
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
    return_dict[i] = G

def main():
    G.add_node('Philosophy')
    crawler_pages = Queue()
    crawler_pages.put('Philosophy')
    manager = multiprocessing.Manager()
    return_dict = manager.dict()
    link_dict = manager.dict()

    lock = manager.lock()
    add_link(link_dict, '/wiki/Philosophy')
    processes = [Process(target=task, args=(i, crawler_pages, link_dict, return_dict, lock, G)) for i in range(8)]
    for proc in processes:
        proc.join()
    #process subgraphs
    # nx.write_adjlist(G, "wikipedia.adjlist")

if __name__ == '__main__':
    main()
