import json
import networkx as nx
import re
from node2vec import Node2Vec
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn import metrics
from itertools import combinations_with_replacement, product
import matplotlib.pyplot as plt
import numpy as np
import multiprocessing

class AntColony:

    def __init__(self, G, max_iters, alpha, beta, num_ants, beez_kneez_ants, decay, check_rate=5, threads=multiprocessing.cpu_count()):
        self.hive = G
        self.max_iters = max_iters
        self.alpha = alpha
        self.beta = beta
        self.num_ants = num_ants
        self.beez_kneez_ants = beez_kneez_ants
        self.decay = decay
        self.threads = threads
        self.check_rate = check_rate
        self.best_paths = []
        nx.set_node_attributes(self.hive, 0.1, "p_global")

    def simulate(self):
        for i in range(self.max_iters):
            nx.set_node_attributes(hive, 0.0, "p_local")
            self.find_paths()
            self.update_global_pheronome()
            self.pheronome_decay()
            self.check_convergence()
            if i % check_rate == 0:
                self.check_path()
        return

    def find_paths(self):
        return

    def update_global_pheronome(self):
        return

    def pheronome_decay(self):
        return

    def check_path(self):
        return

    def check_convergence(self):
        return


def import_data():
    file = open('data.txt')
    G = nx.Graph()
    for line in file:
        year = 0
        data = json.loads(line)
        nodes = []
        nodes_only = []
        companies = []
        for key in data.keys():
            if key == 'year':
                year = data[key]
                continue
            elif key == 'title':
                nodes.append((data[key].strip(), dict(type=key)))
                nodes_only.append(data[key].strip())
            else:
                elements = list(data[key])
                for element in elements:
                    match = re.search(r'\(\w+\)\|([\w\s]+)]+', element)
                    if match is not None:
                        nodes.append((match.groups(0)[0].strip(), dict(type=key)))
                        nodes_only.append(match.groups(0)[0].strip() )
                        continue
                    match = re.search(r'\[+([\w\s]+).*\]+', element)
                    if match is not None:
                        nodes.append((match.groups(0)[0].strip(), dict(type=key)))
                        nodes_only.append(match.groups(0)[0].strip() )
                        continue
                    match = re.search(r'([\w\s]+)\([\w\s]+\)', element)
                    if match is not None:
                        nodes.append((match.groups(0)[0].strip(), dict(type=key)))
                        nodes_only.append(match.groups(0)[0].strip())
                        continue
                    else:
                        if key == 'companies':
                            companies.append(element.strip())
                            # year == year
                        nodes.append((element.strip(),dict(type=key)))
        G.add_nodes_from(nodes)
        res = [(a, b) for idx, a in enumerate(nodes_only) for b in nodes_only[idx + 1:]]
        G.add_edges_from(res, label=year)
        for company in companies:
            res = [(company, x) for x in nodes_only]
            G.add_edges_from(res, label=year)
    return G

def main():
    G = import_data()
    C = AntColony(G, 10, 1, 2, 10, 1, .5)
    C.simulate()
    # print(G.nodes['Kevin Bacon'])
    # print(nx.shortest_path(G, source='Hot Tub Time Machine', target='Kevin Bacon'))
    # print(G.edges('Hot Tub Time Machine'))
    # print(G.edges('The Dark Night'))
    # print(G.edges('Anne Hathaway'))
    # nx.write_gml(G, 'output.gml', stringizer=None)

if __name__ == "__main__":
    main()
