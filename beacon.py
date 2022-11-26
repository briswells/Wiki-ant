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
import time

class AntColony:

    def __init__(self, G, max_iters, alpha, beta, num_ants, beez_kneez_ants, decay, reset_g=True, check_rate=5, threads=multiprocessing.cpu_count()):
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
        if reset_g:
            nx.set_edge_attributes(self.hive, 0.1, "p_global")
            nx.set_node_attributes(self.hive, np.Inf, "path")
            remove = [node for node,degree in dict(self.hive.degree()).items() if degree == 0]
            self.hive.remove_nodes_from(remove)
        self.degrees = sorted(self.hive.degree, key=lambda x: x[1], reverse=True)

    def simulate(self):
        for i in range(self.max_iters):
            start_time = time.time()
            print("Starting iter: {}".format(i))
            nx.set_edge_attributes(self.hive, 0.0, "p_local")
            paths = self.find_paths()
            self.update_global_pheronome(paths)
            self.pheronome_decay()
            self.check_convergence()
            if i % self.check_rate == 0:
                self.check_path()
            print("iter: %s took %s seconds" % (i, time.time() - start_time))
        return

    def find_paths(self):
        probs = [(self.degrees[0][1] - x[1]) / self.degrees[0][1] for x in self.degrees]
        p_sum = sum(probs)
        probs = [x / p_sum for x in probs]
        starts = np.random.choice([x[0] for x in self.degrees], self.num_ants,
                    replace=False, p=probs)
        paths = [[x] for x in starts]
        done = [False] * self.num_ants
        while False in done:
            for i in range(self.num_ants):
                if done[i] == False:
                    paths[i].append(self.choose_junction(paths[i]))
                    if paths[i][-1] == None:
                        done[i] = True
                    elif self.hive.nodes[paths[i][-1]]['name'] == 'Kevin Bacon':
                        done[i] = True
        return paths

    def update_global_pheronome(self, paths):
        for path in paths:
            # print(path)
            if len(path) < self.hive.nodes[path[0]]["path"] and path[-1] != None:
                self.hive.nodes[path[0]]["path"] = len(path)
                for i in range(1,len(path)):
                    if self.hive.nodes[path[i]]['path'] > len(path) - i:
                        self.hive.nodes[path[i]]['path'] = len(path) - i
                    self.hive.edges[path[i-1], path[i]]['p_global'] += 1 / len(path)
        return

    def pheronome_decay(self):
        for edge in list(self.hive.edges):
            self.hive.edges[edge[0], edge[1]]['p_global'] *= (1 - self.decay)
        return

    def check_path(self):
        return

    def check_convergence(self):
        return

    def choose_junction(self, path):
        p = []
        nodes = []
        for edge in self.hive.edges(path[-1]):
            if edge[1] not in path and self.hive.nodes[edge[1]]['type'] != 'companies':
                p.append((self.hive.edges[edge[0],edge[1]]['p_global'] + self.hive.edges[edge[0],edge[1]]['p_local']) ** self.alpha *
                (( 1.0 / (self.degrees[0][1] - self.hive.degree[edge[1]] + 1)) ** self.beta))
                nodes.append(edge[1])
        if len(p) > 0:
            p_sum = sum(p)
            probs = [x / p_sum for x in p]
            edge = np.random.choice(nodes, 1, p = probs)
            self.hive.edges[path[-1], edge[0]]['p_local'] += ( 1.0 / len(path))
            return edge[0]
        else:
            return None

    def save_run(self):
        nx.write_gml(self.hive, 'output.gml', stringizer=None)
        print("Saved Hive")

def import_data():
    start_time = time.time()
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
                        nodes_only.append(element.strip())
        G.add_nodes_from(nodes)
        res = [(a, b) for idx, a in enumerate(nodes_only) for b in nodes_only[idx + 1:]]
        G.add_edges_from(res, label=year)
        for company in companies:
            res = [(company, x) for x in nodes_only]
            G.add_edges_from(res, label=year)
    G = nx.convert_node_labels_to_integers(G, first_label=0, ordering='default', label_attribute='name')
    print("Data import took %s seconds" % (time.time() - start_time))
    return G

def main():
    G = import_data()
    # start_time = time.time()
    # G = nx.read_gml('output.gml')
    # print("Data import took %s seconds" % (time.time() - start_time))
    C = AntColony(G, 1, 1, 2, 5, 1, .25, reset_g=True)
    C.simulate()
    C.save_run()

if __name__ == "__main__":
    main()
