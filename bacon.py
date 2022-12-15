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
import nodevectors
import csrgraph as cg

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
        self.used_paths_lens = []
        self.bacon = 0
        if reset_g:
            nx.set_edge_attributes(self.hive, 0.1, "p_global")
            nx.set_node_attributes(self.hive, np.Inf, "path")
            remove = [node for node,degree in dict(self.hive.degree()).items() if degree == 0]
            self.hive.remove_nodes_from(remove)
        for node in self.hive.nodes:
            if self.hive.nodes[node]['name'] == 'Kevin Bacon':
                self.bacon = node
                break
        self.degrees = sorted(self.hive.degree, key=lambda x: x[1], reverse=True)
        self.communities = {}
        self.find_communities()

    def simulate(self):
        probs = [(self.degrees[0][1] - x[1]) / self.degrees[0][1] for x in self.degrees]
        p_sum = sum(probs)
        probs = [x / p_sum for x in probs]
        start = list(np.random.choice([x[0] for x in self.degrees], 1,
                    replace=False, p=probs))
        paths = []
        print('finding path from node {}:{}'.format(start[0], self.hive.nodes[start[0]]['name']))
        for i in range(self.max_iters):
            start_time = time.time()
            print("Starting iter: {}".format(i))
            nx.set_edge_attributes(self.hive, 1.0, "p_local")
            start_time = time.time()
            paths = self.find_paths(start)
            print("finding paths took %s seconds" % (time.time() - start_time))
            self.update_global_pheronome(paths)
            self.pheronome_decay()
            if self.check_convergence():
                break
            if i % self.check_rate == 0:
                self.check_path()
            print("iter: %s took %s seconds" % (i, time.time() - start_time))
        final_path = ''
        for node in paths[0]:
            final_path += self.hive.nodes[node]['name'] + ' '
        print("shortest path found was {}:{}".format(len(paths[0]),final_path))
        return

    def find_paths(self, start):
        paths = [[start[0]] for x in range(self.num_ants)]
        done = [False] * self.num_ants
        num_finished = 0
        while False in done:
            for i in range(self.num_ants):
                if done[i] == False:
                    p = []
                    nodes = []
                    for edge in self.hive.edges(paths[i][-1]):
                        if edge[1] not in paths[i] and self.hive.nodes[edge[1]]['type'] != 'companies':
                            if self.communities[edge[1]] == self.communities[self.bacon]:
                                community_mod = .9
                                p.append(((self.hive.edges[edge[0],edge[1]]['p_global'] + self.hive.edges[edge[0],edge[1]]['p_local']) ** self.alpha *
                                (( 1.0 / (abs(self.hive.degree[self.bacon] - self.hive.degree[edge[1]]) + 1)) ** self.beta)) * community_mod)
                            else:
                                community_mod = .1
                                p.append(((self.hive.edges[edge[0],edge[1]]['p_global'] + self.hive.edges[edge[0],edge[1]]['p_local']) ** self.alpha *
                                (( 1.0 / (abs(self.degrees[0][1] - self.hive.degree[edge[1]]) + 1)) ** self.beta)) * community_mod)
                            nodes.append(edge[1])
                    if len(p) > 0:
                        p_sum = sum(p)
                        probs = [x / p_sum for x in p]
                        edge = np.random.choice(nodes, 1, p = probs)
                        if paths[i][-1] == None:
                            print(paths[i], i, done[i])
                        self.hive.edges[paths[i][-1], edge[0]]['p_local'] *= (1 - self.decay)
                        paths[i].append(edge[0])
                        if self.hive.nodes[paths[i][-1]]['name'] == 'Kevin Bacon':
                            done[i] = True
                            # print("Ant {} finished with path len {}".format(i, len(paths[i])))
                    else:
                        paths[i] = [start[0]]
        path_lengths = [len(x) for x in paths]
        print("average path length: {}".format(sum(path_lengths) / len(path_lengths)))
        return paths

    def update_global_pheronome(self, paths):
        paths.sort(key = len)
        print('path found was of len {}'.format(len(paths[0])))
        for ant in range(self.beez_kneez_ants):
            self.used_paths_lens.append(len(paths[ant]))
            self.hive.nodes[paths[ant][0]]["path"] = len(paths[ant])
            for i in range(1,len(paths[ant])):
                if self.hive.nodes[paths[ant][i]]['path'] > len(paths[ant]) - i:
                    self.hive.nodes[paths[ant][i]]['path'] = len(paths[ant]) - i
                self.hive.edges[paths[ant][i-1], paths[ant][i]]['p_global'] += 1 / len(paths[ant])
        return

    def pheronome_decay(self):
        for edge in list(self.hive.edges):
            self.hive.edges[edge[0], edge[1]]['p_global'] *= (1 - self.decay)
        return

    def check_path(self):
        return

    def check_convergence(self):
        if len(self.used_paths_lens) > 5:
            last = self.used_paths_lens[-1]
            for i in range(5):
                if self.used_paths_lens[-1 - i] != last:
                    return False
            return True
        return False

        return

    def save_run(self):
        nx.write_gml(self.hive, 'output.gml', stringizer=None)
        print("Saved Hive")

    def find_communities(self):
        G = cg.read_edgelist('bacon.edg')
        ggvec_model = nodevectors.GGVec()
        embeddings = ggvec_model.fit_transform(G)
        kmeans = KMeans(25)
        kmeans.fit(embeddings)
        nodes = list(sorted(G.nodes()))
        y_hat = kmeans.labels_
        y_hat = list(y_hat)
        print("Detected Communities")
        for i in range(len(nodes)):
            self.communities[nodes[i]] = y_hat[i]

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
                continue
                # nodes.append((data[key].strip(), dict(type=key)))
                # nodes_only.append(data[key].strip())
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
    # nx.write_edgelist(G, "bacon.edg", data=False)
    print("Data import took %s seconds" % (time.time() - start_time))
    return G

def main():
    G = import_data()
    # print(G)
    # G = nx.read_gml('output.gml')
    # print("Data import took %s seconds" % (time.time() - start_time))
    C = AntColony(G, 100, 1, 2, 10, 1, .20, reset_g=True)
    C.simulate()
    C.save_run()

if __name__ == "__main__":
    main()
