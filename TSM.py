import json
import networkx as nx
import re
from sklearn.cluster import KMeans
from sklearn import metrics
from itertools import combinations_with_replacement, product
import matplotlib.pyplot as plt
import numpy as np
import multiprocessing
import time
import nodevectors
import csrgraph as cg
import random as rm
from itertools import permutations, combinations
from time import sleep
# Copied from https://www.geeksforgeeks.org/prims-minimum-spanning-tree-mst-greedy-algo-5/
# and modified
# implementation of traveling Salesman Problem using Prims

class Prim_TSM():
    def __init__(self, G, start):
        self.start = start
        self.G = G


      # A utility function to find the vertex with
      # minimum distance value, from the set of vertices
      # not yet included in shortest path tree
    def minKey(self, key, mstSet):
            # Initialize min value
        min = np.Inf

        for v in self.G.nodes:
            if key[v] < min and mstSet[v] == False:
                min = key[v]
                min_index = v

        return min_index

      # Function to construct and print MST for a graph
      # represented using adjacency matrix representation
    def h_cycle(self, path):
         weight = 0
         for i in range(1,len(path)):
             weight += self.G.edges[path[i-1], path[i]]['weight']
         weight += self.G.edges[self.start, path[-1]]['weight']
         # print("Final weight using MST: {}".format(weight))
         return weight

    def primMST(self):
        found = []
            # Key values used to pick minimum weight edge in cut
        key = [np.Inf] * len(self.G.nodes)
        parent = [None] * len(self.G.nodes)  # Array to store constructed MST
            # Make key 0 so that this vertex is picked as first vertex
        key[self.start] = -1
        mstSet = [False] * len(self.G.nodes)

        parent[0] = self.start  # First node is always the root of

        for cout in range(len(self.G.nodes)):

                  # Pick the minimum distance vertex from
                  # the set of vertices not yet processed.
                  # u is always equal to src in first iteration
            u = self.minKey(key, mstSet)

                  # Put the minimum distance vertex in
                  # the shortest path tree
            mstSet[u] = True
            found.append(u)
                  # Update dist value of the adjacent vertices
                  # of the picked vertex only if the current
                  # distance is greater than new distance and
                  # the vertex in not in the shortest path tree
            for edge in self.G.edges(u):

                        # graph[u][v] is non zero only for adjacent vertices of m
                        # mstSet[v] is false for vertices not yet included in MST
                        # Update the key only if graph[u][v] is smaller than key[v]
                if mstSet[edge[1]] == False and key[edge[1]] > self.G.edges[u, edge[1]]['weight']:
                    key[edge[1]] = self.G.edges[u, edge[1]]['weight']
                    parent[edge[1]] = u
        return self.h_cycle(found)


class AntColony:

    def __init__(self, G, start, max_iters, alpha, beta, num_ants, beez_kneez_ants, decay, reset_g=True, check_rate=5, threads=multiprocessing.cpu_count()):
        self.hive = G
        self.start = start
        self.max_iters = max_iters
        self.alpha = alpha
        self.beta = beta
        self.num_ants = num_ants
        self.beez_kneez_ants = beez_kneez_ants
        self.decay = decay
        self.threads = threads
        self.check_rate = check_rate
        self.best_path = {'path':None, 'weight':len(self.hive.nodes)*4, 'iter':None}
        self.used_paths_lens = []
        nx.set_edge_attributes(self.hive, 0.1, "p_global")
        nx.set_node_attributes(self.hive, np.Inf, "path")
        self.degrees = sorted(self.hive.degree, key=lambda x: x[1], reverse=True)

    def simulate(self):
        paths = []
        aver_paths = []
        best_paths = []
        # print('finding cycle from node {}'.format(start))
        for i in range(self.max_iters):
            start_time = time.time()
            # print("Starting iter: {}".format(i))
            nx.set_edge_attributes(self.hive, 1/len(self.hive.nodes)*5, "p_local")
            start_time = time.time()
            paths, final_weights = self.find_paths(self.start)
            aver_paths.append(sum(final_weights) / len(final_weights))
            best_paths.append(max(final_weights))
            # print("finding paths took %s seconds" % (time.time() - start_time))
            self.update_global_pheronome(paths, final_weights, i)
            self.pheronome_decay()
            if i % 100 == 0:
                self.alpha *= .9
            # if self.check_convergence():
            #     break
            # print("iter: %s took %s seconds" % (i, time.time() - start_time))
        # final_path = ''
        # plt.scatter(range(0,len(aver_paths)), aver_paths, label="Average Path")
        # z = np.polyfit(range(len(aver_paths)), aver_paths, 1)
        # p = np.poly1d(z)
        # plt.plot(range(len(aver_paths)), p(range(len(aver_paths))), label="Average Trend")
        # plt.scatter(range(0,len(best_paths)), best_paths, label="Best Path")
        # z = np.polyfit(range(len(best_paths)), best_paths, 1)
        # p = np.poly1d(z)
        # plt.plot(range(len(best_paths)), p(range(len(best_paths))), label="Best Trend")
        # plt.xlabel('Iteration')
        # plt.ylabel('Average Path')
        # plt.title('Average vs Best Path Length for TSM')
        # plt.legend()
        # plt.savefig("Average vs Best Paths")
        # plt.clf()
        # print("Final Weight using ants: {}".format(self.best_path['weight']))
        return self.best_path['weight']

    def find_paths(self, start):
        paths = [[start] for x in range(self.num_ants)]
        weights = [[] for x in range(self.num_ants)]
        done = [False] * self.num_ants
        num_finished = 0

        for i in range(self.num_ants):
            done = False
            while not done:
                p = []
                nodes = []
                for edge in self.hive.edges(paths[i][-1]):
                    if len(self.hive.nodes) == len(paths[i]):
                        if edge[1] == paths[i][0]:
                            nodes.append(edge[1])
                            p = [1]
                            done = True
                            break
                    elif edge[1] not in paths[i]:
                        p.append((self.hive.edges[edge[0],edge[1]]['p_global'] + self.hive.edges[edge[0],edge[1]]['p_local']) ** self.alpha *
                        (( 1.0 / self.hive.edges[edge[0],edge[1]]['weight']) ** self.beta))
                        nodes.append(edge[1])
                p_sum = sum(p)
                probs = [x / p_sum for x in p]
                edge = np.random.choice(nodes, 1, p = probs)
                self.hive.edges[paths[i][-1], edge[0]]['p_local'] *= (1 - self.decay)
                paths[i].append(edge[0])
                weights[i].append(self.hive.edges[paths[i][-2],edge[0]]['weight'])
        final_weights = [sum(x) for x in weights]
        # print("average path weight: {}".format(sum(final_weights) / len(final_weights)))
        return paths, final_weights

    def update_global_pheronome(self, paths, weights, iter):
        best_ant = 0
        for i in range(1,len(paths)):
            if weights[i] < weights[best_ant]:
                best_ant = i
        # print('path found was of weight {}'.format(weights[best_ant]))

        self.used_paths_lens.append(weights[best_ant])
        for i in range(1,len(paths[best_ant])):
            self.hive.edges[paths[best_ant][i-1], paths[best_ant][i]]['p_global'] += 1 / (weights[best_ant])
            if weights[best_ant] < self.best_path['weight']:
                self.hive.edges[paths[best_ant][i-1], paths[best_ant][i]]['p_global'] += 1 / (weights[best_ant] * 2)
        if weights[best_ant] < self.best_path['weight']:
            self.best_path['weight'] = weights[best_ant]
            self.best_path['path'] = paths[best_ant]
            self.best_path['iter'] = iter
        return

    def pheronome_decay(self):
        for edge in list(self.hive.edges):
            self.hive.edges[edge[0], edge[1]]['p_global'] *= (1 - self.decay)
        return

def random_path(G, start):
    shortest_path = 100
    for i in range(10000):
        cur_path = 0
        cur_location = start
        path = list(range(10))
        rm.shuffle(path)
        path.remove(start)
        sleep(.02)
        for i in path:
            cur_path += G.edges[cur_location, i]['weight']
        if cur_path < shortest_path:
            shortest_path = cur_path
    return shortest_path

def main():
    MST_paths = []
    MST_time = []
    ACO_paths = []
    ACO_time = []
    RM_time = []
    RM_paths =[]
    for i in range(0,100):
        nodes = 10
        G = nx.complete_graph(nodes)
        start = rm.randint(0, nodes-1)
        for (u, v) in G.edges():
            G.edges[u,v]['weight'] = rm.randint(1,10)
        start_time = time.time()
        C = AntColony(G, start, 1000, 2, 1, 10, 1, .99, reset_g=True)
        path = C.simulate()
        stop = time.time() - start_time
        ACO_paths.append(path)
        ACO_time.append(stop)
        # print("Ants took %s seconds" % (time.time() - start_time))
        start_time = time.time()
        prims = Prim_TSM(G, start)
        path = prims.primMST()
        stop = time.time() - start_time
        MST_paths.append(path)
        MST_time.append(stop)
        start_time = time.time()
        path = random_path(G, start)
        stop = time.time() - start_time
        RM_paths.append(path)
        RM_time.append(stop)
    print("Average ACO Path: {}".format(sum(ACO_paths)/len(ACO_paths)))
    print("Average ACO Time: {}".format(sum(ACO_time)/len(ACO_time)))
    print("Average MST Path: {}".format(sum(MST_paths)/len(MST_paths)))
    print("Average MST Path: {}".format(sum(MST_time)/len(MST_time)))
    print("Average MST Path: {}".format(sum(RM_paths)/len(RM_paths)))
    print("Average MST Path: {}".format(sum(RM_time)/len(RM_time)))
        # print("MST took %s seconds" % (time.time() - start_time))
    # C.save_run()

if __name__ == "__main__":
    main()
