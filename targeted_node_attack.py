import numpy as np
from matplotlib import pyplot as plt
from GA import Ga
from GA import *
from operator import attrgetter
from collections import namedtuple
import networkx as nx
import operator

plt.rc('font', family='Times New Roman', size=12)

Node = namedtuple('Node', 'node, degree')

obj_result = np.load('./best_obj_new_80.npy')
result = np.load('./best_new_80.npy')
best_result = result[40]
best_obj = fitness(50, best_result)

node_best = best_result.sum(axis=0)
node_best_sorted = sorted(node_best, reverse=True)
# print(node_best_sorted)
a = np.zeros((50, 50))
for i in range(50):
    for j in range(i, 50):
        p = random.randint(1, 100) / 100
        if p < 0.12:
            a[i][j] = 1
            a[j][i] = 1

best_a = fitness(50, a)

g = nx.Graph()
for i in range(i):
    g.add_node(i)
    for j in range(i, 50):
        if best_result[i][j] == 1:
            g.add_edge(i, j)
x_edge_betnx = nx.edge_betweenness_centrality(g)
print(x_edge_betnx)

x_edge_betnx_sorted = sorted(x_edge_betnx.items(), key=operator.itemgetter(1), reverse=True)
x_opt = []

for i in range(100):
    edge_drop_num = int(len(x_edge_betnx) * i / 100)
    edge_drop_set = x_edge_betnx_sorted[:edge_drop_num]
    x = best_result.copy()
    for e in edge_drop_set:
        x[e[0][0]][e[0][1]] = 0
        x[e[0][1]][e[0][0]] = 0
    x_opt.append(fitness(50, x) / best_obj)

g_a = nx.Graph()
for i in range(50):
    g_a.add_node(i)
    for j in range(i, 50):
        if a[i][j] == 1:
            g_a.add_edge(i, j)

a_edge_betnx = nx.edge_betweenness_centrality(g_a)
a_edge_betnx_sorted = sorted(a_edge_betnx.items(), key=operator.itemgetter(1), reverse=True)
a_opt = []
for i in range(100):
    edge_drop_num = int(len(a_edge_betnx) * i / 100)
    edge_drop_set = a_edge_betnx_sorted[:edge_drop_num]
    a_c = a.copy()
    for e in edge_drop_set:
        a_c[e[0][0]][e[0][1]] = 0
        a_c[e[0][1]][e[0][0]] = 0
    a_opt.append(fitness(50, a_c) / best_a)

plt.figure(figsize=(6, 3))
a1, = plt.plot(x_opt, marker='o', markerfacecolor='none', label='w/ optimization')
a2, = plt.plot(a_opt, marker='^', markerfacecolor='none', label='w/o optimization')
plt.xlabel('Number of nodes removed(%)')
plt.ylabel('Normalized natural connectivity')
plt.grid(axis="y", ls="-", color="gray", alpha=0.7)
plt.legend(loc='best')

plt.tight_layout()
# plt.savefig('./5.pdf', format='pdf')
plt.show()
