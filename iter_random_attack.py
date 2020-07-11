# Created by Hanlong Liao
# 2020/7/2

import numpy as np
from matplotlib import pyplot as plt
from GA import Ga
from GA import *

plt.rc('font', family='Times New Roman', size=12)

obj_result = np.load('./best_obj_new_80.npy')[:60]
plt.figure(figsize=(8, 3))
a1, = plt.plot(obj_result, marker='o', markerfacecolor='none')
plt.xlabel('Iterations times')
plt.ylabel('Natural connectivity of Graphs')
plt.grid(axis="y", ls="-", color="gray", alpha=0.7)
plt.tight_layout()
# plt.savefig('./3.pdf', format='pdf')
plt.show()

result = np.load('./best_new_80.npy')
best_result = result[40]
best_obj = fitness(50, best_result)

node_best = best_result.sum(axis=0)
node_best_sorted = sorted(node_best, reverse=True)
print(node_best_sorted)
a = np.zeros((50, 50))
for i in range(50):
    for j in range(i, 50):
        p = random.randint(1, 100) / 100
        if p < 0.12:
            a[i][j] = 1
            a[j][i] = 1

node = a.sum(axis=0)
node_new = sorted(node, reverse=True)
best_a = fitness(50, a)

nodes = [i for i in range(50)]

x_opt = []
a_opt = []
for i in range(0, 50):
    del_node = random.sample(nodes, i)
    graph_new = np.delete(best_result, del_node, axis=0)
    graph_new = np.delete(graph_new, del_node, axis=1)
    x_opt.append(fitness(50 - i, graph_new) / best_obj)

    a_new = np.delete(a, del_node, axis=0)
    a_new = np.delete(a_new, del_node, axis=1)
    a_opt.append(fitness(50 - i, a_new) / best_a)

plt.figure(figsize=(6, 3))
a1, = plt.plot(x_opt, marker='o', markerfacecolor='none', label='w/ optimization')
a2, = plt.plot(a_opt, marker='^', markerfacecolor='none', label='w/o optimization')
plt.xlabel('Number of nodes removed')
plt.ylabel('Normalized natural connectivity')
plt.grid(axis="y", ls="-", color="gray", alpha=0.7)
plt.legend(loc='best')
plt.tight_layout()
# plt.savefig('./10.pdf', format='pdf')
plt.show()



edge_x = best_result.sum() / 2
edge_a = a.sum() / 2
drop = 0.

x_opt = []
a_opt = []
for i in range(100):
    drop_rate = i / 100
    drop_edge_x = drop_rate * edge_x

    x_new = best_result.copy()
    drop_edge_x_set = []
    while len(drop_edge_x_set) < drop_edge_x:
        v1 = random.sample(nodes, 1)[0]
        nodes_new = [i for i in range(v1, 50)]
        v2 = random.sample(nodes_new, 1)[0]
        # print((v1, v2))
        if ((v1, v2) not in drop_edge_x_set) and best_result[v1][v2] == 1:
            drop_edge_x_set.append((v1, v2))

    for edge in drop_edge_x_set:
        x_new[edge[0]][edge[1]] = 0
        x_new[edge[1]][edge[0]] = 0

    x_opt.append(fitness(50, x_new) / best_obj)

    drop_edge_a = drop_rate * edge_a
    a_new = a.copy()
    drop_edge_a_set = []
    while len(drop_edge_a_set) < drop_edge_a:
        v1 = random.sample(nodes, 1)[0]
        nodes_new = [i for i in range(v1, 50)]
        v2 = random.sample(nodes_new, 1)[0]
        if ((v1, v2) not in drop_edge_a_set) and a[v1][v2] == 1:
            drop_edge_a_set.append((v1, v2))

    for edge in drop_edge_a_set:
        a_new[edge[0]][edge[1]] = 0
        a_new[edge[1]][edge[0]] = 0

    a_opt.append(fitness(50, a_new) / best_a)

plt.figure(figsize=(6, 3))
a1, = plt.plot(x_opt, marker='o', markerfacecolor='none', label='w/ optimization')
a2, = plt.plot(a_opt, marker='^', markerfacecolor='none', label='w/o optimization')
plt.xlabel('Number of nodes removed(%)')
plt.ylabel('Normalized natural connectivity')
plt.grid(axis="y", ls="-", color="gray", alpha=0.7)
plt.legend(loc='best')

plt.tight_layout()
# plt.savefig('./1.pdf', format='pdf')
plt.show()










