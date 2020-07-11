import numpy as np
from matplotlib import pyplot as plt
from GA import Ga
from GA import *
from operator import attrgetter
from collections import namedtuple

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

drop_nodes = [i for i in range(50)]
x = best_result.copy()
x_degree = x.sum(axis=0)
nodes_x = []
for i in range(50):
    nodes_x.append(Node(node=i, degree=x_degree[i]))

x_opt = []
nodes_x = sorted(nodes_x, key=attrgetter('degree'), reverse=True)
for i in range(50):
    node_drop = [nodes_x[n].node for n in range(i)]
    node_drop_new = np.delete(x, node_drop, axis=0)
    node_drop_new = np.delete(node_drop_new, node_drop, axis=1)
    x_opt.append(fitness(50 - i, node_drop_new) / best_obj)


a_opt = []
nodes_a = []
a_new = a.copy()
a_degree = a.sum(axis=0)
for i in range(50):
    nodes_a.append(Node(node=i, degree=a_degree[i]))

a_degree_new = sorted(nodes_a, key=attrgetter('degree'), reverse=True)
for i in range(50):
    node_a_drop = [a_degree_new[n].node for n in range(i)]
    a_ = np.delete(a_new, node_a_drop, axis=0)
    a_ = np.delete(a_, node_a_drop, axis=1)
    a_opt.append(fitness(50 - i, a_) / best_a)

plt.figure(figsize=(6, 3))
a1, = plt.plot(x_opt, marker='o', markerfacecolor='none', label='w/ optimization')
a2, = plt.plot(a_opt, marker='^', markerfacecolor='none', label='w/o optimization')
plt.xlabel('Number of nodes removed')
plt.ylabel('Normalized natural connectivity')
plt.grid(axis="y", ls="-", color="gray", alpha=0.7)
plt.legend(loc='best')
plt.tight_layout()
# plt.savefig('./1.pdf', format='pdf')
plt.show()
