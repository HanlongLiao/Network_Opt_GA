# Created by Hanlong Liao
# 2020/7/2
import networkx as nx
import matplotlib.pyplot as plt
from GA import *
import numpy as np

a = np.zeros((50, 50))
for i in range(50):
    for j in range(i, 50):
        p = random.randint(1, 100) / 100
        if p < 0.12:
            a[i][j] = 1
            a[j][i] = 1

g = nx.Graph()
for i in range(50):
    g.add_node(i)
    for j in range(i, 50):
        if a[i][j] == 1:
            g.add_edge(i, j)

position = nx.circular_layout(g)
nx.draw(g)
plt.figure(figsize=(4, 4))
plt.savefig('random4.pdf', format='pdf')
plt.show()

g = nx.Graph()
result = np.load('./best_new_80.npy')[40]
for i in range(50):
    g.add_node(i)
    for j in range(i, 50):
        if result[i][j] == 1:
            g.add_edge(i, j)

nx.draw(g)
plt.figure(figsize=(4, 4))
plt.savefig('opt4.pdf', format='pdf')
plt.show()