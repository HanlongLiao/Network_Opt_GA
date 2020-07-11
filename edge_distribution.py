import numpy as np
from matplotlib import pyplot as plt
from GA import *
plt.rc('font', family='Times New Roman', size=12)
result = np.load('./best_new_80.npy')
best_result = result[40]
nodes = best_result.sum(axis=0)
nodes = sorted(nodes)
print(nodes)
node_d = [0] * 25
for d in nodes:
    node_d[d] += 1

plt.figure(figsize=(6, 3))
plt.bar([i for i in range(25)], node_d)
plt.xlabel('Degree of node')
plt.ylabel('Number of nodes')
plt.tight_layout()
plt.grid(axis="y", ls="-", color="gray", alpha=0.7)
plt.tight_layout()
plt.savefig('5.pdf', format='pdf')
plt.show()

a = np.zeros((50, 50))
for i in range(50):
    for j in range(i, 50):
        p = random.randint(1, 100) / 100
        if p < 0.12:
            a[i][j] = 1
            a[j][i] = 1

nodes_a = a.sum(axis=0)
nodes_a = sorted(nodes_a)
print(nodes_a)
node_a_d = [0] * 25
for d in nodes_a:
    node_a_d[int(d)] += 1

plt.figure(figsize=(6, 3))
plt.bar([i for i in range(25)], node_a_d)
plt.xlabel('Degree of node')
plt.ylabel('Number of nodes')
plt.tight_layout()
plt.grid(axis="y", ls="-", color="gray", alpha=0.7)
# plt.savefig('4.pdf', format='pdf')
plt.show()