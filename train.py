import numpy as np
from GA import Ga
import networkx as nx
import random
from GA import *



population = Ga(50, 100, 150, 0.92, 0.012, 300)
population.ga_iter()
best_obj = np.array(population.best_obj)
best = np.array(population.best)
np.save('./best_obj_new_a.npy', best_obj)
np.save('./best_new_a.npy', best)

best_obj = np.load('./best_obj_new_a.npy')
print(best_obj)
best_x = np.load('./best_new_80_a.npy')
for x in best_x:
    print(fitness(50, x))
