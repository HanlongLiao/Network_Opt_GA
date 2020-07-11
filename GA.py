# Created by Hanlong Liao
# 2020/7/2

import numpy as np
import networkx as nx
from operator import attrgetter
from collections import namedtuple
import random
import copy

# chromosome with fitness
Chromosome = namedtuple('Chromosome', 'chromosome, fitness')


class Ga:
    def __init__(self, node_num,  popu_size, w, pc, vc, iter_time):
        self.NODE_NUM = node_num
        self.W = w
        self.P_C = pc
        self.POPU_NUM = popu_size
        self.V_C = vc
        self.ITER_TIME = iter_time # GA algorithm iter times
        self.best = []
        self.best_obj = []

    def init_population(self):
        """
        Init the population
        :return: None
        """
        self.x_set = []
        while len(self.x_set) < self.POPU_NUM:
            ch = np.random.randint(0, 2, (self.NODE_NUM, self.NODE_NUM))
            for i in range(self.NODE_NUM):
                ch[i][i] = 0
                for j in range(i, self.NODE_NUM):
                    ch[j][i] = ch[i][j]
            # judge if the graph connected
            if self.judge_connected(ch):
                self.x_set.append(Chromosome(chromosome=ch, fitness=self.fitness(ch)))

        print('Population Initial Ending...')

    def fitness(self, X):
        """
        Fitness Function for GA
        :param X: chromosome
        :return: fitness
        """
        obj_value = np.log(1 / self.NODE_NUM * np.exp(self.get_eig_value(X)).sum())

        return obj_value



    def select(self):
        """
        Select the fix number chromosomes
        1. Roulette algorithm
        2. Electing the top 30% of the population
        :return: Selected chromosomes
        """
        # Roulette algorithm
        # fitness_set = [0] * self.POPU_NUM
        # for i in range(len(self.x_set)):
        #     fitness_set[i] = self.fitness(self.x_set[i])
        # chromosome_sum = 0
        # for x in self.x_set:
        #     chromosome_sum += x.fitness
        #
        # fitness_rle = [0] * self.POPU_NUM
        # fitness_rle[0] = self.x_set[0].fitness / chromosome_sum
        # for i in range(1, self.POPU_NUM):
        #     fitness_rle[i] = fitness_rle[i-1] + self.x_set[i].fitness / chromosome_sum
        #
        # x_selectd = []
        # for i in range(self.POPU_NUM):
        #     p = np.random.rand()
        #     for j in range(self.POPU_NUM):
        #         if fitness_rle[j] >= p:
        #             x_selectd.append(self.x_set[j])
        #             break

        # Selecting the top 30% of the population
        x_set_ = sorted(self.x_set, key=attrgetter('fitness'), reverse=True)
        x_selected = x_set_[:int(0.4 * self.POPU_NUM)]
        return x_selected

    def overlap(self, X1, X2):
        """
        Overlap function for chromosome
        :param X1: chromosome 1
        :param X2: chromosome 2
        :return: overlapped x1 and overlapped x2
        """
        site1 = int(self.NODE_NUM * 1 / 3)
        site2 = int(self.NODE_NUM * 2 / 3)

        X1_new = np.hstack((X2[:, :site1], X1[:, site1:site2]))
        X1_new = np.hstack((X1_new, X2[:, site2:]))
        X2_new = np.hstack((X1[:, :site1], X2[:, site1:site2]))
        X2_new = np.hstack((X2_new, X1[:, site2:]))

        return X1_new, X2_new

    def vatiation(self, X):
        """
        Variation function for chromosome
        :param X: chromosome
        :return: variant chromosome
        """
        for i in range(self.NODE_NUM):
            for j in range(self.NODE_NUM):
                p = np.random.rand()
                if p <= self.V_C:
                    X[i][j] = 1 - X[i][j]

        return X

    def get_eig_value(self, X):
        """
        Get the eig_value and feature matrix
        :param X: chromosome
        :return: eig_value
        """
        eig_value, feature_mat = np.linalg.eig(X)
        return eig_value

    def judge_connected(self, X):
        """Get the minor eigenvalue of matrix
        :param X: chromosome
        :return: minor eigenvalue of matrix
        """
        # node_x = X.sum(axis=0)
        # laplace_x = np.diag(node_x) - X
        #
        g = nx.Graph()
        for i in range(self.NODE_NUM):
            g.add_node(i)
            for j in range(i, self.NODE_NUM):
                if X[i][j] == 1:
                    g.add_edge(i, j)
        connected = nx.is_connected(g)
        return connected

    def recover_graph(self, X):
        for i in range(self.NODE_NUM):
            X[i][i] = 0
            for j in range(i, self.NODE_NUM):
                X[j][i] = X[i][j]

        g = nx.Graph()
        for i in range(self.NODE_NUM):
            g.add_node(i)
            for j in range(i, self.NODE_NUM):
                if X[i][j] == 1:
                    g.add_edge(i, j)

        delta_edge_num = len(g.edges) - self.W

        if delta_edge_num > 0:
            edges = random.sample(g.edges, delta_edge_num)
            for e in edges:
                X[e[0]][e[1]] = 0
                X[e[1]][e[0]] = 0

        elif delta_edge_num < 0:
            delta_edge_num = np.abs(delta_edge_num)
            edge_add = []
            while len(edge_add) < delta_edge_num:
                v1, v2 = random.sample(g.nodes, 2)
                if ((v1, v2) not in g.edges) and ((v2, v1) not in g.edges):
                    if ((v1, v2) not in edge_add) and ((v2, v1) not in edge_add):
                        edge_add.append((v1, v2))

            for e in edge_add:
                X[e[0]][e[1]] = 1
                X[e[1]][e[0]] = 1

        return X

    def local_search(self, X):
        nodes = [i for i in range(self.NODE_NUM)]
        for i in range(self.NODE_NUM):
            x1, x2 = random.sample(nodes, 2)
            x_new = X.copy()
            x_new[x1][x2] = 1 - x_new[x1][x2]
            x_new[x2][x1] = x_new[x1][x2]
            x_new = self.recover_graph(x_new)
            if (self.fitness(x_new) > self.fitness(X)) and self.judge_connected(x_new):
                X = x_new

        return X

    def ga_iter(self):
        self.init_population()

        for iter in range(self.ITER_TIME):
            if self.P_C >= 0.88:
                self.P_C = self.P_C * 0.995
            if self.V_C >= 0.008:
                self.V_C = self.V_C * 0.995

            x_selected = self.select()
            x_son = []
            for ch in x_selected:
                x_son.append(ch.chromosome)

            while len(x_son) < self.POPU_NUM:
                x1, x2 = random.sample(x_selected, 2)
                p = np.random.rand()
                if p <= self.P_C:
                    x1_son, x2_son = self.overlap(x1.chromosome, x2.chromosome)
                    if self.judge_connected(x1_son) and self.judge_connected(x2_son):
                        x_son.append(x1_son)
                        x_son.append(x2_son)
                else:
                    x_son.append(x1.chromosome)
                    x_son.append(x2.chromosome)

            x_new_set = []
            for x in x_son:
                while True:
                    x_new = self.vatiation(x)
                    x_new = self.recover_graph(x_new)
                    if self.judge_connected(x_new):
                        x_new_set.append(x_new)
                        break

            x_local = []
            for i in range(len(x_new_set)):
                local_opt = self.local_search(x_new_set[i])
                x_local.append(Chromosome(chromosome=local_opt, fitness=self.fitness(local_opt)))

            self.x_set = copy.deepcopy(x_local)
            x_local = sorted(x_local, key=attrgetter('fitness'), reverse=True)
            print('ITER: {}, best: {}'.format(iter, x_local[0].fitness))
            if x_local[0].fitness == fitness(50, x_local[0].chromosome):
                print('True')
            else:
                print('False')
            self.best_obj.append(x_local[0].fitness)
            self.best.append(x_local[0].chromosome)


def fitness(node_num, X):
    obj_value = np.log(1 / node_num * np.exp(get_eig_value(X)).sum())
    return obj_value


def get_eig_value(X):
    eig_value, feature_mat = np.linalg.eig(X)
    return eig_value