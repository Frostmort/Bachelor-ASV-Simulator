import sys, time
import heapq

import random
import numpy as np
import matplotlib.pyplot as plt

from map import Map
from utils import Controller
from vessel import Vessel


class Mopso(Controller):
    def __init__(self, x0, xg, the_map, maxiter=150, w=0.5, c1=1.5, c2=0.9, max_=0.9, min_=0.2,
                 swarmsize=100):
        self.start = x0[0:3]
        self.goal = xg[0:3]
        self.world = the_map

        self.w, self.c1, self.c2 = w, c1, c2  # weight,individual best score, global best score
        self.psize = 5              # number of decision variables
        self.searchrange = [-1, 1]  # Search range
        self.swarmsize = swarmsize  # Number of particles
        self.maxiter = maxiter  # Number of iterations
        self.vmax = (max_ - min_) * 0.05  # Maximum speed
        self.vmin = (max_ - min_) * 0.05 * (-1)  # Minimum speed
        self.vesselArray = vesselArray
        self.is_initialized = False
        self.x = []
        self.v = []
        self.best_all = []

    def update(self,vesselArray):                   #main loop for search
        vesselarray=vesselArray


        if not self.is_initialized:
            self.initialize_swarm(self.swarmsize, self.psize)
            self.is_initialized = True
        fitness = self.calculate_fitness(self.x, self.v)
        self.update_particles(self.x, self.v)
        self.calculate_fitness(self.x, self.v)




    def initialize_swarm(self, swarmsize, psize):
        for j in range(swarmsize):
            self.x.append([random.random() for i in range(psize)])
            self.v.append([random.random() for m in range(psize)])


    def calculate_fitness(self, x, v):
        fitness = [self.fun(x[j]) for j in range(self.swarmsize)]
        p = x
        best = min(fitness)
        pg = x[fitness.index(min(fitness))]
        best_all = []
        return p, best, pg, best_all

    def update_particles(self, x, v):
        fitness = [self.fun(x[j]) for j in range(self.swarmsize)]
        p = x
        best = min(fitness)
        pg = x[fitness.index(min(fitness))]
        for i in range(self.maxiter):
            for j in range(self.swarmsize):
                for m in range(self.psize):
                    v[j][m] = self.w * v[j][m] + self.c1 * random.random() * (
                            p[j][m] - x[j][m]) + self.c2 * random.random() * (pg[m] - x[j][m])
            for j in range(self.swarmsize):
                for m in range(self.psize):
                    x[j][m] = x[j][m] + v[j][m]
                    if x[j][m] > self.searchrange[1]:
                        x[j][m] = self.searchrange[1]
                    if x[j][m] < self.searchrange[0]:
                        x[j][m] = self.searchrange[0]
            fitness_ = []
            for j in range(self.swarmsize):
                fitness_.append(self.fun(self.x[j]))
            if min(fitness_) < best:
                pg = self.x[fitness_.index(min(fitness_))]
                best = min(fitness_)
            self.best_all.append(best)

            print('the ' + str(i) + 'rd iteration: the optimal solution position is in ' + str(
                pg) + ', the fitness value of the optimal solution is:' + str(best))

        plt.plot([i for i in range(self.maxiter)], self.best_all)
        plt.ylabel('fitness value')
        plt.xlabel('Number of iterations')
        plt.title('Particle swarm fitness trend')


        plt.show()

    def fun(self, x):
        result = 0
        for i in x:
            result = result + pow(i, 2)
        return result


if __name__ == "__main__":
    mymap = Map("s1", gridsize=1.0, safety_region_length=4.5)

    x0 = np.array([0, 0, np.pi / 2, 3.0, 0.0, 0])
    xg = np.array([100, 100, np.pi / 4])
    myvessel = Vessel(x0, xg, 0.05, 0.5, 1, [], True, 'viknes')
    vesselArray=[myvessel]
    mopso = Mopso(x0, xg, mymap)

    mopso.update(myvessel)

    fig = plt.figure()
    ax = fig.add_subplot(111, autoscale_on=False)
    ax.plot(myvessel.waypoints[:, 0],
            myvessel.waypoints[:, 1],
            '-')

    # ax.plot(x0[0], x0[1], 'bo')
    ax.plot(xg[0], xg[1], 'ro')
    myvessel.draw_patch(ax, myvessel.x, fcolor='b')

    # nonpass = np.array(nonpassable)
    # ax.plot(nonpass[:,0], nonpass[:,1], 'rx')

    ax.axis('equal')
    ax.axis('scaled')
    ax.axis([-10, 160, -10, 160])
    mymap.draw(ax, 'g', 'k')
    ax.grid()

 #   tikz_save('../../../latex/fig/testfig2.tikz',
 #             figureheight='8cm',
 #             figurewidth='8cm',
 #             textsize=11)

    plt.show()
