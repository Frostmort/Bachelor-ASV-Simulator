#!/usr/bin/env python

"""
Rapidly-exploring Random Tree
"""

import time

import numpy as np
import copy

from matplotlib import pyplot as plt
from matplotlib import cm

from map import Map
from utils import Controller

from matplotlib2tikz import save as tikz_save

class RRT(Controller):
    def __init__(self, x0, xg, the_map):
        self.start = x0[0:2]
        self.goal  = xg[0:2]

        self.map = the_map

        self.T = Tree(self.start)

        self.max_iter = 500
        self.eps = 2
        self.goal_bias = 100
        self.counter = 0

    def update(self, vobj):
        ii = 0
        while ii < self.max_iter:
            xrand = self.get_random_state()
            xnear = self.T.get_nearest_neighbor(xrand)
            u     = self.select_input(xrand, xnear)
            xnew  = self.new_state(xnear, u)

            if self.map.is_occupied_discrete(xnew):
               continue

            self.T.add_vertex(xnew)
            self.T.add_edge(xnear, xnew, u)

            if np.linalg.norm(self.goal-xnew) < self.eps:
                # Goal found
                print("GOAL FOUND")
                break

            ii += 1

        # Find shortest path through tree
        # :todo: find path

    def get_random_state(self):
        self.counter += 1
        if self.counter % self.goal_bias == 0:
            return self.goal[0:2]
        else:
            xnew = np.random.random(2)*160
            return xnew

    def select_input(self, xrand, xnear):
        return np.arctan2(xrand[1]-xnear[1],
                          xrand[0]-xnear[0])

    def new_state(self, xnear, u):
        return xnear + np.array([np.cos(u), np.sin(u)])*5.

    def draw(self, axes, n, vcolor='b', ecolor='k'):
        self.T.draw(axes, vcolor, ecolor)

        axes.plot(self.start[0], self.start[1], 'g.', ms=15, label="Start")
        axes.plot(self.goal[0], self.goal[1], 'r.', ms=15, label="Goal")

class Tree(object):
    def __init__(self, start):
        self.V = np.array([start])
        self.E = {}
        self.E[tuple(start)] = []

    def add_vertex(self, xnew):
        self.V = np.vstack((self.V,xnew))

    def add_edge(self, xnear, xnew, u):
        self.E[tuple(xnear)] += [tuple(xnew)]
        self.E[tuple(xnew)] = []

    def get_nearest_neighbor(self, xrand):
        e = self.V - np.array([xrand])
        e = np.sum(np.abs(e)**2,axis=-1)**(1./2)

        return self.V[np.argmin(e)]

    def draw(self, axes, vcolor='b', ecolor='k'):
        #axes.plot(self.V[:,0], self.V[:,1], vcolor+'.', ms=10)

        for node, subnodes in self.E.iteritems():
            for snode in subnodes:
                x,y = zip(*[node,snode])
                axes.plot(x,y, ecolor)



if __name__ == '__main__':

    fig = plt.figure()
    ax  = fig.add_subplot(111, autoscale_on=False)


    the_map = Map('s1', safety_region_length=4.)
    x0 = np.array([5.,6.,0.3,0,0,0])
    xg = np.array([60.,64.,2.])

    rrt = RRT(x0, xg, the_map)

    rrt.update(None)

    the_map.draw(ax)
    rrt.draw(ax,0)
    ax.axis('scaled')
    ax.set_xlim((-10, 160))
    ax.set_ylim((-10, 160))
    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')

    tikz_save('../../../latex/fig/rrt-test4.tikz',
              figureheight='1.5\\textwidth',
              figurewidth='1.5\\textwidth')

    ax.legend(numpoints=1)
    plt.show()
