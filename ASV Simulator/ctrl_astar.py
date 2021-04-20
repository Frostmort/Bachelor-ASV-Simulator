#!/usr/bin/env python
import sys, time
import heapq

import numpy as np
import matplotlib.pyplot as plt

from map import Map
from vessel import Vessel
from utils import Controller, PriorityQueue

from matplotlib2tikz import save as tikz_save

BIGVAL = 10000.

class AStar(Controller):
    def __init__(self, x0, xg, the_map):
        self.graph = SearchGrid(the_map)
        self.start = self.graph.get_grid_id(x0[0:2])
        self.goal  = self.graph.get_grid_id(xg[0:2])

        self.map   = the_map

        self.to_be_updated = True

    def update(self, vobj, vesselArray):
        if self.to_be_updated:
            tic = time.process_time()
            vobj.waypoints = self.search(vobj)
            print("A-star CPU time: %.3f" % (time.process_time() - tic))
            self.to_be_updated = False

    def search(self, vobj):
        frontier = PriorityQueue()
        frontier.put(self.start, 0)
        came_from = {}
        cost_so_far = {}
        came_from[self.start] = None
        cost_so_far[self.start] = 0

        while not frontier.empty():
            current = frontier.get()

            if current == self.goal:
                break

            for next in self.graph.neighbors(current):
                new_cost = cost_so_far[current] + self.graph.cost(current, next)

                if next not in cost_so_far or new_cost < cost_so_far[next]:
                    cost_so_far[next] = new_cost
                    priority = new_cost + self.heuristic(self.goal, next)
                    frontier.put(next, priority)
                    came_from[next] = current

        # Reconstruct path
        path = [current]
        while current != self.start:
            current = came_from[current]
            path.append(current)

        nppath = np.asarray(path[::-4]) * self.graph.gridsize
        return np.copy(nppath[:])

    def heuristic(self, a,b):
        # D  = 1
        # D2 = np.sqrt(2)*D
        dx = abs(a[0] - b[0])
        dy = abs(a[1] - b[1])

        # dx1 = b[0] - self.goal[0]
        # dy1 = b[1] - self.goal[1]
        # dx2 = self.start[0] - self.goal[0]
        # dy2 = self.start[1] - self.goal[1]
        # cross = abs(dx1*dy2 - dx2*dy1)
        return 20*np.sqrt(dx*dx + dy*dy) #+ cross*0.001

class SearchGrid(object):
    """General purpose N-dimentional search grid."""
    def __init__(self, the_map):
        self.grid     = the_map.get_discrete_grid()
        self.map      = the_map
        self.gridsize = the_map.get_gridsize()
        self.dim      = the_map.get_dimension()
        self.discrete_dim = [int(self.dim[0]/self.gridsize),
                             int(self.dim[1]/self.gridsize)]


        """
        In the discrete map, an obstacle has the value '1'.
        We multiply the array by a big number such that the
        grid may be used as a costmap.
        """
        self.grid *= BIGVAL


    def get_grid_id(self, state):
        """Returns a tuple (x,y) with grid positions."""
        return (int(state[0]/self.gridsize),
                int(state[1]/self.gridsize))

    def in_bounds(self, state):
        return 0 <= state[0] < self.discrete_dim[0] and 0 <= state[1] < self.discrete_dim[1]

    def cost(self, a, b):
        if b[0] > self.discrete_dim[0] or b[1] > self.discrete_dim[1]:
            return 0
        return np.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2)#self.grid[b[0], b[1]]

    def passable(self, state):
        return self.grid[state[0]][state[1]] < BIGVAL

    def neighbors(self, xy):

        results = [(xy[0] + 1, xy[1]    ), (xy[0]    , xy[1] + 1),
                   (xy[0] - 1, xy[1]    ), (xy[0]    , xy[1] - 1),
                   (xy[0] + 1, xy[1] + 1), (xy[0] - 1, xy[1] + 1),
                   (xy[0] - 1, xy[1] - 1), (xy[0] + 1, xy[1] - 1)]

        results = filter(self.in_bounds, results)
        results = filter(self.passable, results)

        return results



if __name__ == "__main__":
    mymap = Map("s2", gridsize=1.0, safety_region_length=5.0)

    x0 = np.array([10, 10, np.pi/4, 3.0, 0.0, 0])
    xg = np.array([140, 140, 5*np.pi/4])
    myvessel = Vessel(x0, xg, 0.05, 0.5, 1, [], True, 'viknes')

    myastar = AStar(x0, xg, mymap)

    myastar.update(myvessel)

    fig = plt.figure()
    ax  = fig.add_subplot(111, autoscale_on=False)

    ax.plot(myvessel.waypoints[:,0],
            myvessel.waypoints[:,1],
            '-')

    ax.plot(x0[0], x0[1], 'bo')
    ax.plot(xg[0], xg[1], 'ro')
    myvessel.draw_patch(ax, myvessel.x, fcolor='b')

    #nonpass = np.array(nonpassable)
    #ax.plot(nonpass[:,0], nonpass[:,1], 'rx')

    ax.axis('scaled')
    ax.set_xlim((-10, 160))
    ax.set_ylim((-10, 160))
    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')
    mymap.draw(ax)
#    ax.grid()
#    plt.tight_layout()

#    tikz_save('../../../latex/fig/'+"contour-hugging-astar-no-safety"+'.tikz',
#              figureheight='1.5\\textwidth',
#              figurewidth='1.5\\textwidth')
    plt.show()

    #plt.show()



