#!/usr/bin/env python
import sys, time
import numpy as np
import matplotlib.pyplot as plt

from map import Map
from vessel import Vessel
from utils import Controller, PriorityQueue, eucledian_path_length
from matplotlib2tikz import save as tikz_save

import dubins

BIGVAL = 10000.

class HybridAStar(Controller):
    def __init__(self, x0, xg, the_map, replan=False):
        self.start = x0[0:3]
        self.goal  = xg[0:3]

        self.world = None

        self.graph = SearchGrid(the_map, [1.0, 1.0, 25.0/360.0], N=3, parent=self)
        # :todo: Should be the_world?
        self.map = the_map
        self.eps = 5.0
        self.to_be_updated = True
        self.replan = replan
        self.path_found = False

        self.gridsize = the_map.get_gridsize()

        self.turning_radius = 20.0
        self.step_size      = 1.5*the_map.get_gridsize()
        self.dubins_expansion_constant = 50

    def update(self, vessel_object):
        if self.to_be_updated:

            vessel_object.waypoints = self.search(vessel_object)

            self.to_be_updated = False
            #self.map.disable_safety_region()

    def draw(self, axes, n, fcolor, ecolor):
        pass

    def visualize(self, fig, axarr, t, n):
        pass

    def search(self, vobj):
        """The Hybrid State A* search algorithm."""

        tic = time.process_time()

        get_grid_id = self.graph.get_grid_id

        frontier = PriorityQueue()
        frontier.put(list(self.start), 0)
        came_from = {}
        cost_so_far = {}

        came_from[tuple(self.start)] = None
        cost_so_far[get_grid_id(self.start)] = 0

        dubins_path = False

        num_nodes = 0

        while not frontier.empty():
            current = frontier.get()

            if num_nodes % self.dubins_expansion_constant == 0:
                dpath,_ = dubins.path_sample(current, self.goal, self.turning_radius, self.step_size)
                if not self.map.is_occupied_discrete(dpath):
                    # Success. Dubins expansion possible.
                    self.path_found = True
                    dubins_path = True
                    break

            if np.linalg.norm(current[0:2] - self.goal[0:2]) < self.eps \
               and np.abs(current[2]-self.goal[2]) < np.pi/8:
                self.path_found = True
                break

            for next in self.graph.neighbors(current, vobj.model.est_r_max):
                new_cost = cost_so_far[get_grid_id(current)] + \
                           self.graph.cost(current, next)

                if get_grid_id(next) not in cost_so_far or new_cost < cost_so_far[get_grid_id(next)]:
                    cost_so_far[get_grid_id(next)] = new_cost
                    priority = new_cost + heuristic(self.goal, next)
                    frontier.put(list(next), priority)
                    came_from[tuple(next)] = current

            num_nodes += 1
        # Reconstruct path
        path = [current]
        while tuple(current) != tuple(self.start):
            current = came_from[tuple(current)]
            path.append(current)

        if dubins_path:
            path = np.array(path[::-1] + dpath)
        else:
            path = np.array(path[::-2])

        print("Hybrid A-star CPU time: %.3f. Nodes expanded: %d" % ( time.process_time() - tic, num_nodes))
        #print(self.start)

        return np.copy(path)


def heuristic(a, b):
    """The search heuristics function."""
    return np.linalg.norm(a-b)

class SearchGrid(object):
    """General purpose N-dimentional search grid."""
    def __init__(self, the_map, gridsize, N=2, parent=None):
        self.N        = N
        self.grid     = the_map.get_discrete_grid()
        self.map      = the_map
        self.gridsize = gridsize
        self.gridsize[0] = the_map.get_gridsize()
        self.gridsize[1] = the_map.get_gridsize()
        dim = the_map.get_dimension()

        self.width  = dim[0]
        self.height = dim[1]

        """
        In the discrete map, an obstacle has the value '1'.
        We multiply the array by a big number such that the
        grid may be used as a costmap.
        """
        self.grid *= BIGVAL


    def get_grid_id(self, state):
        """Returns a tuple (x,y,psi) with grid positions."""
        return (int(state[0]/self.gridsize[0]),
                int(state[1]/self.gridsize[1]),
                int(state[2]/self.gridsize[2]))

    def in_bounds(self, state):
        return 0 <= state[0] < self.width and 0 <= state[1] < self.height

    def cost(self, a, b):
        #if b[0] > self.width or b[1] > self.height:
        #    return 0
        return np.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2)#self.grid[int(b[0]/self.gridsize[0]), int(b[1]/self.gridsize[1])]

    def passable(self, state):
        # TODO Rename or change? Only returns true if object is _inside_ obstacle
        # Polygons add safety zone by default now.

        if state[0] > self.width or state[1] > self.height:
            return True

        return self.grid[int(state[0]/self.gridsize[0]),
                         int(state[1]/self.gridsize[1])] < BIGVAL

    def neighbors(self, state, est_r_max):
        """
        Applies rudder commands to find the neighbors of the given state.

        For the Viknes 830, the maximum rudder deflection is 15 deg.
        """
        step_length = 2.5*self.gridsize[0]
        avg_u       = 3.5
        Radius      = 2.5*avg_u / est_r_max
        dTheta      = step_length / Radius
        #print(Radius, dTheta*180/np.pi)

        trajectories = np.array([[step_length*np.cos(dTheta), step_length*np.sin(dTheta), dTheta],
                                 [step_length, 0.,  0.],
                                 [step_length*np.cos(dTheta), -step_length*np.sin(dTheta), -dTheta]])

        #print(trajectories)
        results = []
        for traj in trajectories:
            newpoint = state + np.dot(Rz(state[2]), traj)
            if self.passable(newpoint):
                results.append(newpoint)

        #results = filter(self.in_bounds, results)

        return results

def Rz(theta):
    return np.array([[np.cos(theta), -np.sin(theta), 0],
                     [np.sin(theta),  np.cos(theta), 0],
                     [0            ,  0            , 1]])

if __name__ == "__main__":
    mymap = Map("s1", gridsize=1.0, safety_region_length=4.5)

    x0 = np.array([0, 0, np.pi/2, 3.0, 0.0, 0])
    xg = np.array([100, 100, np.pi/4])
    myvessel = Vessel(x0, xg, 0.05, 0.5, 1, [], True, 'viknes')

    myastar = HybridAStar(x0, xg, mymap)

    myastar.update(myvessel)

    fig = plt.figure()
    ax  = fig.add_subplot(111,autoscale_on=False)
    ax.plot(myvessel.waypoints[:,0],
            myvessel.waypoints[:,1],
            '-')

    #ax.plot(x0[0], x0[1], 'bo')
    ax.plot(xg[0], xg[1], 'ro')
    myvessel.draw_patch(ax, myvessel.x, fcolor='b')

    #nonpass = np.array(nonpassable)
    #ax.plot(nonpass[:,0], nonpass[:,1], 'rx')

    ax.axis('equal')
    ax.axis('scaled')
    ax.axis([-10, 160, -10, 160])
    mymap.draw(ax, 'g', 'k')
    ax.grid()

    tikz_save('../../../latex/fig/testfig2.tikz',
              figureheight='8cm',
              figurewidth='8cm',
              textsize=11)

    plt.show()
