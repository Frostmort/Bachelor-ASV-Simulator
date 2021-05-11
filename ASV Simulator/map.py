#!/usr/bin/env python

import numpy as np

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm

import copy

import time

class Map(object):
    """This class provides a general map."""
    def __init__(self, maptype=None, gridsize=1.0, safety_region_length=1.0):
        """Initialize map. Default map is blank 160x160m.'"""
        self._dim = [160, 160]
        self._obstacles = []

        if maptype=='s1':
            self._obstacles = [Polygon([(41.44, 39.01),
                                        (32.78, 47.13),
                                        (29.09, 60.75),
                                        (31.65, 75.95),
                                        (39.02, 85.38),
                                        (49.8, 80.14),
                                        (53.49, 65.47),
                                        (49.24, 49.75)], safety_region_length),
                               Polygon([(130.1, 78.7),
                                        (111.86, 59.98),
                                        (90.58, 73.86),
                                        (86.04, 105.04),
                                        (87.89, 122.59),
                                        (108.6, 117.62),
                                        (123.07, 99.54)], safety_region_length)]

        if maptype == 's2':
            self._obstacles = [Polygon([(39.9, 8.28),
                                        (78.06, 26.7),
                                        (84.54, 54.08),
                                        (80.32, 72.17),
                                        (67.06, 98.87),
                                        (49.57, 112.39),
                                        (25.51, 107.15),
                                        (18.31, 87.38),
                                        (24.27, 69.13),
                                        (41.34, 76.73),
                                        (60, 70),
                                        (66.75, 48.17),
                                        (56.26, 28.22),
                                        (42.58, 19.27)], safety_region_length)]

        if maptype == 's3':
            pass
            # self._obstacles = [Polygon([(0, 30),
            #                             (50, 30),
            #                             (60, 40),
            #                             (60, 50),
            #                             (50, 60),
            #                             (0, 60)], safety_region_length),
            #                    Polygon([(100, 0),
            #                             (100, 20),
            #                             (120, 40),
            #                             (120, 80),
            #                             (130, 90),
            #                             (160, 90),
            #                             (160, 0)], safety_region_length)]


        if maptype == 'islands':
            self._obstacles = [Polygon([(30.76, 23.75),
                                        (51.92, 20.79),
                                        (63.32, 35.44),
                                        (64.06, 47.28),
                                        (50.00, 50.00),
                                        (43.64, 35.89)],safety_region_length),
                               Polygon([(24.40, 55.13),
                                        (22.62, 69.04),
                                        (43.04, 76.59),
                                        (47.04, 67.71),
                                        (40.00, 60.00)],safety_region_length),
                               Polygon([(46.45, 94.35),
                                        (85.22, 69.04),
                                        (80.00, 90.00),
                                        (59.77, 94.64)],safety_region_length)]
        if maptype == 'pænis':
            self._dim=[225, 225]
            self._obstacles = [Polygon([(46.1,47.9),
                                        (39.5,55.4),
                                        (26.7,53.9),
                                        (21.0,66.1),
                                        (24.9,82.1),
                                        (37.8,89.5),
                                        (53.2,86.9),
                                        (87.0,113.4),
                                        (98.4,120.2),
                                        (108.9,117.4),
                                        (112.4,107.4),
                                        (108.1,95.2),
                                        (74.1,66.1),
                                        (78.2,51.7),
                                        (70.4,36.6),
                                        (52.1,37.1)], safety_region_length)]


        if maptype == 'tæst':
            self._dim=[225, 225]
            self._dim=[225, 225]
            self._obstacles = [Polygon([(80.2,56.8), (66.1,98.5), (80.8,149.8), (105.6,158.7), (130.4,133.8), (120.1,105.4), (101.2,77.4)], safety_region_length)]

        elif maptype == 'VO_land_test':
            self._dim = [160, 160]
            self._obstacles = [Polygon([(100, 0), (100, 160), (160, 160), (160,0)])]

        elif maptype == 'triangle':
            self._dim = [160, 160]
            self._obstacles = [Polygon([(31.5, 51.5),
                                        (90.5,60.5),
                                        (70.7, 90.3)],safety_region_length)]

        elif maptype == 'polygon':
            self._obstacles = [Polygon([(30.76, 23.75),
                                        (51.92, 20.79),
                                        (63.32, 35.44),
                                        (64.06, 47.28),
                                        (50.00, 50.00),
                                        (43.64, 35.89)],safety_region_length)]

            self._dim = [80, 80]

        elif maptype == 'minima':
            self._obstacles = [Polygon([(39.9, 8.28),
                                        (78.06, 26.7),
                                        (84.54, 54.08),
                                        (80.32, 72.17),
                                        (67.06, 98.87),
                                        (49.57, 112.39),
                                        (25.51, 107.15),
                                        (18.31, 87.38),
                                        (24.27, 69.13),
                                        (41.34, 76.73),
                                        (60, 70),
                                        (66.75, 48.17),
                                        (56.26, 28.22),
                                        (42.58, 19.27)], safety_region_length)]
            self._dim = [160,160]

        elif maptype == 'blank':
            self._dim = [160,160]

        self._is_discretized = False
        self._gridsize = gridsize
        self._grid = None

    def discretize_map(self):
        """
        Creates a discrete version of the map.

        This algorithm is based on the in_polygon-algorithm.
        See: http://alienryderflex.com/polygon_fill/
        """
        tic = time.process_time()

        scale = 1/self._gridsize
        self._discrete_dim = [int(self._dim[0]*scale),
                              int(self._dim[1]*scale)]

        self._grid = np.zeros(self._discrete_dim)

        for o in self._obstacles:
            V = o.get_vertices(safe=True)

            xymax = np.amax(V, axis=0)
            xmax  = np.ceil(xymax[0]*scale) / scale
            ymax  = np.ceil(xymax[1]*scale) / scale

            xymin = np.amin(V, axis=0)
            xmin  = np.floor(xymin[0]*scale) / scale
            ymin  = np.floor(xymin[1]*scale) / scale

            for gridY in np.arange(ymin, ymax, self._gridsize):
                # Build a list of nodes
                xnodes = []
                j     = len(V) - 1 # Index of last vertice
                for i in range(0, len(V)):
                    if (V[i][1] < gridY and V[j][1] >= gridY) or \
                       (V[j][1] < gridY and V[i][1] >= gridY):
                        x = (V[i][0] + \
                             (gridY - V[i][1])/(V[j][1] - V[i][1])*(V[j][0] - V[i][0]))
                        xnodes.append(x)
                    j = i

                # Sort the nodes
                xnodes.sort()

                # Fill the pixels/cells between node pairs
                for i in range(0, len(xnodes), 2):
                    if xnodes[i] >= xmax:
                        # :todo: will this happen?
                        break
                    if xnodes[i+1] > xmin:
                        if xnodes[i] < xmin:
                            # :todo: will this happen?
                            xnodes[i] = xmin
                        if xnodes[i] > xmax:
                            # :todo: will this happen?
                            xnodes[i] = xmax
                        for j in np.arange(xnodes[i], xnodes[i+1], self._gridsize):
                            if int(j*scale) >= self._discrete_dim[0] or int(gridY*scale) >= self._discrete_dim[1]:
                                continue
                            self._grid[int(j*scale), int(gridY*scale)] = 1

        self._is_discretized = True

        print(("Discretization time: ", time.process_time() - tic))

    def load_map(self, filename):
        with open(filename, 'r') as f:
            line = f.readline().split(" ")
            self._dim[0] = int(line[0])
            self._dim[1] = int(line[1])

            obstacles = f.readlines()
            self._obstacles = []

            for line in obstacles:
                o = line.split(" ")
                n = len(o)
                tmp = []
                for ii in range(0,n,2):
                    tmp.append((float(o[ii]), float(o[ii+1])))

                self._obstacles.append(Polygon(tmp))

    def add_obstacles(self, obstacles):
        """Adds obstacles to map."""
        for o in obstacles:
            self._obstacles.append(o)

    def get_discrete_grid(self):
        if not self._is_discretized:
            self.discretize_map()
        return np.copy(self._grid)

    def get_obstacle_points(self):
        if not self._is_discretized:
            self.discretize_map()
        return np.transpose(np.nonzero(self._grid)) * self._gridsize

    def get_dimension(self):
        return copy.copy(self._dim)

    def get_gridsize(self):
        return self._gridsize

    def disable_safety_region(self):
        for o in self._obstacles:
            o.set_safety_region(False)

    def get_obstacles(self):
        """Returns obstacles."""
        return self._obstacles

    def get_obstacle_edges(self):
        edges = []
        for o in self._obstacles:
            edges += list(o.get_vertices())
        return edges

    def get_obstacle_edge_samples(self, d):
        points = []
        for o in self._obstacles:
            points += o.get_edge_points(d)
        return points

    def is_occupied_discrete(self, point):
        if isinstance(point, list):
            for p in point:
                if self.is_occupied_discrete(p):
                    return True

            return False

        if not self._is_discretized:
            self.discretize_map()

        p = int(point[0]/self._gridsize), int(point[1]/self._gridsize)

        if p[0] >= self._discrete_dim[0] or \
           p[1] >= self._discrete_dim[1] or \
           p[0] < 0 or \
           p[1] < 0 :
            return False # :TODO: WHAT dO?


        return (self._grid[p] > 0.0)

    def is_occupied(self, point, safety_region=True):
        """Returns True if the given points is inside an obstacle."""
        for poly in self._obstacles:
            if poly.in_polygon(point, safety_region):
                return True
        return False


    def draw_discrete(self, axes, fill='Greens'):
        if not self._is_discretized:
            self.discretize_map()
        xvals = np.arange(0, self._dim[0], self._gridsize)
        yvals = np.arange(0, self._dim[1], self._gridsize)

        print(len(xvals), len(self._grid))


        axes.imshow(self._grid.T, origin='lower', cmap=plt.get_cmap(fill), alpha=0.7,
                   extent=(0, self._dim[0], 0, self._dim[1]))

    def draw(self, axes, pcolor='g', ecolor='k', draw_discrete=False):
        """Draws the map in the given matplotlib axes."""

        if draw_discrete:
            self.draw_discrete(axes)
            for poly in self._obstacles:
                poly.draw(axes, pcolor, ecolor, alph=1.)



        else:
            for poly in self._obstacles:
                poly.draw(axes, pcolor, ecolor, alph=1.0)




class Polygon(object):
    """Generalized polygon class."""

    def __init__(self, vertices, safety_region_length=1.0):
        """Initialize polygon with list of vertices."""
        self._V = np.array(vertices)


        if safety_region_length > 0.0:
            self._V_safe = self.extrude(safety_region_length)
            self._safety_region = True
        else:
            self._safety_region = False
            self._V_safe = self._V

    def __str__(self):
        """Return printable string of polygon. For debugging."""
        return str(self._V)

    def get_vertices(self, safe=False):
        if safe:
            return self._V_safe
        else:
            return self._V

    def set_safety_region(self, val):
        self._safety_region = val

    def in_polygon(self, point, safety_region=True):
        """Return True if point is in polygon."""

        if self._safety_region and safety_region:
            vertices = self._V_safe
        else:
            vertices = self._V

        (x,y) = point

        n = len(vertices)
        inside = False

        p1x,p1y = vertices[0]
        for i in range(n+1):
            p2x,p2y = vertices[i % n]
            if y > min(p1y,p2y):
                if y <= max(p1y,p2y):
                    if x <= max(p1x,p2x):
                        if p1y != p2y:
                            xinters = (y-p1y)*(p2x-p1x)/(p2y-p1y)+p1x
                        if p1x == p2x or x <= xinters:
                            inside = not inside
            p1x,p1y = p2x,p2y

        return inside

    def extrude(self, length):
        self._safety_region = False
        n = len(self._V)
        vectors = np.empty([n,2])
        angles  = np.empty([n,2])

        a = self._V[n-1] - self._V[0]
        b = self._V[1]   - self._V[0]
        a = a / np.linalg.norm(a, 2)
        b = b / np.linalg.norm(b, 2)

        vectors[0] = a + b
        angles[0]  = np.dot(a.transpose(), b)
        for ii in range(1, n-1):
            a = self._V[ii-1] - self._V[ii]
            b = self._V[ii+1] - self._V[ii]
            a = a / np.linalg.norm(a, 2)
            b = b / np.linalg.norm(b, 2)

            vectors[ii] = a + b
            angles[ii]  = np.dot(a.transpose(), b)

        a = self._V[n-2] - self._V[n-1]
        b = self._V[0] - self._V[n-1]
        a = a / np.linalg.norm(a, 2)
        b = b / np.linalg.norm(b, 2)

        vectors[n-1] = a + b
        angles[n-1]  = np.dot(a.transpose(), b)
        new_polygon = np.zeros([n,2])

        for ii in range(0,n):
            new_polygon[ii] = self._V[ii] - \
                              length / np.linalg.norm(vectors[ii]) * vectors[ii] * 1.4142 / np.sqrt(1 - angles[ii])
            if self.in_polygon(new_polygon[ii]):
                new_polygon[ii] = self._V[ii] + \
                                  length / np.linalg.norm(vectors[ii]) * vectors[ii] * 1.4142 / np.sqrt(1 - angles[ii])
        self._safety_region = True
        return new_polygon


    def get_edge_points(self, d):
        """Returns list of points along the edges of the polygon."""

        if self._safety_region:
            V = self._V_safe
        else:
            V = self._V
        n = len(V)

        linesample = np.transpose(np.array([np.linspace(V[n-1][0], V[0][0], d),
                                            np.linspace(V[n-1][1], V[0][1], d)]))
        points = linesample.tolist()
        for ii in range(0,n-1):
            linesample = np.transpose(np.array([np.linspace(V[ii][0], V[ii+1][0], d),
                                                np.linspace(V[ii][1], V[ii+1][1], d)]))
            points += linesample.tolist()

        return points

    def draw(self, axes, fcolor, ecolor, alph):
        """Plots the polygon with the given plotter."""
        poly = matplotlib.patches.Polygon(self._V, facecolor=fcolor, edgecolor=ecolor, alpha=alph)
        axes.add_patch(poly)
        if self._safety_region:
            poly_safe = matplotlib.patches.Polygon(self._V_safe, facecolor='none', edgecolor=fcolor)
            axes.add_patch(poly_safe)


if __name__=="__main__":
    amap = Map('minima', gridsize=1., safety_region_length=2.0)
    fig = plt.figure()
    ax  = fig.add_subplot(111, autoscale_on=True)

    #fig, ax = plt.subplots()

    amap.draw(ax, draw_discrete=False)

    ax.axis('scaled')
    ax.set_xlim(-10.,160.)
    ax.set_ylim(-10.,160.)
    #amap.draw_discrete(ax)

    #ax.xaxis.set_major_locator(plt.MultipleLocator(5.0))
    #ax.xaxis.set_minor_locator(plt.MultipleLocator(1.))
    #ax.yaxis.set_major_locator(plt.MultipleLocator(5.0))
    #ax.yaxis.set_minor_locator(plt.MultipleLocator(1.))
    #ax.grid(which='major', axis='x', linewidth=0.75, linestyle='-', color='1.')
    #ax.grid(which='minor', axis='x', linewidth=0.5, linestyle='-', color='0.7')
    #ax.grid(which='major', axis='y', linewidth=0.75, linestyle='-', color='1.')
    #ax.grid(which='minor', axis='y', linewidth=0.5, linestyle='-', color='0.7')
    #ax.set_xticklabels([])
    #ax.set_yticklabels([])
    plt.show()
