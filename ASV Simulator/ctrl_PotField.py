#!/usr/bin/env python

"""
Potential Fields Controller.

"""

import time
from matplotlib import pyplot as plt

import numpy as np

from world import World
from map import Map, Polygon
from vessel import Vessel

from utils import *

class PotentialFields(Controller):
    def __init__(self, the_map, N):
        self.goal   = None

        self.mu = 250.
        self.d_max = 60
        self.k = 40.
        self.xi = 0.4
        self.u_max = 3.0
        self.u_min = 2.0
        self.map_res = the_map.get_gridsize()
        self.x_stride = 1./self.map_res
        self.y_stride = 1./self.map_res

        self.Fo = np.array([0,0])
        self.Fa = np.array([0,0])
        self.F  = np.array([0,0])

        self.Fn = np.zeros([N,3,2])

        self.xy = np.zeros([N,2])
        self.n = 0

    def update(self, vobj):

        self.goal = vobj.current_goal


        self.Fo = np.zeros(2)
        self.Fa = np.zeros(2)
        self.F  = np.zeros(2)

        aligned_x = ( np.floor(vobj.x[0] * self.map_res) + 0.5) / self.map_res
        aligned_y = ( np.floor(vobj.x[1] * self.map_res) + 0.5) / self.map_res

        xmin = aligned_x - self.d_max
        ymin = aligned_y - self.d_max
        xmax = aligned_x + self.d_max
        ymax = aligned_y + self.d_max

        # Calculate virtual force
        self.get_virtual_force(vobj.x, xmin, ymin, xmax, ymax, vobj)

        # Apply force to boat
        self.F = self.Fo + self.Fa
        vobj.psi_d = np.arctan2(self.F[1], self.F[0])
        #print("psi_d:", vobj.psi_d)

        cos_psi = np.cos(vobj.x[2])
        sin_psi = np.sin(vobj.x[2])

        vobj.u_d = max(self.u_min,
                       min(self.u_max,
                           np.sqrt((self.F[0]*cos_psi**2 + self.F[1]*cos_psi*sin_psi)**2 + \
                                   (self.F[1]*sin_psi**2 + self.F[0]*cos_psi*sin_psi)**2) ))


        self.Fn[self.n,0] = np.copy(self.F)
        self.Fn[self.n,1] = np.copy(self.Fo)
        self.Fn[self.n,2] = np.copy(self.Fa)
        self.xy[self.n] = np.copy(vobj.x[0:2])
        self.n += 1

    def get_virtual_force(self, xvec, xmin, ymin, xmax, ymax, vobj):

        d_max_sq = self.d_max**2

        for x in np.arange(xmin, xmax, self.x_stride):
            for y in np.arange(ymin, ymax, self.y_stride):

                if vobj.world.is_occupied(x, y):
                    rho_sq = (x - xvec[0])**2 + (y - xvec[1])**2

                    if rho_sq < d_max_sq:
                        # Generate repulsive force from this tile
                        rho = np.sqrt(rho_sq)
                        self.Fo = self.Fo + \
                                  self.mu*(1/rho - 1/self.d_max) / np.power(rho,3) \
                                  * np.array([xvec[0] - x,
                                              xvec[1] - y])


        #self.Fo = self.Fo/np.linalg.norm(self.Fo)

        cos_psi = np.cos(xvec[2])
        sin_psi = np.sin(xvec[2])

        dx = xvec[3]*cos_psi - xvec[4]*sin_psi
        dy = xvec[3]*sin_psi + xvec[4]*cos_psi

        # e = x - xd
        e = np.array([xvec[0] - self.goal[0],
                      xvec[1] - self.goal[1]])


        enorm = np.linalg.norm(e)

        if (self.k/self.xi)*enorm <= self.u_max:
            self.Fa = - self.xi * np.array([dx, dy]) - self.k*e
        else:
            self.Fa = - self.xi * np.array([dx, dy]) - (self.u_max/enorm)*e

    def visualize(self, fig, axarr, t, n):
        axarr[0].arrow(self.xy[self.n-1,0],
                       self.xy[self.n-1,1],
                       self.F[0]*5,
                       self.F[1]*5, width=0.1, fc='b', ec='b')#, alpha=0.5)
        axarr[0].arrow(self.xy[self.n-1,0],
                       self.xy[self.n-1,1],
                       self.Fo[0]*5,
                       self.Fo[1]*5, width=0.1, fc='r', ec='r', alpha=0.5)
        axarr[0].arrow(self.xy[self.n-1,0],
                       self.xy[self.n-1,1],
                       self.Fa[0]*5,
                       self.Fa[1]*5, width=0.1, fc='g', ec='g', alpha=0.5)

    def draw(self, axes, N, fcolor, ecolor):
        for ii in range(0, self.n, 16):
            axes.arrow(self.xy[ii,0],
                       self.xy[ii,1],
                       self.Fn[ii,0, 0]*5,
                       self.Fn[ii,0, 1]*5, width=0.1, fc='b', ec='b')#, alpha=0.5)
            axes.arrow(self.xy[ii,0],
                       self.xy[ii,1],
                       self.Fn[ii,1, 0]*5,
                       self.Fn[ii,1, 1]*5, width=0.1, fc='r', ec='r')#, alpha=0.5)
            axes.arrow(self.xy[ii,0],
                       self.xy[ii,1],
                       self.Fn[ii,2, 0]*5,
                       self.Fn[ii,2, 1]*5, width=0.1, fc='g', ec='g')#, alpha=0.5)


if __name__ == "__main__":
    a_map = Map('s1', gridsize=0.5)


    tend = 1.0
    dT   = 0.5
    h    = 0.05
    N    = int(np.around(tend/h)) + 1
    N2   = int(np.around(tend/dT)) + 1
    x0   = np.array([5,5,0, 2.0,0,0])
    xg   = np.array([120, 120, 0])

    potfield = PotentialFields(a_map, N2)

    vobj = Vessel(x0, xg, h, dT, N, [potfield], is_main_vessel=True, vesseltype='viknes')

    potfield.update(vobj)

    fig = plt.figure(1)
    ax  = fig.add_subplot(111, autoscale_on=False)
    ax.axis('scaled')
    ax.set_xlim((-10, 160))
    ax.set_ylim((-10, 160))
    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')
    ax.grid()

    a_map.draw(ax)

    potfield.draw(ax)

    plt.show()


