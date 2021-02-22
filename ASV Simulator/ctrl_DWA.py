#!/usr/bin/env python

"""
Dynamic Window controller

This module implements the Dynamic Window controller as proposed by Fox et. al., 1997.

"""

import time, cProfile
from matplotlib import pyplot as plt
from matplotlib import cm

import numpy as np
from scipy.ndimage.filters import uniform_filter

import copy

from world import World
from map import Map, Polygon
from vessel import Vessel

from utils import *

class DynamicWindow(Controller):
    def __init__(self, dT, N, the_map):

        self.window_res = [5, 61]  # xy Dynamic Window resolution
        self.xStride = the_map.get_gridsize()
        self.yStride = the_map.get_gridsize()
        self.dT = dT
        self.N  = N
        self.n  = 0

        self.window          = np.zeros(self.window_res)
        self.velocity_map    = np.zeros(self.window_res)
        self.heading_map     = np.zeros(self.window_res)
        self.dist_map        = np.zeros(self.window_res)
        self.scaled_dist_map = np.zeros(self.window_res)

        self.final_heading = np.empty(self.window_res)
        
        self.u_range = None
        self.r_range = None

        self.current_arches  = [[None for x in xrange(self.window_res[1])] for x in xrange(self.window_res[0])]
        self.arches          = [None] * N
        self.best_arches     = [None] * N

        self.test_arch = True

        self.MAX_REVERSE_SP  = -99999

        # Temporary variables for plotting
        self.rk_best = 5
        self.uk_best = 5
        self.cur_uk = 5
        self.cur_rk = 8

        self.axarr = None

        self.win_radius = 40
        self.pred_horiz = 20
        self.time_step  = 5.
        
        self.alpha = 0.7
        self.beta  = 0.2
        self.gamma = 0.5

        self.sigma = 0.0 # Low pass filter constant

        self.u_max = 3.0
         
        # :todo: Some circular logic here. Need set this after World object is
        # created, which depends on Vessel and Controller objects.

        self.last_update = -self.dT
        self.psi_target = None



    def update(self, vobj):

        x = vobj.x[0]
        y = vobj.x[1]
        psi = vobj.x[2]
        u = vobj.x[3]  # body frame forward velocity
        r = vobj.x[5]

        self.psi_target = np.arctan2(vobj.current_goal[1] - y,
                                     vobj.current_goal[0] - x)
        tic = time.clock()


        #self.window.fill(0)
        self.velocity_map.fill(0)
        self.heading_map.fill(0)
        self.dist_map.fill(0)
        self.scaled_dist_map.fill(0)

        # Determine reachable surge velocities
        u_rad_max = min(u + vobj.model.est_du_max * self.time_step,
                        min(self.u_max, vobj.model.est_u_max))
        u_rad_min = max(u + vobj.model.est_du_min * self.time_step,
                        max(0, vobj.model.est_u_min))

        self.u_range = np.linspace(u_rad_max, u_rad_min, self.window_res[0])
        
        # Determine reachable yaw velocities
        r_rad_max = min(r + vobj.model.est_dr_max * self.time_step,
                        vobj.model.est_r_max)
        r_rad_min = max(r - vobj.model.est_dr_max * self.time_step,
                        -vobj.model.est_r_max)
        
        self.r_range = np.linspace(r_rad_max, r_rad_min, self.window_res[1])

        # print(vobj.model.est_dr_max, vobj.model.est_r_max, r)
        # print(r_rad_max, r_rad_min)
        # print(u_range)

        # Calculate distance map
        for uk in range(0, self.window_res[0]):
            u = self.u_range[uk]

            for rk in range(0, self.window_res[1]):
                r = self.r_range[rk]

                # Calculate distance map. The reachable points.
                self.calc_dist_map(uk, rk, x, y, psi, u, r, vobj)
                
                # Calculate the dynamic window
                self.calc_dyn_wind(uk, rk, x, y, psi, u, r, vobj)

        # Normalize
        heading_min = np.amin(self.heading_map)
        heading_max = np.amax(self.heading_map)
        if heading_min == heading_max:
            self.heading_map.fill(0)
        else:
            self.heading_map  = (self.heading_map - heading_min) / float(heading_max - heading_min)

        velocity_min = np.amin(self.velocity_map)
        velocity_max = np.amax(self.velocity_map)
        if velocity_min == velocity_max:
            self.velocity_map.fill(0)
        else:
            self.velocity_map  = (self.velocity_map - velocity_min) / float(velocity_max - velocity_min)
    

        dist_min = np.amin(self.scaled_dist_map)
        dist_max = np.amax(self.scaled_dist_map)
        if dist_min == dist_max:
            self.scaled_dist_map.fill(0)
        else:
            self.scaled_dist_map  = (self.scaled_dist_map - dist_min) / float(dist_max - dist_min)


        dist_min = np.amin(self.dist_map)
        dist_max = np.amax(self.dist_map)
        if dist_min == dist_max:
            self.dist_map.fill(0)
        else:
            self.dist_map  = (self.dist_map - dist_min) / float(dist_max - dist_min)


        
        # Compose window
        self.window = self.sigma * self.window + \
                      (1.-self.sigma) * (self.alpha*self.heading_map + \
                                        self.beta *self.scaled_dist_map + \
                                        self.gamma*self.velocity_map)

        #uniform_filter(self.window, size=5, output=self.window, mode='nearest')

        # Find the best option
        n = np.argmax(self.window)
        uk_best = int(n / self.window_res[1])
        rk_best = n % self.window_res[1]

        if self.window[uk_best, rk_best] <= 0:
            # No admissible choice. Break with full force.
            vobj.psi_d = np.Inf
            vobj.u_d = self.MAX_REVERSE_SP
            vobj.r_d = 0

            uk_best = self.cur_uk
            rk_best = self.cur_rk
            self.best_arches[self.n] = np.zeros((1,2))

        else:
            # Use best choice
            vobj.psi_d = np.Inf
            vobj.u_d = self.u_range[uk_best]
            vobj.r_d = self.r_range[rk_best]
            #print(vobj.u_d, vobj.r_d)
            self.uk_best = uk_best
            self.rk_best = rk_best
            self.best_arches[self.n] = copy.deepcopy(self.current_arches[self.uk_best][self.rk_best])

        # :todo: draw beautiful paths


        self.n += 1
        toc = time.clock()
        #print("Dynamic window: (%.2f, %.2f, %.2f) CPU time: %.3f" %(x,y,psi,toc-tic))


        #print(r_range, self.rk_best)
        
    def calc_dist_map(self, uk, rk, x, y, psi, u, r, vobj):
        if np.abs(u) < 0.01:
            # No disatance
            self.scaled_dist_map[uk, rk] = 0
            self.dist_map[uk, rk] = 0
            self.current_arches[uk][rk] = np.zeros((1,2))
            return

        if np.abs(r) > 0.0001:
            # The (u, r) circle is defined by:
            #    radius: u/r
            #    center: [x - (u/r)*sin(psi), y + (u/r)*cos(psi)]

            # Circle center
            center = np.array([x - (u/r)*np.sin(psi),
                               y + (u/r)*np.cos(psi)])

            # Angle from circle center to target
            beta = np.arctan2(y - center[1],
                              x - center[0])

            # Radius of turn
            radius = np.abs(u/r)

            # Size of steps taken along circle (in radians) when
            # checking for intersections.
            # Must be equal to the resolution of the world map.
            cstep = (r/np.abs(u)) * min(self.xStride, self.yStride)
            
            # Distance of each step
            step_dist = np.abs(cstep) * radius
            # Time along step
            step_time = step_dist / np.abs(u)

            # Find the last angle to test along the circle by
            # intersection the circle with a cicle in from of
            # our vehicle self.win_radius
            intersected, ints = int_circles(x + np.sign(u)*self.win_radius*np.cos(psi),
                                            y + np.sign(u)*self.win_radius*np.sin(psi),
                                            self.win_radius,
                                            center[0],
                                            center[1],
                                            radius)
            
            if not intersected:
                print("Error, error! Not two points in intersection")
                return

            # The intersection test should return two coordinates:
            # 1. The coordiante of our vehilce
            # 2. The coordinates of the limit of this trajectory
            # We use the manhattan distance to select the latter            
            if np.abs(ints[0,0]-x + ints[0,1]-y) < 0.001:
                coords = ints[1,:]
            else:
                coords = ints[0,:]
                
            # Find the angle of the given intersection.
            # It should be the last angle in the iteration below
            last_alpha = normalize_angle(np.arctan2(coords[1] - center[1],
                                                    coords[0] - center[0]),
                                         beta)
                    
            # Make sure last_alpha is "ahead" of beta in terms of cstep
            if cstep > 0 and last_alpha < beta:
                last_alpha += 2*np.pi
            elif cstep < 0 and last_alpha > beta:
                last_alpha -= 2*np.pi

            # Iterate along circle, testing for intersections
            alpha = beta
            max_dist = 0
            N = int(np.around(self.pred_horiz/step_time))+1
            path = np.empty((N, 2))
            it = 0
            for t in np.arange(step_time, self.pred_horiz, step_time):
                alpha += cstep
                xk = center[0] + radius*np.cos(alpha)
                yk = center[1] + radius*np.sin(alpha)
                
                # :todo: draw paths?


                if (cstep > 0 and alpha >= last_alpha) or \
                   (cstep < 0 and alpha <= last_alpha):
                    # Travelled full path
                    alpha = last_alpha
                    break

                elif vobj.world.is_occupied(xk, yk, t):
                    # Intersection
                    break

                path[it] = xk, yk
                it += 1

                max_dist += step_dist

            self.final_heading[uk, rk] = alpha
            self.current_arches[uk][rk] = path[:(it)]

            # Update distance map with max distance along this path
            # relatice to possible distance
            self.scaled_dist_map[uk, rk] = np.abs((alpha-beta)/(last_alpha-beta))
            self.dist_map[uk, rk] = max_dist
        else:
            # Travelling at a straight line (u, 0)
            # Check this line for intersections with world
            
            # Distance of each step
            step_dist = min(self.xStride, self.yStride)
            
            # Time along step
            step_time = step_dist / abs(u)
            
            # Distance we can travel along line
            max_dist = 0
            
            # :todo: visualization of this path
            
            # Iterate over line
            x_step = np.sign(u)*step_dist*np.cos(psi)
            y_step = np.sign(u)*step_dist*np.sin(psi)
            xk = x
            yk = y
            path = np.empty((int(np.around(self.pred_horiz/step_time))+1, 2))
            it = 0
            for t in np.arange(step_time, self.pred_horiz, step_time):
                xk += x_step
                yk += y_step
                
                # :todo: draw/store paths?

                if max_dist >= 2*self.win_radius:
                    # We are done
                    max_dist = 2*self.win_radius
                    break
                elif vobj.world.is_occupied(xk, yk, t):
                    # Found intersection
                    break
                    
                path[it] = xk, yk
                it += 1

                max_dist += step_dist

            self.final_heading[uk, rk] = psi
            self.current_arches[uk][rk] = path[:(it)]

            # :todo: Draw/update paths
            self.scaled_dist_map[uk, rk] = max_dist / (2*self.win_radius)
            self.dist_map[uk, rk] = max_dist
        
    def calc_dyn_wind(self, uk, rk, x, y, psi, u, r, vobj):
        dist = self.dist_map[uk, rk]

        # Only proceed if this is an admissible velocity, i.e.,
        # the vehicle can come to a complete stop after choosing
        # this alternative
        if np.abs(u) > np.sqrt(2*dist * vobj.model.est_du_max) or \
           np.abs(r) > np.sqrt(2*dist * vobj.model.est_dr_max):
            # Not admissible
            self.window[uk, rk] = 0
            self.heading_map[uk, rk] = 0
            self.velocity_map[uk, rk] = 0

        else:
            # The value of this sector in the dynamic window is
            # value = alpha*heading + beta*dist + gamma*velocity
            #
            # The psi value should be calculated as the obtained psi
            # when applying maximum deceleration to the yaw after the
            # next time step.
            psi_next   = psi + r*self.dT + 0.5*r*np.abs(r/vobj.model.est_dr_max)

            if np.abs(r) > 0.001:
                x1 = x + u/r * (np.sin(psi) - np.sin(psi + r*self.dT))
                y1 = y - u/r * (np.cos(psi) - np.cos(psi + r*self.dT))
            else:
                x1 = x + u * np.cos(psi) * self.dT
                y1 = y + u * np.sin(psi) * self.dT

            if vobj.psi_d is not np.inf:
                self.psi_target = vobj.psi_d
            else:
                self.psi_target = np.arctan2(vobj.current_goal[1] - y1,#self.current_arches[uk][rk][-1,1],
                                             vobj.current_goal[0] - x1)#self.current_arches[uk][rk][-1,0])
            vobj.psi_d = np.inf

            # If in reverse, heading is the opposite
            if u < 0:
                psi_next += np.pi

            while psi_next >= np.pi:
                psi_next -= 2*np.pi

            while psi_next < -np.pi:
                psi_next += 2*np.pi


            heading = np.pi - np.abs(psi_next - self.psi_target)

            self.velocity_map[uk, rk] = u
            self.heading_map[uk, rk] = heading
            
    def draw(self, axes, n, fcolor='y', ecolor='k'):
        return
        for ii in range(0, self.n, 8):
            axes.plot(self.best_arches[ii][:,0], self.best_arches[ii][:,1], 'r', alpha=0.5,
                      lw=2)

    def visualize(self, fig, axarr, t, n):
        if n == 0:
            # First run
            # self.axarr = fig.add_subplots(4)
            # self.axarr[0].contourf(self.r_range, self.u_range, self.window, cmap=plt.get_cmap('Greys'))
            # self.axarr[1].contourf(self.r_range, self.u_range, self.heading_map, cmap=plt.get_cmap('Greys'))
            # self.axarr[2].contourf(self.r_range, self.u_range, self.dist_map, cmap=plt.get_cmap('Greys'))
            # self.axarr[3].contourf(self.r_range, self.u_range, self.velocity_map, cmap=plt.get_cmap('Greys'))

            axarr[1].set_title("Combined Window")

            for ii in range(1,5):
                axarr[ii].set_xlabel("r")
                axarr[ii].set_ylabel("u")
                axarr[ii].set_xticklabels([])
                axarr[ii].set_yticklabels([])
                axarr[ii].set_zticklabels([])
            # axarr[2].set_xlabel("r")
            # axarr[2].set_ylabel("u")
            # axarr[3].set_xlabel("r")
            # axarr[3].set_ylabel("u")
            # axarr[4].set_xlabel("r")
            # axarr[4].set_ylabel("u")
            
            
            axarr[2].set_title("Heading Map")
            axarr[3].set_title("Distance Map")
            axarr[4].set_title("Velocity Map")



        X, Y = np.meshgrid(self.r_range, self.u_range)
        # Visualize windows
        for ii in range(1,5):
            del axarr[ii].collections[:]

        axarr[1].plot_surface(X, Y, self.window, rstride=1, cstride=4, cmap=plt.get_cmap("coolwarm"))
        axarr[2].plot_surface(X, Y, self.heading_map, rstride=1, cstride=4, cmap=plt.get_cmap("coolwarm"))
        axarr[3].plot_surface(X, Y, self.scaled_dist_map, rstride=1, cstride=4, cmap=plt.get_cmap("coolwarm"))
        axarr[4].plot_surface(X, Y, self.velocity_map, rstride=1, cstride=4, cmap=plt.get_cmap("coolwarm"))
        
        #axarr[1].plot(self.r_range[self.rk_best], self.u_range[self.uk_best], 'rx', ms=10)

        for rk in range(0, self.window_res[1]):
            if rk == self.rk_best:
                continue
            axarr[0].plot(self.current_arches[self.uk_best][rk][:,0],
                          self.current_arches[self.uk_best][rk][:,1], 'b', alpha=0.6)

        """Visualize current arch. For real-time plotting."""
        axarr[0].plot(self.best_arches[self.n-1][:,0],
                      self.best_arches[self.n-1][:,1], 'r', alpha=0.8, lw=3)

def test():
    the_map = Map()
    tic = time.clock()
    the_map.discretize_map()
    print(time.clock() - tic)
    #obstacle = Polygon([(40,15), (45,15), (45,20), (40,20)], safety_region_length=4.0)
    #the_map.add_obstacles([obstacle])

    tend = 10
    dT = 1
    h  = 0.05
    N  = int(tend/h)  + 1
    N2 = int(tend/dT) + 1

    x0 = np.array([10, 10, 0.0, 3.0, 0, 0])
    xg = np.array([50, 50, 0])

    myDynWnd = DynamicWindow(dT, N2)

    v = Vessel(x0, xg, h, dT, N, [myDynWnd], is_main_vessel=True, vesseltype='viknes')
    v.current_goal = np.array([50, 50])

    world = World([v], the_map)

    myDynWnd.the_world = world
    
    world.update_world(0,0)

    fig = plt.figure()
    ax  = fig.add_subplot(111, aspect='equal', autoscale_on=False,
                          xlim=(-10, 160), ylim=(-10, 160))
    
    world.visualize(ax, 0, 0)
    
    plt.show()


    


def simple_scenario_with_plot():
    the_map = Map('s1')
        
    #obstacle = Polygon([(30,15), (35,15), (35,20), (30,20)], safety_region_length=4.0)
    #the_map.add_obstacles([obstacle])

    tend = 50
    dT = 1
    h  = 0.05
    N  = int(tend/h) + 1
    N2 = int(tend/dT) + 1
    x0 = np.array([0, 0, 0.5, 3.0, 0, 0])
    xg = np.array([120, 120, 0])

    myDynWnd = DynamicWindow(dT, N2)

    v = Vessel(x0, xg, h, dT, N, [myDynWnd], is_main_vessel=True, vesseltype='viknes')
    v.current_goal = np.array([50, 50])

    world = World([v], the_map)

    myDynWnd.the_world = world

    n = 0
    for t in np.linspace(0, tend, N):
        world.update_world(t, n, dT)
        n += 1

    fig = plt.figure()
    ax  = fig.add_subplot(111, aspect='equal', autoscale_on=False,
                          xlim=(-10, 10), ylim=(-10, 10))

    #fig, ax = plt.subplots()
    ax.grid()

    world.draw(ax, N)
    print("N, N2, n: ", N, N2, n)
    myDynWnd.draw(ax, N2)


    plt.show()


if __name__ == "__main__":
    #simple_scenario_with_plot()
    test()
