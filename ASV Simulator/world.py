#!/usr/bin/env python

import numpy as np
import math

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle

from matplotlib2tikz import save as tikz_save

class World(object):
    def __init__(self, vessels, the_map):
        self._vessels = vessels

        for v in self._vessels:
            v.world = self

        self._map = the_map

        self._is_collided = False
        self.minDistance = math.sqrt(the_map._dim[0]**2 + the_map._dim[1]**2)

    def get_num_vessels(self):
        return len(self._vessels)

    def get_discrete_grid(self):
        return self._map.get_discrete_grid()

    def get_simulation_data(self, n):
        data = self._vessels[0].get_simulation_data(n)
        data.append(self.minDistance)
        return data

    def save_data(self, n, filename):
        self._vessels[0].save_data(n, filename)

    def update_world(self, t, n):
        for v in self._vessels:
            v.time = t
            v.update_model(n)
            if int(t*100)%int(v.dT*100) == 0:  # Always true with dT = 0.5
                v.update_controllers(vesselArray = self._vessels)
        self.minDistance = self.getMinDistance()

    def getMinDistance(self):
        v1 = self._vessels[0]
        v2 = self._vessels[1]
        xd = v2.x[0] - v1.x[0]
        yd = v2.x[1] - v1.x[1]
        d = math.sqrt(xd**2 + yd**2)
        if d < self.minDistance:
            return d
        else:
            return self.minDistance

    def is_occupied_list(self, lst, tlst):
        for ii in range(0,len(lst)):
            if self.is_occupied(lst[ii][0],lst[ii][1],tlst[ii]):
                return True
        return False

    def is_occupied(self, x, y, t=0., R2=100.):
        """Is the point (x,y) occupied at time t?"""
        if self._map.is_occupied_discrete((x,y)):
            return True

        # Check for collision with other vessels
        for ii in range(1, len(self._vessels)):
            xi  = self._vessels[ii].x[0:2]
            psi = self._vessels[ii].x[2]
            u   = self._vessels[ii].x[3]
            v   = self._vessels[ii].x[4]

            # Predict vessel motion
            xnext = xi[0] + ( np.cos(psi)*u ) * t
            ynext = xi[1] + ( np.sin(psi)*u ) * t

            if (x - xnext)**2 + (y - ynext)**2 < R2:
                # Collision
                return True

        return False


    def collision_detection(self):
        p0 = self._vessels[0].model.x[0:2]

        # Have we crashed with land?
        if self._map.is_occupied(p0, safety_region=False):
            self._is_collided = True
            return True

        # Check for collision with other vessels
        for ii in range(1, len(self._vessels)):
            vi = self._vessels[ii].model.x[0:2]
            if (p0[0] - vi[0])**2 + (p0[1] - vi[1])**2 < 50:
                # Collision
                self._is_collided = True
                return True

        # No collision detected
        return False

    def reached_goal(self, R2=50):
        # :Todo: Assuming vessel0 is main vessel, valid?
        p0 = self._vessels[0].model.x[0:2]
        g  = self._vessels[0].goal[0:2]
        return (p0[0] - g[0])**2 + (p0[1] - g[1])**2 < R2


    def draw(self, axes, n):
        for v in self._vessels:
            v.draw(axes, n)

        if self._is_collided:
            self._vessels[0].draw_collision(axes, n)

        self._map.draw(axes, draw_discrete=False)

    def animate(self, fig, ax, n):

        self._map.draw(ax)
        patchlist = []
        shapelist = []

        def init():
            for v in self._vessels:

                if v.is_main_vessel:
                    p = plt.Polygon(v.get_shape(), fc='b', ec='k')
                    v.draw_waypoints(ax, n, 'b')
                else:
                    p = plt.Polygon(v.get_shape(), fc='y', ec='k')
                    v.draw_waypoints(ax, n, 'y')

                patchlist.append(p)
                shapelist.append(p.get_xy().transpose())
                ax.add_patch(p)

            return patchlist

        def update_patches(num):
            for ii in range(0, len(patchlist)):
                p  = patchlist[ii]
                x  = self._vessels[ii].path[num]

                Rz = np.array([[np.cos(x[2]), -np.sin(x[2])],
                               [np.sin(x[2]),  np.cos(x[2])]])
                newp = np.dot(Rz, shapelist[ii]).transpose()
                newp += x[0:2]

                p.set_xy(newp)
            return patchlist

        ani = animation.FuncAnimation(fig, update_patches, range(0, n, 2),
                                      init_func=init,
                                      interval=30,
                                      blit=False)
        return ani

    def visualize(self, fig, axarr, t, n):

        if int(t*100)%int(4*0.5*100) == 0:
            del axarr[0].collections[:]
            del axarr[0].lines[:]
            del axarr[0].patches[:]
            del axarr[0].texts[:]

            self._map.draw(axarr[0])
            for v in self._vessels:
                v.visualize(fig, axarr, t, n)

            plt.draw()
            plt.pause(0.001)

            #print(self.get_simulation_data(n))

            cmd = raw_input('Iteration: %d. Hit ENTER to continue... (s for save)'%n)
            if cmd == 's':
                fig.savefig('sim-step-t-' + str(t) + '.pdf', dpi=600, format='pdf', bbox_inches='tight')
    def save(self, filename):
        """Save simulation to file."""
        # :TODO: implement
        pass
