#!/usr/bin/env python
import numpy as np
import random

from utils import Controller

class Wafi(Controller):
    def __init__(self, R2=60, mode='wafi', seed=1337):
        self.cGoal = None # Current Goal
        self.cWP   = 0    # Used if waypoint-navigation
        self.nWP   = 0
        self.is_initialized = False

        self.iterator = 1
        self.rng = random
        self.R2   = R2
        self.mode = mode

        self.wps  = None

    def update(self, vobj):
        if not self.is_initialized:
            # Reference to the vessel object's waypoints
            self.wps = vobj.waypoints
            if self.mode == 'waypoint':
                self.cWP   = 0
                self.cGoal = self.wps[self.cWP]
                self.nWP   = len(self.wps)
                vobj.u_d   = 3.0
                vobj.current_goal = np.copy(self.cGoal)

            elif self.mode == 'goal-switcher':
                self.cWP = 0
                self.cGoal = self.wps[self.cWP]
                vobj.current_goal = np.copy(self.cGoal)
                self.nWP = len(self.wps)
                self.iterator = int(self.nWP/20) + 1

            elif self.mode == 'wafi':
                self.cWP   = 0
                self.cGoal = self.wps[self.cWP]
                self.nWP   = len(self.wps)
                vobj.u_d   = 3.0
                vobj.current_goal = np.copy(self.cGoal)

            self.is_initialized = True


        x = vobj.x[0]
        y = vobj.x[1]

        if not self.mode == 'pursuit':
            while (x - self.cGoal[0])**2 + (y - self.cGoal[1])**2 < self.R2:
                if self.cWP < self.nWP - self.iterator:
                    self.cWP += self.iterator
                    self.cGoal = self.wps[self.cWP]
                else:
                    vobj.u_d = 0.0
                    break
            while vobj.world.is_occupied(self.cGoal[0], self.cGoal[1], t=0.1, R2=250.):
                if self.cWP < self.nWP - self.iterator:
                    self.cWP += self.iterator
                    self.cGoal = self.wps[self.cWP]
                else:
                    # Final waypoint is infeasible. What do we do now?
                    vobj.u_d = 0.0
                    print("Final waypoint infeasible. Stopping...")
                    break

            vobj.current_goal = np.copy(self.cGoal)

        if self.mode == 'waypoint' or self.mode == 'pursuit':
            vobj.psi_d = np.arctan2(self.cGoal[1] - y,
                                    self.cGoal[0] - x)

    def draw(self, axes, N, fcolor, ecolor):
        axes.plot(self.wps[:,0], self.wps[:,1], 'k--')

    def visualize(self, fig, axarr, t, n):
        if self.mode == 'goal-switcher' or self.mode == 'waypoint':
            axarr[0].plot(self.wps[:,0], self.wps[:,1], 'k--')
            axarr[0].plot(self.cGoal[0], self.cGoal[1], 'rx', ms=10)

    def random_waypoint(self, x1):
        x=self.rng.randrange(-10,10,1)
        y=self.rng.randrange(-10,10,1)

        return (x,y)

if __name__ == "__main__":
    vessel = Wafi()
    vessel.rng.seed(1337)
    for x in range (1,11):
        x1=vessel.random_waypoint(x1=(0,0))
        print("new waypoints:",x1)