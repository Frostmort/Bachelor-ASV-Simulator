#!/usr/bin/env python
import numpy as np
import random

from utils import Controller


class Wafi(Controller):
    def __init__(self, R2=60, mode='wafi', seed=1337, turnfreq=30):
        self.cGoal = None  # Current Goal
        self.cWP = 0  # Used if waypoint-navigation
        self.nWP = 0
        self.is_initialized = False

        self.goaly = 0
        self.goalx = 0

        self.seed = seed
        self.iterator = 1
        self.rng = random
        self.R2 = R2
        self.mode = mode
        self.stepcounter = 0
        self.turnfreq = turnfreq

        self.wps = None

    def update(self, vobj, vesselArray):
        if not self.is_initialized:
            # Reference to the vessel object's waypoints
            self.wps = vobj.waypoints

            #initial setup for wafi, set seed and get starting goal
            if self.mode == 'wafi':
                self.rng.seed(self.seed)
                self.new_goal(vobj)
            self.is_initialized = True

        x = vobj.x[0]
        y = vobj.x[1]

        if self.mode == 'waypoint' or self.mode == 'pursuit':
            vobj.psi_d = np.arctan2(self.cGoal[1] - y,
                                    self.cGoal[0] - x)





        if self.mode == 'wafi':
            vobj.psi_d = np.arctan2(self.goalx -x,
                                    self.goaly - y)
            self.stepcounter = self.stepcounter + 1
#            print('step:',self.stepcounter)
            if self.stepcounter >= self.turnfreq:
                self.new_goal(vobj)
                self.stepcounter = 0



    def draw(self, axes, N, fcolor, ecolor):
        axes.plot(self.wps[:, 0], self.wps[:, 1], 'k--')

    def visualize(self, fig, axarr, t, n):
        if self.mode == 'goal-switcher' or self.mode == 'waypoint':
            axarr[0].plot(self.wps[:, 0], self.wps[:, 1], 'k--')
            axarr[0].plot(self.cGoal[0], self.cGoal[1], 'rx', ms=10)

    def random_waypoint(self):
        x = self.rng.randrange(-50, 50, 1)
        y = self.rng.randrange(-50, 50, 1)
        return (x, y)

    def new_goal(self,vobj):
        x1, y1 = self.random_waypoint()
        list = np.copy(vobj.x)
        x2, y2 = list[0:2]
        self.goalx= x1 + x2
        self.goaly= y1 + y2

        print('set new goal', self.goalx,',',self.goaly)


if __name__ == "__main__":
    vessel = Wafi()
    vessel.rng.seed(1337)
    for x in range(1, 11):
        x1 = vessel.random_waypoint(x1=(0, 0))
        print("new waypoints:", x1)
