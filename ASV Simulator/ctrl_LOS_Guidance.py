#!/usr/bin/env python
import numpy as np

from matplotlib.patches import Circle

from utils import Controller


class LOSGuidance(Controller):
    """This class implements  """
    def __init__(self, R2=20**2, u_d = 3.0, switch_criterion='circle'):
        self.R2 = R2 # Radii of acceptance (squared)
        self.R  = np.sqrt(R2)
        self.de = 20 # Lookahead distance

        self.cWP = 0 # Current waypoint
        self.wp = None

        self.wp_initialized = False

        if switch_criterion == 'circle':
            self.switching_criterion = self.circle_of_acceptance
        elif switch_criterion == 'progress':
            self.switching_criterion = self.progress_along_path

        self.Xp = 0.0
        self.u_d = u_d

    def circle_of_acceptance(self, x, y):
        return \
            (x - self.wp[self.cWP+1][0])**2 + \
            (y - self.wp[self.cWP+1][1])**2 < self.R2

    def progress_along_path(self, x, y):
        return \
            np.abs((self.wp[self.cWP+1][0] - x)*np.cos(self.Xp) + \
                   (self.wp[self.cWP+1][1] - y)*np.sin(self.Xp)) < self.R

    def update(self, vobj, vesselArray):
        if not self.wp_initialized:
            self.wp = None
            self.cWP = 0
            if vobj.waypoints.any():
                self.wp = vobj.waypoints
                self.nWP = len(self.wp[:,0])

                if self.nWP < 2:
                    print("Error! There must be more than 1 waypoint in the list!")
                    self.wp_initialized = False
                else:
                    self.wp_initialized = True
                    self.Xp = np.arctan2(self.wp[self.cWP + 1][1] - self.wp[self.cWP][1],
                                         self.wp[self.cWP + 1][0] - self.wp[self.cWP][0])
                    vobj.current_goal = self.wp[self.cWP + 1]


        x = vobj.x[0]
        y = vobj.x[1]

        if self.switching_criterion(x, y):
            while self.switching_criterion(x,y):
                if self.cWP < self.nWP - 2:
                # There are still waypoints left
                    print("Waypoint %d: (%.2f, %.2f) reached!" % (self.cWP,
                                                                  self.wp[self.cWP][0],
                                                                  self.wp[self.cWP][1]))
                    # print"Next waypoint: (%.2f, %.2f)" % (self.wp[self.cWP+1][0],
                    #                                        self.wp[self.cWP+1][1])
                    self.cWP += 1
                    vobj.current_goal = self.wp[self.cWP + 1]
                    self.Xp = np.arctan2(self.wp[self.cWP + 1][1] - self.wp[self.cWP][1],
                                         self.wp[self.cWP + 1][0] - self.wp[self.cWP][0])
                else:
                    # Last waypoint reached

                    if self.R2 < 50000:
                        print("Waypoint %d: (%.2f, %.2f) reached!" % (self.cWP,
                                                                      self.wp[self.cWP][0],
                                                                      self.wp[self.cWP][1]))
                        print("Last Waypoint reached!")
                        vobj.u_d = 0.0
                        self.R2 = np.Inf
                    return


        xk = self.wp[self.cWP][0]
        yk = self.wp[self.cWP][1]

        # Cross track error from Eq. (10.10), [Fossen, 2011]
        e  = -(x - xk)*np.sin(self.Xp) + (y - yk)*np.cos(self.Xp)

        Xr = np.arctan2( -e, self.de)
        psi_d = self.Xp + Xr

        vobj.psi_d = psi_d
        vobj.u_d   = self.u_d

    def visualize(self, fig, axarr, t, n):
        axarr[0].plot(self.wp[:,0], self.wp[:,1], 'k--')
        axarr[0].plot(self.wp[self.cWP+1,0], self.wp[self.cWP+1,1], 'rx', ms=10)

    def draw(self, axes, N, wpcolor='y', ecolor='k'):

        axes.plot(self.wp[:,0], self.wp[:,1], wpcolor+'--')
        return
        #ii = 0
        for wp in self.wp[1:]:
            circle = Circle((wp[0], wp[1]), 10, facecolor=wpcolor, alpha=0.3, edgecolor='k')
            axes.add_patch(circle)
            #axes.annotate(str(ii), xy=(wp[0], wp[1]), xytext=(wp[0]+5, wp[1]-5))
            #ii += 1
