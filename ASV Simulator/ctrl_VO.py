import sys, time
import heapq

import numpy as np
import math as math
import matplotlib.pyplot as plt

from map import Map
from vessel import Vessel
from utils import Controller, PriorityQueue

from matplotlib2tikz import save as tikz_save


class VO(Controller):
    def __init__(self, scanDistance = 50):
        self.scanDistance = scanDistance
        self.VOarray = [0, 0, 0, 0, 0, 0, 0, 0, 0]
        self.aboutToCollide = False

    def update(self, vobj, vesselArray):

        for v in vesselArray:
            if not v.is_main_vessel:
                self.scan(vobj, v)

        if self.aboutToCollide:
            self.dont(vobj)



    def scan(self, vessel1, vessel2):
        xd = (vessel2.x[0] - vessel1.x[0])
        yd = (vessel2.x[1] - vessel1.x[1])
        distance = abs(math.sqrt(xd**2 + yd**2))
        angle = np.arctan2(yd, xd)

        if distance <= self.scanDistance:
            self.createVO(vessel1, vessel2, distance, angle)

    # Creates the VO array for use in collision detection and
    # Array has following contents:
    # [0 crossing distance, 1 distance between ships, 2 angle between ships, 3 left collision cone edge,
    # 4 right collision cone edge, 5 velocity of A, 6 velocity of B, 7 relative velocity magnitude,
    # 8 relative velocity angle]
    def createVO(self, vessel1, vessel2, distance, angle):

         #find which side crossing vessel is coming from
        if vessel2.x[0] > vessel1.x[0] and (np.pi / 2 < vessel2.x[2] < 3 * np.pi / 2):
            self.VOarray[0] = 'r'
        elif vessel2.x[0] < vessel1.x[0] and (vessel2.x[2] < np.pi/2 or vessel2.x[2] > 3*np.pi/2):
            self.VOarray[0] = 'l'
        else:
            self.VOarray[0] = 'n'

        # find left and right boundaries of collision cone
        self.VOarray[1] = distance
        self.VOarray[2] = angle
        self.VOarray[3] = self.VOarray[2] + np.arctan2(distance/2, distance)
        self.VOarray[4] = self.VOarray[2] - np.arctan2(distance/2, distance)

        # find vector (xab) and angle (lab) of relative velocity
        self.VOarray[5] = [vessel1.x[3] * np.cos(vessel1.x[2]), vessel1.x[3] * np.sin(vessel1.x[2])]
        self.VOarray[6] = [-(vessel2.x[3] * np.cos(vessel2.x[2])), -(vessel2.x[3] * np.sin(vessel2.x[2]))]
        self.VOarray[7] = [self.VOarray[5][0] + self.VOarray[6][0], self.VOarray[5][1] + self.VOarray[6][1]]
        self.VOarray[8] = np.arctan2(self.VOarray[7][1], self.VOarray[7][0])

        if self.VOarray[3] > self.VOarray[8] > self.VOarray[4]:
            self.aboutToCollide = True
            print("Collision imminent!")
        else:
            self.aboutToCollide = False

    def dont(self, vobj):

        vobj.psi_d = 0