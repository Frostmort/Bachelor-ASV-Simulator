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
        self.VOarray = []
        self.collision = False

    def update(self, vobj, vesselArray):

        for v in vesselArray:
            if not v.is_main_vessel:
                self.scan(vobj, v)

        if self.collision:
            pass



    def scan(self, vessel1, vessel2):
        xd = (vessel2.x[0] - vessel1.x[0])
        yd = (vessel2.x[1] - vessel1.x[1])
        distance = abs(math.sqrt(xd**2 + yd**2))
        angle = np.arctan2(yd, xd)

        if distance <= self.scanDistance:
            self.createVO(vessel1, vessel2, distance, angle)

    def createVO(self, vessel1, vessel2, distance, angle):
        VOarray = []

        # find which side crossing vessel is coming from
        # if vessel2.x[0] > vessel1.x[0] and (np.pi / 2 < vessel2.x[2] < 3 * np.pi / 2):
        #     VOarray[0] = 'r'
        # elif vessel2.x[0] < vessel1.x[0] and (vessel2.x[2] < np.pi/2 or vessel2.x[2] > 3*np.pi/2):
        #     VOarray[0] = 'l'
        # else:
        #     VOarray[0] = 'n'

        # find left and right boundaries of collision cone
        l = distance
        langle = angle
        ll = langle + np.arctan2(distance/2, distance)
        lr = langle - np.arctan2(distance/2, distance)

        # find vector (xab) and angle (lab) of relative velocity
        xa = [vessel1.x[3] * np.cos(vessel1.x[2]), vessel1.x[3] * np.sin(vessel1.x[2])]
        xb = [-(vessel2.x[3] * np.cos(vessel2.x[2])), -(vessel2.x[3] * np.sin(vessel2.x[2]))]
        xab = [xa[0] + xb[0], xa[1] + xb[1]]
        lab = np.arctan2(xab[1], xab[0])

        if ll > lab > lr:
            self.collision = True
            print("Collision imminent!")
