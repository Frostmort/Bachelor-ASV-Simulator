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
    def __init__(self, vesselArray=[], scanDistance=50):
        self.scanDistance = scanDistance
        self.VOarray = [0, 0, 0, 0, 0, 0, 0, 0, 0]
        self.collision = False

    def update(self, vobj, vesselArray):

        for v in vesselArray:
            if not v.is_main_vessel:
                self.scan(vobj, v)
                if self.collision:
                    self.collisionAvoidance(vobj, v)

    def scan(self, vessel1, vessel2):
        xd = (vessel2.x[0] - vessel1.x[0])
        yd = (vessel2.x[1] - vessel1.x[1])
        distance = abs(math.sqrt(xd ** 2 + yd ** 2))
        angle = np.arctan2(yd, xd)

        if distance <= self.scanDistance:
            self.createVO(vessel1, vessel2, distance, angle)

    # Creates the VO array for use in collision detection and
    # Array has following contents:
    # [0 crossing direction, 1 distance between ships, 2 angle between ships, 3 left collision cone edge,
    # 4 right collision cone edge, 5 velocity of A, 6 velocity of B, 7 relative velocity magnitude,
    # 8 relative velocity angle]
    def createVO(self, vessel1, vessel2, distance, angle):

        # find which side crossing vessel is coming from
        if vessel2.x[0] > vessel1.x[0] and (np.pi / 2 < vessel2.x[2] < 3 * np.pi / 2):
            self.VOarray[0] = 'r'
        elif vessel2.x[0] < vessel1.x[0] and (vessel2.x[2] < np.pi / 2 or vessel2.x[2] > 3 * np.pi / 2):
            self.VOarray[0] = 'l'
        else:
            self.VOarray[0] = 'n'

        # find left and right boundaries of collision cone
        self.VOarray[1] = distance
        self.VOarray[2] = angle
        self.VOarray[3] = self.VOarray[2] + np.arctan2(distance / 2, distance)
        self.VOarray[4] = self.VOarray[2] - np.arctan2(distance / 2, distance)

        # find vector (xab) and angle (lab) of relative velocity
        self.VOarray[5] = [vessel1.x[3] * np.cos(vessel1.x[2]), vessel1.x[3] * np.sin(vessel1.x[2])]
        self.VOarray[6] = [-(vessel2.x[3] * np.cos(vessel2.x[2])), -(vessel2.x[3] * np.sin(vessel2.x[2]))]
        self.VOarray[7] = [self.VOarray[5][0] + self.VOarray[6][0], self.VOarray[5][1] + self.VOarray[6][1]]
        self.VOarray[8] = np.arctan2(self.VOarray[7][1], self.VOarray[7][0])

        if self.VOarray[3] > self.VOarray[8] > self.VOarray[4]:
            self.collision = True
            print("Collision imminent!")
        else:
            self.collision = False

    def collisionAvoidance(self, v1, v2):

        # xyc = self.getCollisionPoint(v1, v2)
        xyc = [0, 0]
        tc = self.getCollisionTime(v1, v2, xyc)
        RV = self.getRV(v1,tc)
        RAV = self.getRAV(v1, RV)

    def getCollisionPoint(self, v1, v2):
        xc = ((v2.x[1] - v1.x[1]) - (v2.x[0] * np.tan(v2.x[2]) - v1.x[0] * np.tan[v1.x[2]])) \
             / (np.tan(v1.x[2]) - np.tan(v2.x[2]))
        yc = ((v2.x[0] - v1.x[0]) - (v2.x[1] * (1 / np.tan(v2.x[2])) - v1.x[1] * (1 / np.tan(v1.x[2])))) \
             / ((1 / np.tan(v1.x[2])) - (1 / np.tan(v2.x[2])))

        return [xc, yc]

    def getCollisionTime(self, v1, v2, xyc):
        r1 = ([v1.x[0], v1.x[3] * np.cos(v1.x[2]), v1.x[1], v1.x[3] * np.sin(v1.x[2])])  # vessel 1 velocity vector
        r2 = ([v2.x[0], v2.x[3] * np.cos(v2.x[2]), v2.x[1], v2.x[3] * np.sin(v2.x[2])])  # vessel 2 velocity vector

        tx = (r2[0] - r1[0]) / (r1[1] - r2[1]) # time to intersect in x
        ty = (r2[2] - r1[2]) / (r1[3] - r2[3]) # time to intersect in y

        return (tx + ty) / 2 # returns average of x and y times

    def getRV(self, v1, tc):
        u_max = v1.model.est_u_max # max surge velocity
        u_min = v1.model.est_u_min # min surge velocity (reverse)
        r_max = v1.model.est_r_max # max yaw velocity

        du_max = v1.model.est_du_max # max surge acceleration
        du_min = v1.model.est_du_min # min surge acceleration (reverse)
        dr_max = v1.model.est_dr_max # max yaw acceleration

        t = tc/4

        rt = dr_max * t
        ut = du_max

        if rt > r_max:
            rt = r_max

        if ut > u_max:
            ut = u_max

        maxstraight = [ut * np.cos(np.pi/2), ut * np.sin(np.pi/2)]
        maxreverse = -1 * maxstraight
        maxstarboard = [ut * np.cos(rt), ut * np.sin(rt)]
        maxport = -1 * maxstarboard

        return [maxstraight, maxreverse, maxstarboard, maxport]

    def getRAV(self, v1, RV):
        pass