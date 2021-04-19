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
    def __init__(self, scanDistance = 20):
        self.scanDistance = scanDistance
        self.VOarray
        self.collision = False
        pass

    def update(self, vobj, vessels):

        for v in vessels:
            if not v.isMainVessel:
                self.scan(vobj, v)

        if self.collsion:



    def scan(self, vessel1, vessel2):
        xd = abs(vessel1.x[0] - vessel2.x[0])
        yd = abs(vessel1.x[1] - vessel2.x[1])
        distance = math.sqrt(xd**2 + yd**2)

        if distance <= self.scanDistance:

        pass

    def createVO(self):

        pass