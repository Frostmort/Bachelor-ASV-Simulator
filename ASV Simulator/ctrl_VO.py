import copy
import time


import numpy as np

from vessel import Vessel
from utils import Controller, PriorityQueue

from matplotlib2tikz import save as tikz_save


class VO(Controller):
    def __init__(self, scanDistance=50):
        self.scanDistance = scanDistance
        self.tc = 0
        self.world = 0
        self.newVesselParams = [0, 0]
        self.totalTime = 0

    def update(self, vobj, world, vesselArray):
        self.world = world
        tic = time.process_time_ns()
        for v in vesselArray:
            if not v.is_main_vessel:
                scanData = self.scan(vobj, v)
                if scanData[0] <= self.scanDistance:
                    VOarray = self.createVO(vobj, v, scanData)
                    if VOarray[3] > VOarray[8] > VOarray[4]:
                        print("Collision imminent!")
                        self.newVesselParams = self.collisionAvoidance(vobj, v, scanData)
                        vobj.u_d = self.newVesselParams[0]
                        vobj.psi_d = self.newVesselParams[1]
                        self.totalTime = self.totalTime + (time.process_time_ns() - tic)
                        print(self.totalTime)


    def scan(self, vessel1, vessel2):
        xd = (vessel2.x[0] - vessel1.x[0])
        yd = (vessel2.x[1] - vessel1.x[1])
        distance = abs(np.sqrt(xd ** 2 + yd ** 2))
        angle = np.arctan2(yd, xd)

        return [distance, angle]

    # Creates the VO array for use in collision detection and
    # Array has following contents:
    # [0 crossing direction, 1 distance between ships, 2 angle between ships, 3 left collision cone edge,
    # 4 right collision cone edge, 5 velocity of A, 6 velocity of B, 7 relative velocity magnitude,
    # 8 relative velocity angle]
    def createVO(self, vessel1, vessel2, scanData):
        VO = [0, 0, 0, 0, 0, 0, 0, 0, 0]
        # find which side crossing vessel is coming from
        if vessel2.x[0] > vessel1.x[0] and (np.pi / 2 < vessel2.x[2] < 3 * np.pi / 2):
            VO[0] = 'r'
        elif vessel2.x[0] < vessel1.x[0] and (vessel2.x[2] < np.pi / 2 or vessel2.x[2] > 3 * np.pi / 2):
            VO[0] = 'l'
        else:
            VO[0] = 'n'

        # find left and right boundaries of collision cone
        VO[1] = scanData[0]
        VO[2] = scanData[1]
        angle = np.arctan2(scanData[0] / 2, scanData[0])

        VO[3] = VO[2] + np.arctan2((scanData[0] / 2) + 5, scanData[0])
        VO[4] = VO[2] - np.arctan2((scanData[0] / 2) + 5, scanData[0])

        # find vector (xab) and angle (lab) of relative velocity
        VO[5] = [np.cos(vessel1.x[2]), np.sin(vessel1.x[2])]
        VO[6] = [(np.cos(vessel2.x[2])), (np.sin(vessel2.x[2]))]
        VO[7] = [VO[5][0] - VO[6][0], VO[5][1] - VO[6][1]]
        VO[8] = np.arctan2(VO[7][1], VO[7][0])

        return VO



    def collisionAvoidance(self, v1, v2, scanData):

        xyc = [0, 0]
        self.tc = self.getCollisionTime(v1, v2, xyc)
        RV = self.getRV(v1)
        newParams = self.getRAV(v1, v2, RV, scanData)
        return newParams

    def getCollisionTime(self, v1, v2, xyc):
        r1 = ([v1.x[0], v1.x[3] * np.cos(v1.x[2]), v1.x[1], v1.x[3] * np.sin(v1.x[2])])  # vessel 1 velocity vector
        r2 = ([v2.x[0], v2.x[3] * np.cos(v2.x[2]), v2.x[1], v2.x[3] * np.sin(v2.x[2])])  # vessel 2 velocity vector

        tx = (r2[0] - r1[0]) / (r1[1] - r2[1]) # time to intersect in x
        ty = (r2[2] - r1[2]) / (r1[3] - r2[3]) # time to intersect in y

        return (tx + ty) / 2 # returns average of x and y times

    def getRV(self, v1):
        u_max = v1.model.est_u_max # max surge velocity
        u_min = v1.model.est_u_min # min surge velocity (reverse)
        r_max = v1.model.est_r_max # max yaw velocity

        du_max = v1.model.est_du_max # max surge acceleration
        du_min = v1.model.est_du_min # min surge acceleration (reverse)
        dr_max = v1.model.est_dr_max # max yaw acceleration

        t = 1

        rt = dr_max * t
        ut = du_max * t

        if rt > r_max:
            rt = r_max

        if ut > u_max:
            ut = u_max

        maxstraight = ut
        maxreverse = -ut
        maxstarboard = [ut * np.cos(rt), ut * np.sin(rt)]
        maxport = [-1 * maxstarboard[0], maxstarboard[1]]

        return [maxstraight, maxreverse, maxstarboard, maxport]

    def getRAV(self, v1, v2, RV, scanData):
        testVessel = Vessel(copy.deepcopy(v1.x), np.zeros((1, 6)), v1.h, v1.dT, v1.N, [], False, 'viknes')
        testVessel.world = copy.deepcopy(self.world)

        testVessel.x = copy.deepcopy(v1.x)

        print('test starboard')
        testVessel.x[3] = np.sqrt(RV[2][0]**2 + RV[2][1]**2)
        testVessel.x[2] = testVessel.x[2] - (np.pi/2 - np.arctan2(RV[2][1], RV[2][0]))
        testVO = self.createVO(testVessel, v2, scanData)
        if self.checkNewVO(testVO):
            newParams = [testVessel.x[3], testVessel.x[2]]
            return newParams

        print('test port')
        testVessel.x[3] = np.sqrt(RV[3][0]**2 + RV[3][1]**2)
        testVessel.x[2] = testVessel.x[2] - (np.pi/2 - np.arctan2(RV[3][1],RV[3][0]))
        testVO = self.createVO(testVessel, v2, scanData)
        if self.checkNewVO(testVO):
            newParams = [testVessel.x[3], testVessel.x[2]]
            return newParams

        print('test ahead')
        testVessel.x[3] = RV[0]
        testVessel.x[2] = v1.x[2]
        testVO = self.createVO(testVessel, v2, scanData)
        if self.checkNewVO(testVO):
            newParams = [testVessel.x[3], testVessel.x[2]]
            return newParams

        print('test reverse starboard')
        testVessel.x[3] = RV[1]
        testVessel.x[2] = testVessel.x[2] - (np.pi/2 - np.arctan2(RV[2][1], RV[2][0]))
        testVO = self.createVO(testVessel, v2, scanData)
        if self.checkNewVO(testVO):
            newParams = [testVessel.x[3], testVessel.x[2]]
            return newParams

        print('test reverse port')
        testVessel.x[3] = RV[1]
        testVessel.x[2] = testVessel.x[2] - (np.pi/2 - np.arctan2(RV[3][1],RV[3][0]))
        testVO = self.createVO(testVessel, v2, scanData)
        if self.checkNewVO(testVO):
            newParams = [testVessel.x[3], testVessel.x[2]]
            return newParams

        print('test reverse')
        testVessel.x[3] = RV[1]
        testVessel.x[2] = v1.x[2]
        testVO = self.createVO(testVessel, v2, scanData)
        if self.checkNewVO(testVO):
            newParams = [testVessel.x[3], testVessel.x[2]]
            return newParams

        print('No RAV found')
        return [0, v1.x[2]]

    def checkNewVO(self, VO):
        if not VO[3] > VO[8] > VO[4]:
            return True
        else:
            return False

    def checkLand(self, vessel1, params):
        vessel1.u_d = params[0]
        vessel1.psi_d = params[1]
        for x in range(self.world.n, round(self.world.n + (self.tc * 100))):
            vessel1.update_model(x)
            p0 = vessel1.model.x[0:2]
            if self.world._map.is_occupied(p0, safety_region=False):
                return True

        return False
