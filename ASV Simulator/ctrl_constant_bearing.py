#!/usr/bin/env python
import numpy as np

from utils import Controller

class ConstantBearing(Controller):
    def __init__(self, target):
        self.target = target

        self.u_a_max = 2.0
        self.delta_p2 = 1.0

        # For plotting purposes
        self.v_d  = np.empty(2)
        self.vobj = None
        self.initialzed = False

    def update(self, vobj):
        if not self.initialzed:
            self.initialzed = True
            self.vobj = vobj

        # According to (Fossen, 2011) page 244.
        p_err = vobj.x[0:2] - self.target[0:2]
        v_approach = - self.u_a_max * p_err / np.sqrt( np.dot(p_err.T, p_err) + self.delta_p2)

        self.v_d = self.target[3:5] + v_approach

        vobj.u_d = self.v_d[0]
        vobj.psi_d = np.arctan2(self.v_d[1], self.v_d[0])

    def visualize(self, fig, axarr, t, n):
        return
        # Visualize predicted collision point
        def perp(a):
            b = np.empty_like(a)
            b[0] = -a[1]
            b[1] = a[0]
            return b

        def seg_intersect( a1, a2, b1,b2 ):
            da = a2-a1
            db = b2-b1
            dp = a1-b1
            dap = perp(da)
            denom = np.dot(dap, db)
            if denom == 0.0:
                return np.zeros(2)

            num = np.dot(dap, dp)
            return (num / denom ) * db + b1

        # Target vector
        psi = self.target[2]
        tv1 = self.target[0:2]
        tv2 = self.target[0:2] + np.array([np.cos(psi)*self.target[3],
                                           np.sin(psi)*self.target[3]])
        # Vessel vector
        psi = self.vobj.psi_d
        v1 = self.vobj.x[0:2]
        v2 = self.vobj.x[0:2] + np.array([np.cos(psi)*self.vobj.u_d,
                                           np.sin(psi)*self.vobj.u_d])
        i = seg_intersect(v1,v2,tv1,tv2)

        axarr[0].plot(i[0],i[1], 'rx', ms=5)

