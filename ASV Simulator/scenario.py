#!/usr/bin/env python

import numpy as np

import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib import gridspec
from mpl_toolkits.mplot3d import Axes3D

from map import Map
from vessel import Vessel
from world import World
from simulation import Simulation

from ctrl_DWA import DynamicWindow
from ctrl_hybrid_astar import HybridAStar
from ctrl_LOS_Guidance import LOSGuidance
from ctrl_PotField import PotentialFields
from ctrl_astar import AStar
from ctrl_purepursuit import PurePursuit
from ctrl_constant_bearing import ConstantBearing
from ctrl_wafi import Wafi
from ctrl_VO import VO
from ctrl_MOPSO import Mopso

from matplotlib2tikz import save as tikz_save


# noinspection PyUnreachableCode
class Scenario(object):
    def __init__(self, mapname, ctrlnames, scenname, name='s1'):

        self.name = scenname + "-" + "-".join(ctrlnames)

        self.tend = 150   # Simulation time (seconds)
        self.h    = 0.05  # Integrator time step
        self.dT   = 0.5   # Controller time step
        self.N    = int(np.around(self.tend / self.h)) + 1 # Number of simulation steps
        N2 = int(self.tend/self.dT) + 1

        if scenname == "s1":
            # Vessel 1 (Main vessel)
            x01 = np.array([0.0, 0.0, 0.0, 2.5, 0, 0])
            xg1 = np.array([140, 140, 0])

        elif scenname == "s2":
            # Vessel 1 (Main vessel)
            x01 = np.array([0.0, 0.0, 0.0, 2.5, 0, 0])
            xg1 = np.array([140, 140, 0])

        elif scenname == "s3":
            # Vessel 1 (Main vessel)
            x01 = np.array([100.0, 0.0, 3.14/4, 2.5, 0, 0])
            xg1 = np.array([80, 145, 3.14/2])

            # Follower
            x0f = np.array([120.,110,-np.pi,1.5,0,0])
            xgf = np.array([250,110,0])
            ppf = PurePursuit(mode='pursuit')

        elif scenname == "VO_test":
            # Vessel 1 (Main vessel)
            x01 = np.array([75, 0.0, np.pi/2, 2.5, 0, 0]) # Starting position x, y, angle & starting acceleration u,v,r
            xg1 = np.array([75, 150, 0])

            # Vessel 2 (WAFI)
            x02 = np.array([150, 80, np.pi, 2.5, 0, 0])
            xg2 = np.array([0, 80, 0])

        elif scenname == "wafi":
            # Vessel 1 (Main vessel)
            x01 = np.array([80, 0.0, np.pi / 2, 2.5, 0, 0])
            xg1 = np.array([80, 150, 0])

            # Vessel 2 (WAFI)
            x0f = np.array([80, 80, np.pi*1.5, 2.5, 0, 0])
            xgf = np.array([250, 10, 0])
            ppf = Wafi(mode='wafi')

        elif scenname == "mopso_test":
            # Vessel 1
            x01 = np.array([0, 0, np.pi / 4, 3.0, 0.0, 0])
            xg1 = np.array([30, 86, np.pi / 4])

        else:
            # Vessel 1 (Main vessel)
            x01 = np.array([10.0, 10.0, 3.14/4, 2.5, 0, 0])
            xg1 = np.array([100, 125, 3.14/2])


        the_map = Map(mapname, gridsize=1., safety_region_length=4.0)

        controllers = []

        for name in ctrlnames:
            if name == "dwa":
                controllers.append(DynamicWindow(self.dT,
                                                 N2,
                                                 the_map))
            elif name == "potfield":
                controllers.append(PotentialFields(the_map, N2))
                if len(controllers) > 1:
                    controllers[-1].d_max = 20.

            elif name == "astar":
                controllers.append(AStar(x01, xg1, the_map))
                controllers.append(LOSGuidance(switch_criterion="circle"))

            elif name == "hastar":
                controllers.append(HybridAStar(x01, xg1, the_map))
                controllers.append(LOSGuidance(switch_criterion="progress"))

            elif name == "mopso":
                controllers.append(Mopso(x01, xg1, the_map))

        v1 = Vessel(x01,
                    xg1,
                    self.h,
                    self.dT,
                    self.N,
                    controllers,
                    is_main_vessel=True,
                    vesseltype='viknes')

        vessels = [v1]

        if scenname == "s3":
            ppf.cGoal = v1.x
            vf = Vessel(x0f,
                        xgf,
                        self.h,
                        self.dT,
                        self.N,
                        [ppf],
                        is_main_vessel=False,
                        vesseltype='viknes')
            vf.u_d = 2.5
            vessels.append(vf)

        elif scenname == "VO_test":
            controllers2 = []
            controllers2.append(AStar(x02, xg2, the_map))
            controllers2.append(LOSGuidance(switch_criterion="progress"))
            v2 = Vessel(x02,
                        xg2,
                        self.h,
                        self.dT,
                        self.N,
                        controllers2,
                        is_main_vessel=False,
                        vesseltype='viknes')
            v2.u_d = 2.5
            vessels.append(v2)

        elif scenname == "wafi":
            ppf.cGoal = v1.x
            vf = Vessel(x0f,
                        xgf,
                        self.h,
                        self.dT,
                        self.N,
                        [ppf],
                        is_main_vessel=False,
                        vesseltype='viknes')
            vf.u_d = 2
            vessels.append(vf)


        elif scenname == "mopso_test":
            pass

        self.world = World(vessels, the_map)
        return


        if name == 's1':
            the_map = Map('s1', gridsize=2.0, safety_region_length=4.0)

            self.tend = 150
            self.h    = 0.05
            self.dT   = 0.5
            self.N    = int(np.around(self.tend / self.h)) +1

            # Vessel 1 (Main vessel)
            x01 = np.array([0, 0, 0, 1.0, 0, 0])
            xg1 = np.array([150, 150, np.pi/4])

            myastar = AStar(x01, xg1, the_map)
            mypp    = PurePursuit()

            v1 = Vessel(x01, xg1, self.h, self.dT, self.N, [myastar, mypp], is_main_vessel=True, vesseltype='viknes')

            self.world = World([v1], the_map)

        elif name == 'collision':
            the_map = Map('s1')

            self.tend = 100
            self.h    = 0.05
            self.dT   = 0.5
            self.N    = int(np.around(self.tend / self.h)) + 1
            self.h    = 0.05
            # Vessel 1 (Main vessel)
            x01 = np.array([0, 0, 0, 3.0, 0, 0])
            xg1 = np.array([120, 120, np.pi/4])

            myLOS1 = LOSGuidance()
            myAstar = HybridAStar(x01, xg1, the_map)

            v1 = Vessel(x01, xg1,self.h, self.dT, self.N, [myLOS1], is_main_vessel=True, vesseltype='viknes')
            v1.waypoints = np.array([[0, 0], [50, 60], [70, 60], [120, 10], [120, 120]])
            #v1.waypoints = np.array([[0, 0], [140, 0], [120, 120]])

            # Vessel 2
            x02 = np.array([0, 120, 0, 3.0, 0, 0])
            xg2 = np.array([120, 0, 0])

            myLOS2 = LOSGuidance()

            v2 = Vessel(x02, xg2,self.h, self.dT, self.N, [myLOS2], is_main_vessel=False, vesseltype='viknes')
            v2.waypoints = np.array([[0, 120], [120, 120], [120, 0]])

            # Vessel 3
            x03 = np.array([0, 50, np.pi/2, 3.0, 0, 0])
            xg3 = np.array([140, 0, 0])

            myLOS3 = LOSGuidance()

            v3 = Vessel(x03, xg3, self.h, self.dT, self.N, [myLOS3], is_main_vessel=False, vesseltype='viknes')
            v3.waypoints = np.array([[0, 50], [0, 120], [120, 120]])


            self.world = World([v1, v2], the_map)

        elif name == 's1-dynwnd':
            the_map = Map('s1', gridsize=0.5,safety_region_length=4.0)

            self.tend = 80   # Simulation time (seconds)
            self.h    = 0.05 # Integrator time step
            self.dT   = 0.5  # Controller time step
            self.N    = int(np.around(self.tend / self.h)) + 1

            # Vessel 1 (Main vessel)
            x01 = np.array([0.0, 0.0, 0.0, 2.5, 0, 0])
            xg1 = np.array([140, 140, 0])

            myDynWnd = DynamicWindow(self.dT, int(self.tend/self.dT) + 1, the_map)

            v1 = Vessel(x01, xg1, self.h, self.dT, self.N, [myDynWnd], is_main_vessel=True, vesseltype='viknes')
            v1.goal = np.array([140, 140, 0])

            self.world = World([v1], the_map)

        elif name == 's1-potfield':
            the_map = Map('s1', gridsize=0.5, safety_region_length=4.0)

            self.tend = 140.0
            self.dT   = 0.5
            self.h    = 0.05
            self.N    = int(np.around(self.tend/self.h)) + 1

            N2   = int(np.around(self.tend/self.dT)) + 1
            x0   = np.array([0, 0, 0, 2.5, 0, 0])
            xg   = np.array([140, 140, 0])

            potfield = PotentialFields(the_map, N2)

            vobj = Vessel(x0, xg, self.h, self.dT, self.N, [potfield], is_main_vessel=True, vesseltype='viknes')

            self.world = World([vobj], the_map)

        elif name=='s1-hybridastar':
            the_map = Map('s1', gridsize=2.0, safety_region_length=4.0)

            self.tend = 120.0
            self.dT   = 0.5
            self.h    = 0.05
            self.N    = int(np.around(self.tend/self.h)) + 1

            N2   = int(np.around(self.tend/self.dT)) + 1
            x0 = np.array([0, 0, 0.0, 2.5, 0.0, 0])
            xg = np.array([140, 140, 0.0])

            hastar = HybridAStar(x0, xg, the_map)
            pp    = PurePursuit(R2=50)

            vobj = Vessel(x0, xg, self.h, self.dT, self.N, [hastar, pp], is_main_vessel=True, vesseltype='viknes')

            self.world = World([vobj], the_map)

        elif name=='s1-astar':
            the_map = Map('s1', gridsize=2.0, safety_region_length=4.0)

            self.tend = 120.0
            self.dT   = 0.5
            self.h    = 0.05
            self.N    = int(np.around(self.tend/self.h)) + 1

            N2   = int(np.around(self.tend/self.dT)) + 1
            x0 = np.array([0, 0, 0.0, 2.5, 0.0, 0])
            xg = np.array([140, 140, np.pi/4])

            astar = AStar(x0, xg, the_map)
            pp    = PurePursuit(R2=50)

            vobj = Vessel(x0, xg, self.h, self.dT, self.N, [astar, pp], is_main_vessel=True, vesseltype='viknes')

            self.world = World([vobj], the_map)


        elif name=='s2-potfield':
            the_map = Map('s2', gridsize=1.0, safety_region_length=4.0)

            self.tend = 150.0
            self.dT   = 0.5
            self.h    = 0.05
            self.N    = int(np.around(self.tend/self.h)) + 1

            N2   = int(np.around(self.tend/self.dT)) + 1
            x0   = np.array([0, 0, 0, 2.5, 0, 0])
            xg   = np.array([140, 140, 0])

            potfield = PotentialFields(the_map, N2)

            vobj = Vessel(x0, xg, self.h, self.dT, self.N, [potfield], is_main_vessel=True, vesseltype='viknes')

            self.world = World([vobj], the_map)

        elif name == 's2-dynwnd':
            the_map = Map('s2', gridsize=0.5,safety_region_length=4.0)

            self.tend = 80   # Simulation time (seconds)
            self.h    = 0.05 # Integrator time step
            self.dT   = 0.5  # Controller time step
            self.N    = int(np.around(self.tend / self.h)) + 1

            # Vessel 1 (Main vessel)
            x01 = np.array([0.0, 0.0, 0.0, 2.5, 0, 0])
            xg1 = np.array([140, 140, 0])

            myDynWnd = DynamicWindow(self.dT, int(self.tend/self.dT) + 1, the_map)

            v1 = Vessel(x01, xg1, self.h, self.dT, self.N, [myDynWnd], is_main_vessel=True, vesseltype='viknes')
            v1.goal = np.array([140, 140, 0])

            self.world = World([v1], the_map)

        elif name=='s2-astar':
            the_map = Map('s2', gridsize=2.0, safety_region_length=4.0)

            self.tend = 120.0
            self.dT   = 0.5
            self.h    = 0.05
            self.N    = int(np.around(self.tend/self.h)) + 1

            N2   = int(np.around(self.tend/self.dT)) + 1
            x0 = np.array([0, 0, 0.0, 2.5, 0.0, 0])
            xg = np.array([140, 140, np.pi/4])

            astar = AStar(x0, xg, the_map)
            pp    = PurePursuit(R2=50)

            vobj = Vessel(x0, xg, self.h, self.dT, self.N, [astar, pp], is_main_vessel=True, vesseltype='viknes')

            self.world = World([vobj], the_map)

        elif name=='s2-hybridastar':
            the_map = Map('s2', gridsize=2.0, safety_region_length=4.0)

            self.tend = 120.0
            self.dT   = 0.5
            self.h    = 0.05
            self.N    = int(np.around(self.tend/self.h)) + 1

            N2   = int(np.around(self.tend/self.dT)) + 1
            x0 = np.array([0, 0, 0.0, 2.5, 0.0, 0])
            xg = np.array([140, 140, 0.0])

            hastar = HybridAStar(x0, xg, the_map)
            pp    = PurePursuit(R2=50)

            vobj = Vessel(x0, xg, self.h, self.dT, self.N, [hastar, pp], is_main_vessel=True, vesseltype='viknes')

            self.world = World([vobj], the_map)

        elif name=='s3-potfield':
            the_map = Map('', gridsize=1.0, safety_region_length=4.0)
            self.tend = 120   # Simulation time (seconds)
            self.h    = 0.05 # Integrator time step
            self.dT   = 0.5  # Controller time step
            self.N    = int(np.around(self.tend / self.h)) + 1
            N2        = int(self.tend/self.dT) + 1

            # Vessel 1 (Main vessel)
            x01 = np.array([60.0, 0.0, 3.14/4, 2.5, 0, 0])
            xg1 = np.array([80, 145, 0])

            potfield = PotentialFields(the_map, N2)
            vobj = Vessel(x01, xg1, self.h, self.dT, self.N, [potfield], is_main_vessel=True, vesseltype='viknes')

            # Follower
            x0f = np.array([120.,110,-np.pi,1.5,0,0])
            xgf = np.array([250,110,0])

            pp = PurePursuit(mode='pursuit')
            pp.cGoal = vobj.x
            vobj3 = Vessel(x0f, xgf, self.h, self.dT, self.N, [pp], is_main_vessel=False, vesseltype='viknes')
            vobj3.u_d = 2.5

            self.world = World([vobj, vobj3], the_map)

        elif name == 's3-dynwnd':
            the_map = Map('', gridsize=1.0, safety_region_length=4.0)

            self.tend = 80   # Simulation time (seconds)
            self.h    = 0.05 # Integrator time step
            self.dT   = 0.5  # Controller time step
            self.N    = int(np.around(self.tend / self.h)) + 1

            N2        = int(self.tend/self.dT) + 1

            # Vessel 1 (Main vessel)
            x01 = np.array([100.0, 0.0, 3.14/4, 2.5, 0, 0])
            xg1 = np.array([80, 145, 0])

            myDynWnd = DynamicWindow(self.dT, N2, the_map)

            vobj = Vessel(x01, xg1, self.h, self.dT, self.N, [myDynWnd], is_main_vessel=True, vesseltype='viknes')
            #v1.goal = np.array([140, 140, 0])

            # Follower
            x0f = np.array([120.,110,-np.pi,1.5,0,0])
            xgf = np.array([250,110,0])

            pp = PurePursuit(mode='pursuit')
            pp.cGoal = vobj.x
            vobj3 = Vessel(x0f, xgf, self.h, self.dT, self.N, [pp], is_main_vessel=False, vesseltype='viknes')
            vobj3.u_d = 2.5

            self.world = World([vobj, vobj3], the_map)

        elif name == 's3-astar':
            the_map = Map('', gridsize=1.0, safety_region_length=4.0)

            self.tend = 80   # Simulation time (seconds)
            self.h    = 0.05 # Integrator time step
            self.dT   = 0.5  # Controller time step
            self.N    = int(np.around(self.tend / self.h)) + 1

            N2        = int(self.tend/self.dT) + 1

            # Vessel 1 (Main vessel)
            x01 = np.array([100.0, 0.0, 3.14/4, 2.5, 0, 0])
            xg1 = np.array([80, 145, 0])

            astar = AStar(x01,xg1,the_map)
            pp    = PurePursuit(R2=50)

            vobj = Vessel(x01, xg1, self.h, self.dT, self.N, [astar, pp], is_main_vessel=True, vesseltype='viknes')
            #v1.goal = np.array([140, 140, 0])

            # Follower
            x0f = np.array([120.,110,-np.pi,1.5,0,0])
            xgf = np.array([250,110,0])

            pp = PurePursuit(mode='pursuit')
            pp.cGoal = vobj.x
            vobj3 = Vessel(x0f, xgf, self.h, self.dT, self.N, [pp], is_main_vessel=False, vesseltype='viknes')
            vobj3.u_d = 2.5

            self.world = World([vobj, vobj3], the_map)

        elif name == 's3-hybridastar':
            the_map = Map('', gridsize=1.0, safety_region_length=4.0)

            self.tend = 300   # Simulation time (seconds)
            self.h    = 0.05 # Integrator time step
            self.dT   = 0.5  # Controller time step
            self.N    = int(np.around(self.tend / self.h)) + 1

            N2        = int(self.tend/self.dT) + 1

            # Vessel 1 (Main vessel)
            x01 = np.array([100.0, 0.0, 3.14/4, 2.5, 0, 0])
            xg1 = np.array([80, 145, 3.14/2])

            hastar   = HybridAStar(x01, xg1, the_map)
            los      = LOSGuidance(switch_criterion="progress")
            dwa      = DynamicWindow(self.dT, int(self.tend/self.dT) + 1, the_map)

            dwa.alpha = .5
            dwa.beta  = .4
            dwa.gamma = .4

            pot = PotentialFields(the_map, N2)
            pot.d_max = 40

            vobj = Vessel(x01, xg1, self.h, self.dT, self.N, [hastar, los, dwa], is_main_vessel=True, vesseltype='viknes')
            #v1.goal = np.array([140, 140, 0])

            # Follower
            x0f = np.array([120.,110,-np.pi,1.5,0,0])
            xgf = np.array([250,110,0])

            pp = PurePursuit(mode='pursuit')
            pp.cGoal = vobj.x
            vobj3 = Vessel(x0f, xgf, self.h, self.dT, self.N, [pp], is_main_vessel=False, vesseltype='viknes')
            vobj3.u_d = 2.5

            self.world = World([vobj, vobj3], the_map)

        elif name == 'hastar+dynwnd':
            the_map = Map('s2', gridsize=1.0, safety_region_length=4.0)

            self.tend = 100   # Simulation time (seconds)
            self.h    = 0.05 # Integrator time step
            self.dT   = 0.5  # Controller time step
            self.N    = int(np.around(self.tend / self.h)) + 1

            # Vessel 1 (Main vessel)
            x01 = np.array([0.0, 0.0, 3.14/4, 2.5, 0, 0])
            xg1 = np.array([80, 145, 0])

            hastar   = HybridAStar(x01, xg1, the_map)
            pp       = PurePursuit(mode='goal-switcher')
            los      = LOSGuidance(switch_criterion="progress")
            myDynWnd = DynamicWindow(self.dT, int(self.tend/self.dT) + 1, the_map)

            myDynWnd.alpha = .5
            myDynWnd.beta  = .4
            myDynWnd.gamma = .4

            vobj = Vessel(x01, xg1, self.h, self.dT, self.N, [hastar, los, myDynWnd], is_main_vessel=True, vesseltype='viknes')


            xo0 = np.array([50.,130,5*np.pi/4,0.0,0,0])
            xog = np.array([250,110,0])


            vobj2 = Vessel(xo0, xog, self.h, self.dT, self.N, [], is_main_vessel=False, vesseltype='hurtigruta')


            self.world = World([vobj, vobj2], the_map)
            myDynWnd.world = self.world

        elif name == 'cb-test':
            the_map = Map('', gridsize=1.0, safety_region_length=4.0)

            self.tend = 80   # Simulation time (seconds)
            self.h    = 0.05 # Integrator time step
            self.dT   = 0.5  # Controller time step
            self.N    = int(np.around(self.tend / self.h)) + 1

            # Vessel 1 (Main vessel)
            x01 = np.array([60.0, 0.0, 3.14/4, 2.5, 0, 0])
            xg1 = np.array([80, 145, 0])

            pp = PurePursuit()

            vobj = Vessel(x01, xg1, self.h, self.dT, self.N, [pp], is_main_vessel=True, vesseltype='viknes')

            # Other vessel
            xo0 = np.array([40.,60,-np.pi/4,1.5,0,0])
            xog = np.array([250,110,0])

            cb  = ConstantBearing(vobj.x)

            vobj2 = Vessel(xo0, xog, self.h, self.dT, self.N, [cb], is_main_vessel=False, vesseltype='hurtigruta')

            self.world = World([vobj, vobj2], the_map)


        elif name == 'dwacollision':
            the_map = Map(gridsize=0.5,safety_region_length=4.5)

            self.tend = 60   # Simulation time (seconds)
            self.h    = 0.05 # Integrator time step
            self.dT   = 0.5  # Controller time step
            self.N    = int(np.around(self.tend / self.h)) + 1

            # Vessel 1 (Main vessel)
            x01 = np.array([5, 5, np.pi/4, 3.0, 0, 0])
            xg1 = np.array([120, 120, np.pi/4])

            myLOS1 = LOSGuidance()
            myDynWnd = DynamicWindow(self.dT, int(self.tend/self.dT) + 1)

            v1 = Vessel(x01, xg1, self.h, self.dT, self.N, [myDynWnd], is_main_vessel=True, vesseltype='viknes')
            v1.waypoints = np.array([[0, 0], [120, 120]])
            #v1.waypoints = np.array([[0, 0], [140, 0], [120, 120]])

            # Vessel 2
            x02 = np.array([80, 80, -3*np.pi/4, 3.0, 0, 0])
            xg2 = np.array([0, 0, -3*np.pi/4])

            myLOS2 = LOSGuidance(u_d = 2.0)

            v2 = Vessel(x02, xg2, self.h, self.dT, self.N, [myLOS2], is_main_vessel=False, vesseltype='viknes')
            v2.waypoints = np.array([[120, 120], [0, 0]])

            self.world = World([v1, v2], the_map)

            myDynWnd.the_world = self.world


        elif name=='hybridastar':
            the_map = Map('s1',gridsize=1.0, safety_region_length=6.0)

            self.tend = 140.0
            self.dT   = 0.5
            self.h    = 0.05
            self.N    = int(np.around(self.tend/self.h)) + 1

            N2   = int(np.around(self.tend/self.dT)) + 1
            x0 = np.array([0, 0, 0.0, 2.5, 0.0, 0])
            xg = np.array([130, 130, 8.0])

            hastar = HybridAStar(x0, xg, the_map)
            pp     = PurePursuit(R2=50, mode="goal-switcher")

            dynwnd = DynamicWindow(self.dT, N2, the_map)
            dynwnd.alpha = 0.8
            dynwnd.beta  = 0.1
            dynwnd.gamma = 0.1

            ptf    = PotentialFields(the_map, N2)
            ptf.mu    = 10
            ptf.d_max = 30
            ptf.k     = 10.

            vobj = Vessel(x0, xg, self.h, self.dT, self.N, [hastar, pp, dynwnd], is_main_vessel=True, vesseltype='viknes')

            self.world = World([vobj], the_map)
            dynwnd.the_world = self.world
            ptf.world = self.world
        elif name=='minima':
            the_map = Map('minima', gridsize=0.5, safety_region_length=4.0)

            self.tend = 120.0
            self.dT   = 0.5
            self.h    = 0.1
            self.N    = int(np.around(self.tend/self.h)) + 1

            N2   = int(np.around(self.tend/self.dT)) + 1
            x0   = np.array([30,30,0, 2.0,0,0])
            xg   = np.array([140, 140, 3*np.pi/4])

            hastar = HybridAStar(x0, xg, the_map)
            los    = LOSGuidance()

            vobj = Vessel(x0, xg, self.h, self.dT, self.N, [hastar, los], is_main_vessel=True, vesseltype='viknes')

            self.world = World([vobj], the_map)

        elif name=='pptest':
            the_map = Map('', gridsize=0.5, safety_region_length=4.0)

            self.tend = 120.0
            self.dT   = 0.5
            self.h    = 0.1
            self.N    = int(np.around(self.tend/self.h)) + 1

            N2   = int(np.around(self.tend/self.dT)) + 1
            x0   = np.array([0,0,0, 2.0,0,0])
            xg   = np.array([140, 140, 3*np.pi/4])

            pp   = PurePursuit()

            vobj = Vessel(x0, xg, self.h, self.dT, self.N, [pp], is_main_vessel=True, vesseltype='viknes')
            vobj.waypoints = np.array([(50.,50.), (50., 0.), (100., 100.)])
            self.world = World([vobj], the_map)

        elif name=='wafi':
            self.tend = 80  # Simulation time (seconds)
            self.h = 0.05  # Integrator time step
            self.dT = 0.5  # Controller time step
            self.N = int(np.around(self.tend / self.h)) + 1

            N2 = int(self.tend / self.dT) + 1

            # Vessel 1 (Main vessel)
            x01 = np.array([100.0, 0.0, 3.14 / 4, 2.5, 0, 0])
            xg1 = np.array([80, 145, 0])

            astar = AStar(x01, xg1, the_map)
            pp = PurePursuit(R2=50)

            vobj = Vessel(x01, xg1, self.h, self.dT, self.N, [astar, pp], is_main_vessel=True, vesseltype='viknes')
            # v1.goal = np.array([140, 140, 0])

            # Follower
            x0f = np.array([120., 110, -np.pi, 1.5, 0, 0])
            xgf = np.array([250, 110, 0])

            pp = Wafi(mode='wafi')
            pp.cGoal = 0
            vobj3 = Vessel(x0f, xgf, self.h, self.dT, self.N, [pp], is_main_vessel=False, vesseltype='wafi')
            vobj3.u_d = 2.5

            self.world = World([vobj, vobj3], the_map)

        else:
            print("You might have spelled the scenario name wrong...")
            self.tend = 0
            self.dT = 0.1
            self.h  = 0.05
            self.N  = 0
            self.world = None

    def is_occupied(self, point):
        return self.map.is_occupied(point)

    def draw(self, axes, n, scolor='b', fcolor='r', ocolor='g', ecolor='k'):
        self.world.draw(axes, n)

    def animate(self, fig, axes, n):
        return self.world.animate(fig, axes, n)

def harry_plotter(sim):
    fig = plt.figure()
    ax  = fig.add_subplot(111, autoscale_on=False)

    #ani = sim.animate(fig, ax)
    ax.axis('scaled')
    ax.set_xlim((-10, 160))
    ax.set_ylim((-10, 160))
    ax.set_xlabel('East [m]')
    ax.set_ylabel('North [m]')

    sim.draw(ax)


    #    ax.grid()
    #    plt.tight_layout()

    # tikz_save('astar-admissibility-2.tikz',
    #           figureheight='1.5\\textwidth',
    #           figurewidth='1.5\\textwidth')
    plt.show()

def harry_anim(sim):
    fig2 = plt.figure()
    ax2  = fig2.add_subplot(111, autoscale_on=False )


    ani = sim.animate(fig2,ax2)
    ax2.axis('scaled')
    ax2.set_xlim((-10, 160))
    ax2.set_ylim((-10, 160))
    ax2.set_xlabel('East [m]')
    ax2.set_ylabel('North [m]')

    ax2.grid()

    Writer = animation.writers['ffmpeg']
    writer = Writer(fps=33, bitrate=1800)
    ani.save(sim.scenario.name+'.mp4', writer=writer)
    plt.show()

if __name__ == "__main__":
    # plt.ion()

    # fig = plt.figure()

    # gs    = gridspec.GridSpec(2,4)
    # axarr = [fig.add_subplot(gs[:,0:2], autoscale_on=False)]

    # axarr[0].axis('scaled')
    # axarr[0].set_xlim((-10, 160))
    # axarr[0].set_ylim((-10, 160))
    # axarr[0].set_xlabel('East [m]')
    # axarr[0].set_ylabel('North [m]')

    # axarr += [fig.add_subplot(gs[0, 2], projection='3d'),
    #           fig.add_subplot(gs[1, 2], projection='3d'),
    #           fig.add_subplot(gs[0, 3], projection='3d'),
    #           fig.add_subplot(gs[1, 3], projection='3d')]
    #     #axarr[ii+1].set_aspect('equal')

    # for sname in ["s3"]:
    #     for ctrl in ["potfield", "dwa", "astar", "hastar"]:
    #         scen = Scenario(sname, [ctrl], sname)
    #         sim  = Simulation(scen, savedata=False)
    #         sim.run_sim()

    #     for ctrl in ["astar", "hastar"]:
    #         scen = Scenario(sname, [ctrl,'potfield'], sname)
    #         sim  = Simulation(scen, savedata=False)
    #         sim.run_sim()

    #         scen = Scenario(sname, [ctrl,'dwa'], sname)
    #         sim  = Simulation(scen, savedata=False)
    #         sim.run_sim()
    #sim  = Simulation(scen, fig, axarr)

        #map,controller,scene
    scen = Scenario("s1", ["astar", "mopso"], "mopso_test")
    sim  = Simulation(scen, savedata=False)

    sim.run_sim()
    #plt.show()
    harry_plotter(sim)
    harry_anim(sim)
    plt.show()


