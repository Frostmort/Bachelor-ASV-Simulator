import sys, time
import heapq

import random
import numpy as np
import matplotlib.pyplot as plt

from map import Map
from utils import Controller
from vessel import Vessel

DIMENSIONS = 2  # Number of dimensions
GLOBAL_BEST = 0  # Global Best of Cost function
MIN_RANGE = 0  # Lower boundary of search space
MAX_RANGE = 50 # Upper boundary of search space
POPULATION = 50  # Number of particles in the swarm
V_MAX = 1 # Maximum velocity value
PERSONAL_C = 2.0  # Personal coefficient factor
SOCIAL_C = 2.0  # Social coefficient factor
CONVERGENCE = 0  # Convergence value
MAX_ITER = 200  # Maximum number of iterations
BIGVAL = 10000.
MINDIST = 20


class Vopso(Controller):
    def __init__(self, x0, xg, the_map, search_radius=50, replan=False):
        self.start = x0[0:3]
        self.goal = xg[0:3]
        self.scanRadius = search_radius
        self.world = None
        self.grid_size = the_map.get_dimension
        self.graph = SearchGrid(the_map, [1.0, 1.0, 25.0/360.0])
        self.map = the_map
        self.to_be_updated = True
        self.replan = replan
        self.path_found = False
        self.wpUpdated = False
        self.currentcWP = 0

        self.alter = 0

        self.particles = []  # List of particles in the swarm
        self.best_pos = None  # Best particle of the swarm
        self.best_pos_z = np.inf  # Best particle of the swarm

    def update(self, vobj, world, vesselArray):
        tic = time.process_time()
        if len(vesselArray) > 1:
            v2 = vesselArray[1]
            resetPoint = -1
            scanData = self.scan(vobj.x[0:2], v2.x[0:2])
            if scanData[0] <= self.scanRadius and not self.wpUpdated:

                # Create VO
                VOarray = self.createVO(vobj, v2, scanData)
                if VOarray[3] > VOarray[8] > VOarray[4]:
                    # Implement MOPSO
                    self.currentcWP = vobj.controllers[1].cWP
                    nextWP = self.search(vobj.x[0:2], vesselArray, scanData, VOarray)
                    print("Vessel 1: ", vobj.x[0:2])
                    print("Vessel 2: ", v2.x[0:2])
                    for x in range(0, 2):
                        print("Vegpunkt ", self.currentcWP + x, ": ", nextWP)
                        vobj.controllers[1].wp = np.insert(vobj.waypoints, self.currentcWP + x, nextWP, axis = 0)
                        vobj.waypoints = np.insert(vobj.waypoints, self.currentcWP + x, nextWP, axis = 0)
                        scanData = self.scan(nextWP, v2.x[0:2])
                        nextWP = self.search(nextWP, vesselArray, scanData, VOarray)
                    self.wpUpdated = True

            if vobj.controllers[1].cWP == self.currentcWP + 1:
                vobj.wp = None
                vobj.controllers[1].cWP = 0
                vobj.controllers[0].to_be_updated = True
                vobj.controllers[1].wp_initialized = False
                self.wpUpdated = False


    def search(self, vobjx, vesselArray, scanData, VOarray):

        # Initialize swarm
        x0 = vobjx[0:2]
        print("Sverm0 ", x0)

        swarm = Swarm(POPULATION, V_MAX, self.goal, x0, vesselArray, scanData, VOarray)
        # Initialize inertia weight
        inertia_weight = 0.5 + (np.random.rand() / 2)
        curr_iter=0
        for i in range(MAX_ITER):

            for particle in swarm.particles:

                for i in range(0, DIMENSIONS):
                    r1 = np.random.uniform(0, 1)
                    r2 = np.random.uniform(0, 1)

                    # Update particle's velocity
                    personal_coefficient = PERSONAL_C * r1 * (particle.best_pos[i] - particle.pos[i])
                    social_coefficient = SOCIAL_C * r2 * (swarm.best_pos[i] - particle.pos[i])
                    new_velocity = inertia_weight * particle.velocity[i] + personal_coefficient + social_coefficient

                    # Check if velocity is exceeded
                    if new_velocity > V_MAX:
                        particle.velocity[i] = V_MAX
                    elif new_velocity < -V_MAX:
                        particle.velocity[i] = -V_MAX
                    else:
                        particle.velocity[i] = new_velocity

                # Update particle's current position
                particle.pos += particle.velocity

                particle.pos_z = swarm.cost_function(particle.pos[0], particle.pos[1], vesselArray)

                # Update particle's best known position
                if particle.pos_z < swarm.cost_function(particle.best_pos[0], particle.best_pos[1], vesselArray):
                    particle.best_pos = particle.pos.copy()

                    # Update swarm's best known position
                    if particle.pos_z < swarm.best_pos_z:
                        swarm.best_pos = particle.pos.copy()
                        swarm.best_pos_z = particle.pos_z

                # # Check if particle is within boundaries
                biggest = 0
                hypeCheck = np.hypot(particle.pos[0] - x0[0], particle.pos[1] - x0[1])
                if hypeCheck > biggest:
                    biggest = hypeCheck
                if np.hypot(particle.pos[0] - x0[0], particle.pos[1] - x0[1]) > MAX_RANGE:
                    r = np.random.uniform(MIN_RANGE, MAX_RANGE)
                    theta = np.random.uniform(0, 2 * np.pi)
                    particle.pos[0] = (r * np.cos(theta)) + x0[0]
                    particle.pos[1] = (r * np.sin(theta)) + x0[1]


            # Check for convergence
            if abs(swarm.best_pos_z - GLOBAL_BEST) < CONVERGENCE:
                print("The swarm has met convergence criteria after " + str(curr_iter) + " iterations.", 'at:',swarm.best_pos)
                break
            curr_iter += 1

        if abs(swarm.best_pos_z - GLOBAL_BEST) > CONVERGENCE:
            print("The swarm has reached max iterations after " + str(curr_iter) + " iterations.", 'at:',
                  swarm.best_pos)
        print("Swarm: ", swarm.best_pos)
        print("Biggast: ", biggest)
        return [swarm.best_pos[0], swarm.best_pos[1]]

    def scan(self, vessel1, vessel2):
        xd = (vessel2[0] - vessel1[0])
        yd = (vessel2[1] - vessel1[1])
        distance = abs(np.sqrt(xd**2 + yd**2))
        angle = np.arctan2(yd, xd)

        return [distance, angle]

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
##########################################################################################################
class Swarm():
    def __init__(self, pop, v_max, goal, x0, vesselArray, scanData, VOarray):
        self.particles = []         # List of particles in the swarm
        self.best_pos = None        # Best particle of the swarm
        self.best_pos_z = np.inf    # Best particle of the swarm
        self.x0 = x0                #ship pos
        self.goal = goal
        self.scanData = scanData
        self.vesselArray = vesselArray
        self.VOarray = VOarray
        for _ in range(pop):
            r = np.random.uniform(MIN_RANGE, MAX_RANGE)
            theta = np.random.uniform(MIN_RANGE, MAX_RANGE*np.pi)
            x = (r * np.cos(theta))+x0[0]
            y = (r * np.sin(theta))+x0[1]

            z = self.cost_function(x, y, goal)
            velocity = np.random.rand(2) * v_max
            particle = Particle(x, y, z, velocity)
            self.particles.append(particle)
            if self.best_pos != None and particle.pos_z < self.best_pos_z:
                self.best_pos = particle.pos.copy()
                self.best_pos_z = particle.pos_z
            else:
                self.best_pos = particle.pos.copy()
                self.best_pos_z = particle.pos_z


    def cost_function(self, x1, y1, goal):
        devW = 1
        statW = 1
        dynW = 1
        weighingMatrix = np.array([[devW], [statW], [dynW]])
        pos = x1,y1
        x2,y2=self.goal[0],self.goal[1]

        deviation_cost = (np.sqrt((x2-x1)**2 + (y2-y1)**2))   #distance from goal

        statitc_obs_cost = 0

        distance = np.hypot(x1 - self.vesselArray[1].x[0], y1 - self.vesselArray[1].x[1])    #check for dynamic obstacle
        if distance <= 5:
            dyn_obs_cost = BIGVAL
        elif 5 < distance <= 10:
            dyn_obs_cost = 100
        elif 10 < distance <= 20:
            dyn_obs_cost = 50
        else:
            dyn_obs_cost = 0

        if self.is_inside(self.get_dangercone(), pos):
            dyn_obs_cost = BIGVAL

        if dyn_obs_cost < 0:
            dyn_obs_cost = 0

        cost = np.sum(np.array([[deviation_cost], [statitc_obs_cost], [dyn_obs_cost]]) * weighingMatrix)
        return cost

    def is_inside(self, triangle, pos):           #check if point is inside cone
        x1 = triangle[0]
        x2 = triangle[1]
        x3 = triangle[2]
        xp = pos

        c1 = (x2[0]-x1[0]) * (xp[1]-x1[1]) - (x2[1]-x1[1]) * (xp[0]-x1[0])
        c2 = (x3[0]-x2[0]) * (xp[1]-x2[1]) - (x3[1]-x2[1]) * (xp[0]-x2[0])
        c3 = (x1[0]-x3[0]) * (xp[1]-x3[1]) - (x1[1]-x3[1]) * (xp[0]-x3[0])
        if (c1 < 0 and c2 < 0 and c3 < 0) or (c1 > 0 and c2 > 0 and c3 > 0):
            return True
        else:
            return False

    def get_dangercone(self):
        v1 = self.vesselArray[0]
        v2 = self.vesselArray[1]
        VOarray = self.VOarray
        tc = self.getCollisionTime(v1, v2)

        p0 = [v1.x[0] + (v2.x[3] * np.cos(v2.x[2]))*tc, v1.x[1] + (v2.x[3] * np.sin(v2.x[2]))*tc]
        p1 = [((self.scanData[0] + v2.x[3]*tc) * np.cos(VOarray[3]) + p0[0]), (self.scanData[0] + v2.x[3]*tc) * np.sin(VOarray[3]) + p0[1]]
        p2 = [((self.scanData[0] + v2.x[3]*tc) * np.cos(VOarray[4]) + p0[0]), (self.scanData[0] + v2.x[3]*tc) * np.sin(VOarray[4]) + p0[1]]

        #print("Trækant: ", p0, p1, p2)

        return [p0, p1, p2]

    def getCollisionTime(self, v1, v2):
        r1 = ([v1.x[0], v1.x[3] * np.cos(v1.x[2]), v1.x[1], v1.x[3] * np.sin(v1.x[2])])  # vessel 1 velocity vector
        r2 = ([v2.x[0], v2.x[3] * np.cos(v2.x[2]), v2.x[1], v2.x[3] * np.sin(v2.x[2])])  # vessel 2 velocity vector

        tx = (r2[0] - r1[0]) / (r1[1] - r2[1]) # time to intersect in x
        ty = (r2[2] - r1[2]) / (r1[3] - r2[3]) # time to intersect in y

        return (tx + ty) / 2 # returns average of x and y times

##########################################################################################################
# Particle class
class Particle():
    def __init__(self, x, y, z, velocity):
        self.pos = [x, y]
        self.pos_z = z
        self.velocity = velocity
        self.best_pos = self.pos.copy()

##########################################################################################################
class SearchGrid(object):
    """General purpose N-dimentional search grid."""
    def __init__(self, the_map, gridsize, N=2, parent=None):
        self.N        = N
        self.grid     = the_map.get_discrete_grid()
        self.map      = the_map
        self.gridsize = gridsize
        self.gridsize[0] = the_map.get_gridsize()
        self.gridsize[1] = the_map.get_gridsize()
        dim = the_map.get_dimension()

        self.width  = dim[0]
        self.height = dim[1]

        """
        In the discrete map, an obstacle has the value '1'.
        We multiply the array by a big number such that the
        grid may be used as a costmap.
        """
        self.grid *= BIGVAL


    def get_grid_id(self, state):
        """Returns a tuple (x,y,psi) with grid positions."""
        return (int(state[0]/self.gridsize[0]),
                int(state[1]/self.gridsize[1]),
                int(state[2]/self.gridsize[2]))

    def in_bounds(self, state):
        return 0 <= state[0] < self.width and 0 <= state[1] < self.height

    def cost(self, a, b):
        #if b[0] > self.width or b[1] > self.height:
        #    return 0
        return np.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2)#self.grid[int(b[0]/self.gridsize[0]), int(b[1]/self.gridsize[1])]

    def passable(self, state):
        # TODO Rename or change? Only returns true if object is _inside_ obstacle
        # Polygons add safety zone by default now.

        if state[0] > self.width or state[1] > self.height:
            return True

        return self.grid[int(state[0]/self.gridsize[0]),
                         int(state[1]/self.gridsize[1])] < BIGVAL

    def neighbors(self, state, est_r_max):
        """
        Applies rudder commands to find the neighbors of the given state.

        For the Viknes 830, the maximum rudder deflection is 15 deg.
        """
        step_length = 2.5*self.gridsize[0]
        avg_u       = 3.5
        Radius      = 2.5*avg_u / est_r_max
        dTheta      = step_length / Radius
        #print(Radius, dTheta*180/np.pi)

        trajectories = np.array([[step_length*np.cos(dTheta), step_length*np.sin(dTheta), dTheta],
                                 [step_length, 0.,  0.],
                                 [step_length*np.cos(dTheta), -step_length*np.sin(dTheta), -dTheta]])

        #print(trajectories)
        results = []
        for traj in trajectories:
            newpoint = state + np.dot(Rz(state[2]), traj)
            if self.passable(newpoint):
                results.append(newpoint)

        #results = filter(self.in_bounds, results)

        return results


#############################################################################


if __name__ == "__main__":
    mymap = Map("s1", gridsize=1.0, safety_region_length=4.5)

    x0 = np.array([0, 0, np.pi / 2, 3.0, 0.0, 0])
    xg = np.array([30, 86, np.pi / 4])
    mopso = Mopso(x0, xg, mymap)
    myvessel = Vessel(x0, xg, 0.05, 0.5, 1, [mopso], True, 'viknes')
    vesselArray = [myvessel]


    mopso.update(myvessel, vesselArray)

    fig = plt.figure()
    ax = fig.add_subplot(111, autoscale_on=False)
    #   ax.plot(myvessel.waypoints[:, 0],
    #           myvessel.waypoints[:, 1],
    #           '-')

    # ax.plot(x0[0], x0[1], 'bo')
    ax.plot(xg[0], xg[1], 'ro')
    myvessel.draw_patch(ax, myvessel.x, fcolor='b')

    # nonpass = np.array(nonpassable)
    # ax.plot(nonpass[:,0], nonpass[:,1], 'rx')

    ax.axis('equal')
    ax.axis('scaled')
    ax.axis([-10, 160, -10, 160])
    mymap.draw(ax, 'g', 'k')
    ax.grid()

    plt.show()
