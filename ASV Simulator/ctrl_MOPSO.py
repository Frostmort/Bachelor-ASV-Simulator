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
B_LO = -200  # Upper boundary of search space
B_HI = 200  # Upper boundary of search space
POPULATION = 300  # Number of particles in the swarm
V_MAX = 0.1  # Maximum velocity value
PERSONAL_C = 2.0  # Personal coefficient factor
SOCIAL_C = 2.0  # Social coefficient factor
CONVERGENCE = 0.001  # Convergence value
MAX_ITER = 300  # Maximum number of iterrations
BIGVAL = 10000.


class Mopso(Controller):
    def __init__(self, x0, xg, the_map, replan=False):
        self.start = x0[0:3]
        self.goal = xg[0:3]

        self.world = None
        self.grid_size=the_map.get_dimension
        self.graph = SearchGrid(the_map, [1.0, 1.0, 25.0/360.0])
        self.map = the_map
        self.to_be_updated = True
        self.replan = replan
        self.path_found = False


        self.particles = []  # List of particles in the swarm
        self.best_pos = None  # Best particle of the swarm
        self.best_pos_z = np.inf  # Best particle of the swarm

    def update(self, vessel_object, animate):
        if self.to_be_updated:
            tic = time.process_time()
            vessel_object.waypoints = self.search(vessel_object, animate)
            print("POS CPU time: %.3f" % (time.process_time() - tic))
            self.to_be_updated = False
            # self.map.disable_safety_region()

    def search(self, vesselarray, animate):
        # Initialize plotting variables
        if animate == True:
            x = np.linspace(B_LO, B_HI, 50)
            y = np.linspace(B_LO, B_HI, 50)
            X, Y = np.meshgrid(x, y)
            fig = plt.figure("Particle Swarm Optimization")

        # Initialize swarm
        swarm = Swarm(POPULATION, V_MAX)
        # Initialize inertia weight
        inertia_weight = 0.5 + (np.random.rand() / 2)
        curr_iter=0
        for i in range(MAX_ITER):

            for particle in swarm.particles:

                if animate:
                    fig.clf()
                    ax = fig.add_subplot(1, 1, 1)
                    ac = ax.contourf(X, Y, self.cost_function(X, Y), cmap='viridis')
                    fig.colorbar(ac)

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

                if animate == True:
                    ax.scatter(particle.pos[0], particle.pos[1], marker='*', c='r')
                    ax.arrow(particle.pos[0], particle.pos[1], particle.velocity[0], particle.velocity[1],
                             head_width=0.1,
                             head_length=0.1, color='k')

                # Update particle's current position
                particle.pos += particle.velocity
                particle.pos_z = self.cost_function(particle.pos[0], particle.pos[1])

                # Update particle's best known position
                if particle.pos_z < self.cost_function(particle.best_pos[0], particle.best_pos[1]):
                    particle.best_pos = particle.pos.copy()

                    # Update swarm's best known position
                    if particle.pos_z < swarm.best_pos_z:
                        swarm.best_pos = particle.pos.copy()
                        swarm.best_pos_z = particle.pos_z

                # Check if particle is within boundaries
                if particle.pos[0] > B_HI:
                    particle.pos[0] = np.random.uniform(B_LO, B_HI)
                    particle.pos_z = self.cost_function(particle.pos[0], particle.pos[1])
                if particle.pos[1] > B_HI:
                    particle.pos[1] = np.random.uniform(B_LO, B_HI)
                    particle.pos_z = self.cost_function(particle.pos[0], particle.pos[1])
                if particle.pos[0] < B_LO:
                    particle.pos[0] = np.random.uniform(B_LO, B_HI)
                    particle.pos_z = self.cost_function(particle.pos[0], particle.pos[1])
                if particle.pos[1] < B_LO:
                    particle.pos[1] = np.random.uniform(B_LO, B_HI)
                    particle.pos_z = self.cost_function(particle.pos[0], particle.pos[1])

            if animate == True:
                plt.subplots_adjust(right=0.95)
                plt.pause(0.00001)
                plt.show()
            # Check for convergence
            if abs(swarm.best_pos_z - GLOBAL_BEST) < CONVERGENCE:
                print("The swarm has met convergence criteria after " + str(curr_iter) + " iterrations.", 'at:',swarm.best_pos)
                break
            curr_iter += 1

    def cost_function(self, x1, y2):
        x2,y1=self.goal[0],self.goal[1]
        deviation_cost = np.sqrt((x1-x2)**2 + (y1-y2)**2)
        cost = deviation_cost
        return cost


class Swarm():
    def __init__(self, pop, v_max):
        self.particles = []  # List of particles in the swarm
        self.best_pos = None  # Best particle of the swarm
        self.best_pos_z = np.inf  # Best particle of the swarm

        for _ in range(pop):
            x = np.random.uniform(B_LO, B_HI)
            y = np.random.uniform(B_LO, B_HI)
            z = self.cost_function(x, y)
            velocity = np.random.rand(2) * v_max
            particle = Particle(x, y, z, velocity)
            self.particles.append(particle)
            if self.best_pos != None and particle.pos_z < self.best_pos_z:
                self.best_pos = particle.pos.copy()
                self.best_pos_z = particle.pos_z
            else:
                self.best_pos = particle.pos.copy()
                self.best_pos_z = particle.pos_z

    def cost_function(self, x, y, a=20, b=0.2, c=2 * np.pi):
        term_1 = np.exp((-b * np.sqrt(0.5 * (x ** 2 + y ** 2))))
        term_2 = np.exp((np.cos(c * x) + np.cos(c * y)) / 2)
        return -1 * a * term_1 - term_2 + a + np.exp(1)


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
    xg = np.array([100, 100, np.pi / 4])
    myvessel = Vessel(x0, xg, 0.05, 0.5, 1, [], True, 'viknes')
    vesselArray = [myvessel]
    mopso = Mopso(x0, xg, mymap)

    mopso.update(myvessel, animate=False)

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
