#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt

import time

class Simulation(object):
    """This class combines the different classes needed to simulate a scenario.

    Args:
        scenario   (class): a Scenario object (contains map, etc.)
        vessel     (class): the vessel to be used (e.g. the Viknes 830)
        controllers (list): list of controller types (A*, LOS, etc.)
        model      (class): the mathematical model for the vessel
                                :todo: make vessel the parent for this class

    Kwargs:
        tend       (float): The end time of the simulation (default 100)

    """
    def __init__(self, scenario, fig=None, axarr=None, savedata=False):
        """Initializes the simulation."""
        self.scenario = scenario

        self.tend = self.scenario.tend
        self.N    = self.scenario.N
        self.n    = 0

        self.fig  = fig
        self.axarr = axarr

        self._end_reason = "SimOK"

        self.savedata = savedata
    def run_sim(self):
        """Runs the simulation."""

        print("Starting simulation")
        tic = time.process_time()

        for t in np.linspace(0, self.tend, self.N):
            """The actual simulation"""


            self.scenario.world.update_world(t, self.n)

            if self.axarr:
                self.scenario.world.visualize(self.fig, self.axarr, t, self.n)

            if self.scenario.world.collision_detection():
                print("We have crashed!")
                self._end_reason = "Collision"
                break
            if self.scenario.world.reached_goal():
                print("Final goal reached!")
                self._end_reason = "Success"
                break

            if np.fmod(t,5.0) == 0:
                pass
                #print("Simulation time: %.3f" % t)

            self.n += 1

        arr = [t, time.process_time()-tic]
        # print("Total simulation CPU time: %.3f"%(time.process_time() - tic))
        arr = self.scenario.world.get_simulation_data(self.n) + arr

        print(self.scenario.name)
        print("Distance\tAvgVel\tVarVel\tVarPsi\tMinDist\tTime\tCPU Time")
        s = '\t'.join(format(x,".4f") for x in arr)
        print(s)
        print()

        if self.savedata:
            self.scenario.world.save_data(self.n, self.scenario.name + '-vessel')
            metadata = [t, time.process_time()-tic]
            np.savetxt(self.scenario.name + '-metadata.txt', np.array(metadata), delimiter='\t')


        if self.axarr:
            raw_input("SIMULATION COMPLETE! Hit enter to exit")

    def draw(self, axes):
        """Draw the simulation.

        Plots the result of the simulation.

        Args:
            axes (class): the matplotlib axes to draw to.
        """
        self.scenario.draw(axes, self.n)

    def animate(self, fig, axes):
        return self.scenario.animate(fig, axes, self.n)

    def print_sim_time(self):
        """Prints the time taken to simulate the scenario.
        :todo: implement
        """
        print("Simulation time: %.3f seconds!" % (self.sim_time))
