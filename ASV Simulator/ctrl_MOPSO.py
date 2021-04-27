import sys, time
import heapq

import random
import numpy as np
import matplotlib.pyplot as plt

from map import Map
from utils import Controller
from vessel import Vessel




class Mopso(Controller):
    def __init__(self, swarmsize=100, w=0.5, c1=0.5, c2=0.9, max_=0.9, min_=0.2, thresh, vesselArray, mesh_div=10):
        self.w,self.c1,self.c2 = w,c1,c2    #weight,individual best score, global best score
        self.mesh_div = mesh_div
        self.psize = 5                      #Particle size
        self.searchrange = [-1, 1]          #Search range
        self.swarmsize = swarmsize          #Number of particles
        self.maxiter = 100                  #Number of iterations
        self.thresh = thresh
        self.max_ = max_
        self.min_ = min_
        self.vmax = (max_-min_)*0.05        #Maximum speed
        self.vmin  = (max_-min_)*0.05*(-1)  #Minimum speed
        self.vesselArray = vesselArray
        self.is_initialized = False
        self.swarm


    def update(self):
        if not self.is_initialized:
            x,v = self.initialize_swarm(self.swarmsize, self.psize)
            fitness=self.calculate_fitness(x, v)
            self.is_initialized = True
        self.update_particles(self,x,v,fitness )
        self.calculate_fitness(x, v)

    def initialize_swarm(self,swarmsize,psize):
        x = []
        v = []
        for j in range(swarmsize):
            x.append([random.random() for i in range(psize)])
            v.append([random.random() for m in range(psize)])
        return x,v

    def calculate_fitness(self,x,v):
        fitness = [self.fun(x[j]) for j in range(self.N)]
        p = x
        best = min(fitness)
        pg = x[fitness.index(min(fitness))]
        best_all = []
        return p,best,pg,best_all


    def update_particles(self,x,v,fitness):
        p,best,pg,best_all=fitness
        for i in range(self.maxiter):
            for j in range(self.swarmsize):
                for m in range(self.psize):
                    v[j][m] = self.w * v[j][m] + self.c1 * random.random() * (
                            p[j][m] - x[j][m]) + self.c2 * random.random() * (pg[m] - x[j][m])
            for j in range(self.swarmsize):
                for m in range(self.psize):
                    x[j][m] = x[j][m] + v[j][m]
                    if x[j][m] > self.searchrange[1]:
                        x[j][m] = self.searchrange[1]
                    if x[j][m] < self.searchrange[0]:
                        x[j][m] = self.searchrange[0]


    def fun(self, x):
        result = 0
        for i in x:
            result = result + pow(i, 2)
        return result




