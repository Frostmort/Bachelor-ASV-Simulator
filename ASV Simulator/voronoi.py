#!/usr/bin/env python

from scipy.spatial import Voronoi, voronoi_plot_2d
import matplotlib.pyplot as plt
import numpy as np

from map import Map, Polygon


if __name__ == "__main__":
    mymap = Map('s1', gridsize=0.5,safety_region_length=4.0)

    points = mymap.get_obstacle_edges()
    #points = mymap.get_obstacle_points()
    vor = Voronoi(points)

    vedges = []
    for v in vor.vertices:
        if not mymap.is_occupied_discrete(v):
            vedges.append(v.tolist())
    fig = plt.figure()
    ax  = fig.add_subplot(111, autoscale_on=False)

    #ani = sim.animate(fig, ax)

    #voronoi_plot_2d(vor, ax)

    ax.axis('scaled')
    ax.set_xlim((-10, 160))
    ax.set_ylim((-10, 160))
    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')


    #    plt.tight_layout()

    v2 = np.array(vedges)

    for v in range(len(v2)):
        ax.annotate(str(v), xy=(v2[v,0], v2[v,1]), xytext=(v2[v,0], v2[v,1]))

    #ax.plot(v2[:,0], v2[:,1], '.')
    mymap.draw(ax, pcolor='g', ecolor='k')
    ax.grid()
    plt.show()
