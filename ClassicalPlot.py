import numpy as np
import scipy.integrate as integrate
import scipy.sparse as sps
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
import time
import cmath
from pylab import meshgrid,cm,imshow,contour,clabel,colorbar,axis,title,show
from numpy import exp,arange
import datetime
import multiprocessing
import datetime
import os
import sys
import tracemalloc
import csv
from tqdm import tqdm

params = {"ytick.color" : "black",
          "xtick.color" : "black",
          "axes.labelcolor" : "black",
          "axes.edgecolor" : "black",
          "text.usetex" : True,
          "font.family" : "serif",
          "font.serif" : ["Computer Modern Serif"]}
plt.rcParams.update(params)
plt.rcParams['savefig.dpi'] = 300

#Set path to CSV files
csvdir = os.path.join(os.getcwd(),"output/125_classical/CSVs/")
plotdir = os.path.join(os.getcwd(),"output/125_classical/plots/")
if not os.path.exists(plotdir):
    os.makedirs(plotdir)

hamiltonian = np.zeros((125,4))
mvals = [0.5, 1, 2, 5, 10]
vyvals = [0.1, 0.5, 1, 2, 5]
for val in tqdm(range(0,125)):
    #Load the two CSVs
    job = np.genfromtxt(csvdir + "Job{}.csv".format(val), delimiter=',')
    m1 = job[0,0]
    m2 = job[0,1]
    vy = job[0,2]
    x = job[1:,0]
    y = job[1:,1]
    px = job[1:,2]
    py = job[1:,3]

    fig = plt.figure(figsize=(6,6))
    ax = fig.add_subplot()

    ax.set_xlabel("$x$ position")
    ax.set_ylabel("$y$ position")
    ax.set_xlim([-10,10])

    sc = ax.plot(x,y)
    plt.savefig(plotdir + "Job{}_Plot.png".format(val))
    plt.close()

    hamiltonian[val] = [m1,m2,vy,(px[-1])**2/(2*m1) + m1*((x[-1])**2)/2]

with open(csvdir + "Hamiltonians.csv".format(val), 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile, delimiter = ",")
        csvwriter.writerows(hamiltonian)

#Plot the hamiltonians with ONLY color based on value
fig = plt.figure(figsize=(6,6))
ax = fig.add_subplot(projection = '3d')
#ax.view_init(30,110,0)

ax.set_xlabel("$m_1$ value")
ax.set_ylabel("$m_2$ value")
ax.set_zlabel("$v_y$ value")

sc = ax.scatter(hamiltonian[:,0], hamiltonian[:,1], hamiltonian[:,2], c = hamiltonian[:,3], cmap = plt.get_cmap('cool'))
plt.colorbar(sc, fraction=0.036, pad = 0.1, label = "$H_{x}$")
plt.savefig(plotdir + "Hamiltonian_Plot_Unsized.png")
plt.close()

#Plot the hamiltonian with color and size based on value on a 2D plot
m1Vals = np.unique(hamiltonian[:,0])
for m in m1Vals:
    fig = plt.figure(figsize=(6,6))
    ax = fig.add_subplot()
    #ax.view_init(30,110,0)

    ax.set_xlabel("$v_y$ value")
    ax.set_ylabel("$H_{x}$ value")

    sc = 0
    vals = hamiltonian[(hamiltonian[:,0] == m)]
    m2Vals = np.unique(hamiltonian[:,1])
    for m2 in m2Vals:
        spec_vals = vals[vals[:,1] == m2]
        sc = ax.scatter(spec_vals[:,2], spec_vals[:,3], label = "$m_2=${}".format(m2))
        ax.plot(spec_vals[:,2], spec_vals[:,3])
    ax.legend()
    #plt.colorbar(sc, fraction=0.046, pad = 0.1, label = "$S_{VN}$")
    plt.savefig(plotdir + "Hamiltonian_Plot_m1_{}.png".format(str(m).replace('.','')))
    plt.close()
