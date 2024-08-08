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

#SET WHICH JOB TO TAKE

#Set path to CSV files
classicaldir = os.path.join(os.getcwd(),"output/125_classical/CSVs/")
quantumdir = os.path.join(os.getcwd(),"output/125_quantum/CSVs/")
plotdir = os.path.join(os.getcwd(),"output/125_comparison/Special_Comparison/")
if not os.path.exists(plotdir):
    os.makedirs(plotdir)

mvals = [0.5, 1, 1.5, 2, 2.5]
vyvals = [5,7.5, 10, 15, 20]

for val in range(0,125):
    m1 = mvals[int(val/25)]
    m2 = mvals[int((val%25)/5)]
    vy = vyvals[int(val%5)]
    try:
        hamiltonianClassical = np.genfromtxt(classicaldir + "Job{}_FullHamil.csv".format(val), delimiter=',')
        timeClassical = np.linspace(start = 0, stop = 40, num = len(hamiltonianClassical))
        hamiltonianQuantum = np.genfromtxt(quantumdir + "Job{}_AllHE.csv".format(val), delimiter=',')
        hamiltonianQuantum[:,0] -= 0.5
        timeQuantum = np.linspace(start = 0, stop = 40, num = len(hamiltonianQuantum))


        #Plot the hamiltonians with ONLY color based on value
        fig = plt.figure(figsize=(6,6))
        ax = fig.add_subplot()
        #ax.view_init(30,110,0)

        ax.set_xlabel("Time Step")
        ax.set_ylabel("$m_2$ value")

        sc = ax.plot(timeClassical, hamiltonianClassical, label = "Classical")
        sc = ax.plot(timeQuantum, hamiltonianQuantum[:,0], label = "Quantum - 0.5")
        ax.legend()
        fig.text(0.5,  0.01, "Params: m1={}, m2={}, vy={}".format(m1,m2,vy), ha='center')
        plt.savefig(plotdir + "Job{}.png".format(val, m1, m2, vy))
        plt.close()
    except OSError as e:
        print("Error: Job {} Not Found.".format(val))
