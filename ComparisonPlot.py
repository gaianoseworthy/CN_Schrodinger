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

for val in tqdm(range(0,125)):
    m1 = mvals[int(val/25)]
    m2 = mvals[int((val%25)/5)]
    vy = vyvals[int(val%5)]
    try:
        classical = np.genfromtxt(classicaldir + "Job{}.csv".format(val), delimiter=',')
        timeClassical = np.linspace(start = 0, stop = 40, num = len(classical))
        quantum = np.genfromtxt(quantumdir + "Job{}_AllXYPxPy.csv".format(val), delimiter=',')
        timeQuantum = np.linspace(start = 0, stop = 40, num = len(quantum))


        #Plot the hamiltonians with ONLY color based on value
        fig = plt.figure(figsize=(6,6))
        ax = fig.add_subplot()
        #ax.view_init(30,110,0)

        ax.set_xlabel("$<x>$")
        ax.set_ylabel("$<y>$")

        sc = ax.plot(classical[1:,0], classical[1:,1], label = "Classical")
        sc = ax.plot(quantum[:,2], quantum[:,3], label = "Quantum")
        ax.legend()
        fig.text(0.5,  0.01, "Params: m1={}, m2={}, vy={}".format(m1,m2,vy), ha='center')
        plt.savefig(plotdir + "Position{}.png".format(val, m1, m2, vy))
        plt.close()

        #Plot the hamiltonians with ONLY color based on value
        fig = plt.figure(figsize=(6,6))
        ax = fig.add_subplot()
        #ax.view_init(30,110,0)

        ax.set_xlabel("$Px$ value")
        ax.set_ylabel("$Py$ value")

        sc = ax.plot(classical[1:,2], classical[1:,3], label = "Classical")
        sc = ax.plot(quantum[:,4], quantum[:,5], label = "Quantum")
        ax.legend()
        fig.text(0.5,  0.01, "Params: m1={}, m2={}, vy={}".format(m1,m2,vy), ha='center')
        plt.savefig(plotdir + "Momentum{}.png".format(val, m1, m2, vy))
        plt.close()
    except OSError as e:
        print("Error: Job {} Not Found.".format(val))
