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
csvdir = os.path.join(os.getcwd(),"output/125_results/")
plotdir = os.path.join(os.getcwd(),"output/125_results/plots/")
if not os.path.exists(plotdir):
    os.makedirs(plotdir)

#Load the two CSVs
entropy = np.genfromtxt(csvdir + "Entropy.csv", delimiter=',')
hamiltonian = np.genfromtxt(csvdir + "Hamiltonian.csv", delimiter=',')

#Get a scale list for entropies, where 1 = max entropy, also make a colormap
max_entropy = max(entropy[:,3])
entropy_sizes = [(10*n/max_entropy)**2 for n in entropy[:,3]]
entropy_cmap = cm.rainbow(entropy[:,3])

#Get a scale list for the hamiltonians
max_hamiltonian = max(hamiltonian[:,3])
hamiltonian_sizes = [n/max_hamiltonian for n in hamiltonian[:,3]]
hamiltonian_cmap = cm.rainbow(hamiltonian[:,3])

#Plot the entropies with color and size based on value
fig = plt.figure(figsize=(6,6))
ax = fig.add_subplot(projection = '3d')
#ax.view_init(30,110,0)

ax.set_xlabel("$m_1$ value")
ax.set_ylabel("$m_2$ value")
ax.set_zlabel("$v_y$ value")

sc = ax.scatter(entropy[:,0], entropy[:,1], entropy[:,2], s=entropy_sizes,c = entropy[:,3], cmap = plt.get_cmap('cool'))
plt.colorbar(sc, fraction=0.036, pad = 0.1, label = "$S_{VN}$")
plt.savefig(plotdir + "Entropy_Plot_Sized.png")

#Plot the entropies with ONLY color based on value
fig = plt.figure(figsize=(6,6))
ax = fig.add_subplot(projection = '3d')
#ax.view_init(30,110,0)

ax.set_xlabel("$m_1$ value")
ax.set_ylabel("$m_2$ value")
ax.set_zlabel("$v_y$ value")

sc = ax.scatter(entropy[:,0], entropy[:,1], entropy[:,2], c = entropy[:,3], cmap = plt.get_cmap('cool'))
plt.colorbar(sc, fraction=0.036, pad = 0.1, label = "$S_{VN}$")
plt.savefig(plotdir + "Entropy_Plot_Unsized.png")

#Plot the entropies with color and size based on value on a 2D plot
m1Vals = np.unique(entropy[:,0])
for m in m1Vals:
    fig = plt.figure(figsize=(6,6))
    ax = fig.add_subplot()
    #ax.view_init(30,110,0)

    ax.set_xlabel("$m_2$ value")
    ax.set_ylabel("$v_y$ value")

    markersize = 50
    def update_prop(handle, orig):
        handle.update_from(orig)
        handle.set_sizes([markersize])

    sc = 0
    ptr = 0
    vals = entropy[(entropy[:,0] == m)]
    sizes = [(10*n/max_entropy)**2 for n in vals[:,3]]
    sc = ax.scatter(vals[:,1], vals[:,2], c = vals[:,3], cmap = plt.get_cmap('cool'))
    ptr += 1
    #ax.legend(title="$m_1$ Value",handler_map={type(sc): HandlerPathCollection(update_func=update_prop)})
    plt.colorbar(sc, fraction=0.046, pad = 0.1, label = "$S_{VN}$")
    plt.savefig(plotdir + "Entropy_Plot_m1_{}.png".format(str(m).replace('.','')))

#Plot the hamiltonians with color and size based on value
fig = plt.figure(figsize=(6,6))
ax = fig.add_subplot(projection = '3d')
#ax.view_init(30,110,0)

ax.set_xlabel("$m_1$ value")
ax.set_ylabel("$m_2$ value")
ax.set_zlabel("$v_y$ value")

sc = ax.scatter(hamiltonian[:,0], hamiltonian[:,1], hamiltonian[:,2], s=hamiltonian[:,3],c = entropy[:,3], cmap = plt.get_cmap('cool'))
plt.colorbar(sc, fraction=0.036, pad = 0.1, label = "$H_{x}$")
plt.savefig(plotdir + "Hamiltonian_Plot_Sized.png")

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

#Plot the hamiltonian with color and size based on value on a 2D plot
m1Vals = np.unique(hamiltonian[:,0])
for m in m1Vals:
    fig = plt.figure(figsize=(6,6))
    ax = fig.add_subplot()
    #ax.view_init(30,110,0)

    ax.set_xlabel("$m_2$ value")
    ax.set_ylabel("$v_y$ value")

    markersize = 50
    def update_prop(handle, orig):
        handle.update_from(orig)
        handle.set_sizes([markersize])

    sc = 0
    ptr = 0
    vals = hamiltonian[(hamiltonian[:,0] == m)]
    sizes = [(10*n/max_hamiltonian)**2 for n in vals[:,3]]
    sc = ax.scatter(vals[:,1], vals[:,2], c = vals[:,3], cmap = plt.get_cmap('cool'))
    ptr += 1
    #ax.legend(title="$m_1$ Value",handler_map={type(sc): HandlerPathCollection(update_func=update_prop)})
    plt.colorbar(sc, fraction=0.046, pad = 0.1, label = "$H_{x}$")
    plt.savefig(plotdir + "Hamiltonian_Plot_m1_{}.png".format(str(m).replace('.','')))
