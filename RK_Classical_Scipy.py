import numpy as np
import scipy.integrate as integrate
from scipy.integrate import solve_ivp
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

print("Number of Cores Available: ", multiprocessing.cpu_count())


#plt.style.use('seaborn') # I personally prefer seaborn for the graph style, but you may choose whichever you want.
params = {"ytick.color" : "black",
          "xtick.color" : "black",
          "axes.labelcolor" : "black",
          "axes.edgecolor" : "black",
          "text.usetex" : True,
          "font.family" : "serif",
          "font.serif" : ["Computer Modern Serif"]}
plt.rcParams.update(params)
plt.rcParams['savefig.dpi'] = 300
n = 1000
m = 1000
N = 10000

mvals = [0.5, 1, 1.5, 2, 2.5]
vyvals = [5,7.5, 10, 15, 20]

def RK4(update_func,
        x0=0,y0=0,vx=0,vy=0,
        t0 = 0, tf = 40, h = 0.01,
        m1 = 1, m2 = 1, lam=1,
        t_max=20,N_frames=201):
    #-----------------------
    #PARAMS
    # f: wavefunction
    # x0 and y0: Initial position
    # vx and vy: Initial velocity
    # n: Number of spatial points in each axis
    # left and right: start and stop of x axis
    # bottom and top: start and stop of y axis
    # N: number of time points to calculate
    # sigmax and sigmay: Initial wave spread
    # t_max: end time point to calculate to
    # mu: factor in potential
    # N_frames: number of frames of data to generate
    #-----------------------

    #Make array of frames
    N = int((tf-t0)/h)
    frames = np.linspace(t0,tf,N_frames).tolist()

    #Define matrices for x and y
    vals_out = np.zeros((4, N_frames), dtype = 'float64')
    cur_vals = np.array([x0, y0, vx*m1, vy*m2])

    vals_out[:,0] = cur_vals

    t = t0
    count = 0
    var = [m1,m2,lam]
    st = time.time()
    sol = solve_ivp(update_func, [t0, tf], cur_vals, method='Radau',
                    t_eval = frames, args = var)
    #print('************************************************************************')
    #t_cur = time.time()-st
    #elapsed = datetime.timedelta(seconds=round(t_cur))
    #elapsed_per_node = seconds=t_cur/n/m/N*1e6
    #print('time elapsed:',elapsed)
    #print('************************************************************************')
    return(sol.y)

def update_func(t, vals, m1,m2,lam):
    x = vals[0]
    y = vals[1]
    px = vals[2]
    py = vals[3]
    #Return x', y', px', py'
    return(np.array([px/m1, py/m2, -1*(m1*x + lam * np.exp(x-y)), lam*np.exp(x-y)]))

for val in tqdm(range(0,125)):
    m1 = mvals[int(val/25)]
    m2 = mvals[int((val%25)/5)]
    vy = vyvals[int(val%5)]
    lam = 1

    sol = RK4(update_func,
            x0=0,y0=20,vx=0,vy=-vy,
            t0 = 0, tf = 40, h = 0.0001,
            m1 = m1, m2 = m2, lam=lam,
            t_max=20,N_frames=401)

    #outdir = os.path.join(os.getcwd(),"output/test_results/Job{}_{}_{}_{}/".format(sys.argv[1], int(m1),int(m2),int(vy)))
    csvdir = os.path.join(os.getcwd(),"output/125_classical/CSVs/")
    UNIQUE_STRING = "mx{}_my{}_vy{}".format(m1,m2,vy)
    if not os.path.exists(csvdir):
        os.makedirs(csvdir)
    #np.save(outdir + "CNSResults_{:02}.npy".format(UNIQUE_STRING), np.array(sol[3]))


    #m1,m2,vy, VAL
    with open(csvdir + "Job{}.csv".format(val), 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile, delimiter = ",")
        csvwriter.writerow([m1,m2,vy,lam])
        csvwriter.writerows(sol.T)

    hamil = np.array((sol[2])**2/(2*m1) + m1*((sol[0])**2)/2)
    np.savetxt(csvdir + "Job{}_FullHamil.csv".format(val), hamil, delimiter='\n')
