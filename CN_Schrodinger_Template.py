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
n = 100
m = 1000
N = 1000

mvals = [0.5, 1, 2, 5, 10]
vyvals = [0.1, 0.5, 1, 2, 5]
# Set up parameters based on which simulation number this is
val = int(sys.argv[1])
m1 = mvals[int(val/25)]
m2 = mvals[int((val%25)/5)]
vy = vyvals[int(val%5)]

#Define the crank nicolson method
def CN(f, x0=0,y0=0,vx=0,vy=0,n=10,m=10,left=0,right=2,
       top=1,bottom=-1,N=500,sigmax=0.1,sigmay=0.1, lam = 1, m1 = 1, m2 = 1,
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
    frames = np.linspace(0,N,N_frames,dtype='int').tolist()

    #Vectorize the frames for calculation
    s = t_max/N
    T = np.vectorize(lambda t : t*s)(frames)
    k = n*m

    #Define matrices for x and y
    x = np.linspace(left,right,n+2)[1:n+1]
    y = np.linspace(bottom,top,m+2)[1:m+1]
    hx = x[1]-x[0]
    hy = y[1]-y[0]

    #initial data
    X,Y = meshgrid(x, y)
    Z = f(X, Y, x0, y0, vx, vy, sigmax, sigmay, m1, m2)

    #Define a basic form of psi for calculating and another for animating
    psi = np.zeros((1,k),dtype='complex')
    psi_out = np.zeros((N_frames,k),dtype='complex')

    #Reshape initial data
    psi[0,:] = np.reshape(Z,k,order='C')

    #Normalize
    norm = hx*hy*np.vdot(psi[0,:],psi[0,:])
    psi[0,:] = norm**(-0.5)*psi[0,:]
    psi_out[0,:] = psi[0,:]

    #Define x and y matrix initial tridiagonal setup
    Px2 = (-1/hx**2)*sps.diags([1, -2, 1], [-1, 0, 1], shape=(n, n))
    Py2 = (-1/hy**2)*sps.diags([1, -2, 1], [-1, 0, 1], shape=(m, m))
    X2 = sps.diags(np.vectorize(lambda x : x**2)(x))
    Ux = sps.diags(np.vectorize(lambda x : np.exp(x))(x))
    Uy = sps.diags(np.vectorize(lambda y : np.exp(-y))(y))

    #Build the A matrix
    A = -1j *(1/(2*m1) * sps.kron(sps.identity(m),Px2)
            + 1/(2*m2) * sps.kron(Py2,sps.identity(n))
            + (m1/2) *sps.kron(sps.identity(m),X2)
            + lam*sps.kron(Uy, Ux))
    PNext = sps.identity(k)-(s*A/2)
    PPrev = sps.identity(k)+(s*A/2)
    COUNT = 0

    st = time.time()
    print("Starting...")
    #Loop through time steps
    for i in range(1,N+1):
        b = PPrev.dot(psi[0,:])
        psi[0,:] = sps.linalg.spsolve(PNext,b)
        if i in frames:
            COUNT = COUNT + 1
            psi_out[COUNT,:] = psi[0,:]
            elapsed = datetime.timedelta(seconds=round(time.time()-st))
            remaining = datetime.timedelta(seconds=round((time.time()-st)/i*(N-i)))
            print(round(i/N*100),'% ',elapsed,'elapsed',remaining,'remaining',end='\r')
    print('************************************************************************')
    t = time.time()-st
    elapsed = datetime.timedelta(seconds=round(t))
    elapsed_per_node = seconds=t/n/m/N*1e6
    print('time elapsed:',elapsed)
    print('time per node (microseconds):',round(elapsed_per_node,4))
    print('************************************************************************')
    return([T,x,y,psi_out,[left,right,bottom,top]])

#Set up the wavefunction
def wavefunc(x,y, x0, y0, vx, vy, sigmax, sigmay, m1, m2):
    return (exp(-vx*1j*x)*exp(-vy*1j*y)*
            exp(-(y-y0)**2/4/sigmay**2)*
            (1/np.sqrt(2) * (m1/np.pi)**(1/4) * np.exp(-m1 * x**2/2)))

#Calculate the solution from the CN method
sol = CN(wavefunc,n=n,m=m,N=N,
         left=-5,right=5,bottom=-5,top=95,
         x0=0,y0=20,sigmax=1,sigmay=3,
         lam=1,m1=m1,m2=m2,vx=0,vy=vy,t_max=40, N_frames = 401)

#outdir = os.path.join(os.getcwd(),"output/test_results/Job{}_{}_{}_{}/".format(sys.argv[1], int(m1),int(m2),int(vy)))

#Set up the location to save CSV files
csvdir = os.path.join(os.getcwd(),"output/125_results/CSVs/")
UNIQUE_STRING = "Test_{}x{}x{}".format(n,m,N)
if not os.path.exists(csvdir):
    os.makedirs(csvdir)
#np.save(outdir + "CNSResults_{:02}.npy".format(UNIQUE_STRING), np.array(sol[3]))

#Set up data for calculating hamiltonians, entropies, etc
T = sol[0]
x = sol[1]
y = sol[2]
n = np.shape(x)[0]
m = np.shape(y)[0]
N = np.shape(sol[3])[0]-1
hx = x[1]-x[0]
hy = y[1]-y[0]
def g(x):
    return(abs(x)**2)
    #return(x.real)
gv = np.vectorize(g)

X = np.diag(x)
X2 = np.diag(np.vectorize(lambda x : x**2)(x))
Y = np.diag(y)
Y2 = np.diag(np.vectorize(lambda x : x**2)(y))
Px2 = (-1/(hx**2))*sps.diags([1, -2, 1], [-1, 0, 1], shape=(n, n)).toarray()
HPx = (1/(2*m1)) * (Px2) + (m1/2)*X2
Px = (-1/(2*hx))*sps.diags([-1, 0, 1], [-1, 0, 1], shape=(n, n)).toarray()
Py2 = (1/(2*m2))*(-1/hy**2)*sps.diags([1, -2, 1], [-1, 0, 1], shape=(m, m)).toarray()
Py = (-1/(2*hx))*sps.diags([-1, 0, 1], [-1, 0, 1], shape=(m, m)).toarray()
psi = np.zeros((N+1,n,m),dtype='complex')
Z = np.zeros((N+1,m,n),dtype='float')

observables = np.zeros((N+1,11))

#I'm going to be honest, I don't know what this does
def F(x):
    if abs(x)>1e-14:
        return(-np.real(x)*np.log(np.real(x)))
    else:
        return(0)

#Calculate useful values
for i in range (0,N+1):
    gnat = sol[3][i]
    gnat = np.reshape(gnat,(m,n),order='C')
    Z[i,:,:] = np.flipud(gv(gnat))
    psi[i,:,:] = np.transpose(gnat)
    rho1 = hx*hy*np.matmul(psi[i,:,:],psi[i,:,:].conj().T)
    rho2 = hx*hy*np.matmul(psi[i,:,:].T,psi[i,:,:].conj())
    # Time
    observables[i,0] = T[i]
    # Error
    observables[i,1] = np.real(np.trace(rho1))-1
    # Von Neumann Entanglement Entropy
    observables[i,2] = np.sum(np.vectorize(F)(np.linalg.eig(rho1)[0]))
    # First Hamiltonian
    observables[i,3] = np.real(np.trace(np.matmul(HPx, rho1)))
    # Second Hamiltonian
    observables[i,4] = np.real(np.trace(np.matmul(Py2, rho2)))
    # <x>
    observables[i,5] =  np.real(np.trace(np.matmul(rho1,X)))
    # Var(x)
    observables[i,6] = (np.real(np.trace(np.matmul(rho1,X2)))-observables[i,5]**2)**(0.5)
    # <y>
    observables[i,7] = np.real(np.trace(np.matmul(rho2,Y)))
    # Var(y)
    observables[i,8] = (np.real(np.trace(np.matmul(rho2,Y2)))-observables[i,4]**2)**(0.5)
    # Px
    observables[i,9] = np.imag(np.trace(np.matmul(rho1,Px)))
    # Py
    observables[i,10] = (-1)*np.imag(np.trace(np.matmul(rho2, Py)))

Zmax = Z.max()

#Write m1, m2, vy, hamiltonian, and entropy to CSV file
with open(csvdir + "Job{}.csv".format(sys.argv[1]), 'w', newline='') as csvfile:
    csvwriter = csv.writer(csvfile, delimiter = ",")
    csvwriter.writerows([[m1,m2,vy, observables[-1,3], observables[-1,2]]])
