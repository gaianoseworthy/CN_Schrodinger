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
m = 5000
N = 1000

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
    Z = f(X, Y, x0, y0, vx, vy, sigmax, sigmay)

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

def wavefunc(x,y, x0, y0, vx, vy, sigmax, sigmay):
    return (exp(-vx*1j*x)*exp(-vy*1j*y)*
            exp(-(y-y0)**2/4/sigmay**2)*
            (1/np.sqrt(2) * np.pi**(-1/4) * np.exp(-x**2/2)))

sol = CN(wavefunc,n=n,m=m,N=N,
         left=-5,right=5,bottom=-25,top=25,
         x0=0,y0=10,sigmax=1,sigmay=3,
         lam=1,m1=1,m2=1,vx=0,vy=1,t_max=100, N_frames = 401);

outdir = os.path.join(os.getcwd(),"output/")
UNIQUE_STRING = "InitYLarge_{}x{}x{}".format(n,m,N)
if not os.path.exists(outdir):
    os.makedirs(outdir)
np.save(outdir + "CNSResults_{}.npy".format(UNIQUE_STRING), np.array(sol[3]))

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
Px = (-1/(hx**2))*sps.diags([1, -2, 1], [-1, 0, 1], shape=(n, n))
HPx = (1/(2*1)) * (Px**2) + (1/2)*X2
psi = np.zeros((N+1,n,m),dtype='complex')
Z = np.zeros((N+1,m,n),dtype='float')

observables = np.zeros((N+1,4))

def F(x):
    if abs(x)>1e-14:
        return(-np.real(x)*np.log(np.real(x)))
    else:
        return(0)

for i in range (0,N+1):
    gnat = sol[3][i]
    gnat = np.reshape(gnat,(m,n),order='C')
    Z[i,:,:] = np.flipud(gv(gnat))
    psi[i,:,:] = np.transpose(gnat)
    rho1 = hx*hy*np.matmul(psi[i,:,:],psi[i,:,:].conj().T)
    rho2 = hx*hy*np.matmul(psi[i,:,:].T,psi[i,:,:].conj())
    observables[i,0] = T[i]
    observables[i,1] = np.real(np.trace(rho1))-1
    #observables[i,2] = np.real(np.trace(np.matmul(rho1,X)))
    #observables[i,3] = (np.real(np.trace(np.matmul(rho1,X2)))-observables[i,2]**2)**(0.5)
    #observables[i,4] = np.real(np.trace(np.matmul(rho2,Y)))
    #observables[i,5] = (np.real(np.trace(np.matmul(rho2,Y2)))-observables[i,4]**2)**(0.5)
    observables[i,2] = np.sum(np.vectorize(F)(np.linalg.eig(rho1)[0]))
    observables[i,3] = np.real(np.trace(np.matmul(HPx, rho1)))

Zmax = Z.max()

np.save(outdir + "CNSObservables_{}.npy".format(UNIQUE_STRING), observables)

fig_norm, ax_norm = plt.subplots()
ax_norm.plot(observables[:,0],observables[:,1])
plt.ylabel('Error in Wavefunction Normalization')
plt.xlabel('Time $t$')
plt.savefig(outdir + "Normalization_{}".format(UNIQUE_STRING))

fig_S, ax_S = plt.subplots()
ax_S.plot(observables[:,0],observables[:,2])
ax_S.set_ylabel('von Neumann entanglement entropy $S$')
ax_S.set_xlabel('Time $t$')
plt.savefig(outdir + "Entropy_{}".format(UNIQUE_STRING))

fig_S, ax_S = plt.subplots()
ax_S.plot(observables[:,0],observables[:,3])
ax_S.set_ylabel('Hamiltonian in $x$ coordinate, $<H>$')
ax_S.set_xlabel('Time $t$')
plt.savefig(outdir + "Hamiltonian_{}".format(UNIQUE_STRING))

def animate(frame):
    global Z, image
    image.set_array(Z[frame,:,:])
    return image,
fig, ax = plt.subplots()
image = ax.imshow(Z[0,:,:],cmap=cm.Purples,vmin=0,vmax=0.5*Zmax,extent=sol[4],aspect='auto');
#image = ax.imshow(Z[0,:,:],cmap=cm.plasma,norm=LogNorm(vmin=1e-10, vmax=1),extent=sol[4],aspect='auto');
bar = plt.colorbar(image)
bar.set_label('probability density $|\psi(t,x,y)|^2$')
plt.ylabel('Projectile Coordinate')
plt.xlabel('Oscillator Coordinate')
ani = animation.FuncAnimation(fig,animate,np.arange(0, N-1), blit=True,interval=10000/N);
ani.save(filename="output/PsiEvo_{}.mp4".format(UNIQUE_STRING), writer="ffmpeg")

#X,Y = np.meshgrid(x,y)
#Y = np.flip(Y)
#fig = plt.figure(figsize=(6,6))
#ax = fig.add_subplot(projection = '3d')
#ax.view_init(30,110,0)
#ax.plot_surface(X,Y,Z[0,:,:], cmap = cm.coolwarm, antialiased=False, linewidth=0)

#def animate3d(frame):
#    ax.cla()
#    ax.plot_surface(X,Y,Z[frame,:,:], rstride=1, cstride=1, cmap = cm.coolwarm, #antialiased=False, linewidth=0)
#    ax.set_zlim(0, 0.15)
#    return fig,
#bar.set_label('probability density $|\psi(t,x,y)|^2$')
#ax.set_ylabel('Projectile Coordinate')
#ax.set_xlabel('Oscillator Coordinate')
#ax.set_zlabel('Psi Amplitude $|\psi(t,x,y)|^2$')
#ani = animation.FuncAnimation(fig = fig, func = animate3d, frames = np.arange(0, #N-1),interval=10000/N, repeat=True);
#ani.save(filename="output/PsiEvo3D_{}.mp4".format(UNIQUE_STRING), writer="ffmpeg")
