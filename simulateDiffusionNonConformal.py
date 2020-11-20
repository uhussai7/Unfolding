import numpy as np
import simulateDiffusion
from coordinates import domainParams
import os
import matplotlib.pyplot as plt


# class simulationFuncion:
#     def __int__(self,phi,phiInv,dphi):


#phi is the mapping from native to unfold dphi the derivative and phiInv the inverse
def phi(X,Y,Z):
    cotphi = 1/np.tan(np.pi/3)
    cscphi = 1/np.sin(np.pi/3)
    U=X-Y*cotphi
    V=Y*cscphi
    return U, V, Z

def dphi(X,Y,Z):
    vx=np.cos(np.pi/3)
    vy=np.sin(np.pi/3)
    norm_v = np.sqrt(vx * vx + vy * vy)
    zeros = np.zeros(X.shape)
    ones = np.ones(X.shape)
    perturb = 1.1*np.sin(2*np.pi *X / 30)
    vx=ones*(vx+perturb)/norm_v
    vy=ones*vy/norm_v
    v1 = [-vy, vx , zeros]
    v2 = [vx, vy, zeros]
    v3 = [zeros, zeros, ones]
    return v1, v2, v3

def phiInv(U,V,W):
    X=U+V*np.cos(np.pi/3)
    Y=V*np.sin(np.pi/3)

    return X, Y, W

@np.vectorize
def L1L2L3(X,Y,Z):

    l1 = 0.1E-4
    l2 = 99.9E-4
    l3 = 0.00

    return l1,l2,l3



#we have to specify the unfolded space params (this the domain of our
# coordinates)
ress=['_nonConformal/']
Ns = [50]

for r in range(0,1):
    base="/home/uzair/PycharmProjects/Unfolding/data/diffusionSimulations"
    path=base+ress[r]
    if not os.path.exists(path):
        os.mkdir(path)
    uu=10
    u=0
    vv=10
    v=0
    N=Ns[r]
    delta=(uu-u)/(N-1)
    Uparams = domainParams(u, uu, v, vv, 0, delta,deltas=[delta,delta,delta])
    #evaluate deltas for native space
    #X, Y, Z = phiInv(Uparams.A, Uparams.B, Uparams.C)
    #Nx =Ns[r]#24 #this number is important it determines the size of native space
    #dx = (np.nanmax(X) - np.nanmin(X)) / (Nx - 1)
    #Uparams = domainParams(u, uu, v, vv, 0, 1*dx,deltas=[delta,delta,dx])



    # bvals = "K:\\Datasets\\HCP_diffusion\\101006\\Diffusion\\Diffusion\\bvals"
    # bvecs = "K:\\Datasets\\HCP_diffusion\\101006\\Diffusion\\Diffusion\\bvecs"
    bvals = "/home/uzair/PycharmProjects/Unfolding/data/101006/Diffusion/Diffusion/bvals"
    bvecs = "/home/uzair/PycharmProjects/Unfolding/data/101006/Diffusion/Diffusion/bvecs"

    L1 = 99.9E-4
    L2 = 0.1E-4
    L3 = 0.00



    sim=simulateDiffusion.simulateDiffusion(phi,dphi,phiInv,Uparams,L1L2L3,bvals=bvals,bvecs=bvecs,N0=N)
    #sim.simulate('K:\\Datasets\\diffusionSimulations\\')
    sim.simulate(path)
    #sim.plotTangVecs()

    #we have to make some masks for
    cortical_layers=np.digitize(sim.U_nii.get_fdata(),np.append(np.linspace(u,uu+1e-3,5),1000))
    radtang=np.digitize(sim.U_nii.get_fdata(),np.append(np.linspace(u,1.5+1e-3,2),1000))
    sixths= np.digitize(sim.V_nii.get_fdata(),np.append(np.linspace(v,vv+1e-3,7),1000))
    halfs= np.digitize(sim.V_nii.get_fdata(),np.append(np.linspace(v,vv+1e-3,3),1000))


