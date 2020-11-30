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
    return U/1, V/1, Z

def dphi(X,Y,Z):
    vx=np.cos(np.pi/3)
    vy=np.sin(np.pi/3)
    norm_v = np.sqrt(vx * vx + vy * vy)
    zeros = np.zeros(X.shape)
    ones = np.ones(X.shape)
    perturb = 0.0*np.sin(2*np.pi *X / 3000)
    vx=ones*(vx+perturb)/norm_v
    vy=ones*vy/norm_v
    v1 = [-vy, vx , zeros]
    v2 = [vx, vy, zeros]
    v3 = [zeros, zeros, ones]
    return v1, v2, v3

def phiInv(U,V,W):
    U=U
    V=V
    X=U+V*np.cos(np.pi/3)
    Y=V*np.sin(np.pi/3)

    return 1*X, 1*Y, W

@np.vectorize
def L1L2L3(X,Y,Z):

    l1 = 0.1E-4
    l2 = 99.9E-4
    l3 = 0.00

    return l1,l2,l3



#we have to specify the unfolded space params (this the domain of our
# coordinates)
ress=['_nonConformal/']
Ns = [25]

r=0
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
X,Y,Z = phiInv(Uparams.A,Uparams.B,Uparams.C)
Uparams = domainParams(u, uu, v, vv, 0, (X.max()-X.min())/(N-1),deltas=[delta,delta,delta])



bvals = "/home/uzair/PycharmProjects/Unfolding/data/101006/Diffusion/Diffusion/bvals"
bvecs = "/home/uzair/PycharmProjects/Unfolding/data/101006/Diffusion/Diffusion/bvecs"


sim=simulateDiffusion.simulateDiffusion(phi,dphi,phiInv,Uparams,L1L2L3,bvals=bvals,bvecs=bvecs,N0=N)
sim.simulate(path)
