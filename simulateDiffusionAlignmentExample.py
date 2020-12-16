import numpy as np
import simulateDiffusion
from coordinates import domainParams
import os
import matplotlib.pyplot as plt
import nibabel as nib


#here we want to gererate some volumes to cover the parameter space of possible fiber states.
#We will vary the following parameters
#1) The curvature of tangential fibres
#2) The resolution of the image
#3) The radial coordinate where we transition from tangential to radial

#We have to bring all steaps under one file. The steps are:
#0) Define the parameter space
#1) Generate the native diffusion volumes
#2) Unfold them
#3) perform tractography on them



def L1L2L3_drt_w_scale(drt,w,scale):
    def wrap(func):
        def inner(X,Y,Z,drt=drt,w=w,scale=scale):
            return func(X,Y,Z,drt=drt,w=w,scale=scale)
        return inner
    return wrap

def change_w_scale(w,scale,beta):
    def wrap(func):
        def inner(X,Y,Z,w=w,scale=scale,beta=beta):
            return func(X,Y,Z,w=w,scale=scale,beta=beta)
        return inner
    return wrap

    #Parameters to vary are drt (rad tang transistion) and resolution
scale=100
#res=[1.75,1.5,1.25,1.00,0.75]
res=[1.25]
res=np.asarray(res)
res=scale*res/100
#drt=np.linspace(0.1,0.25,5)
drt=[1.5]
#w=np.linspace(0.9,0.99,4)
w=[0.99]
beta=np.linspace(0,np.pi/4,5)
for i in range(0,len(res)):
    print(i)
    for j in range(0,len(drt)):
        for k in range(0,len(w)):
            for b in range(0,len(beta)):
            #phi is the mapping from native to unfold dphi the derivative and phiInv the inverse
                @change_w_scale(w=w[k],scale=scale,beta=None)
                def phi(X,Y,Z,w=None,scale=None):
                    if scale is None: scale=5
                    if w is None: w=1
                    C=X+Y*1j
                    A=C/scale + w+1
                    Cout=np.log(0.5*(A+np.sqrt(A*A-4*w)))
                    return np.real(Cout), np.imag(Cout), Z

                @change_w_scale(w=w[k], scale=scale,beta=beta)
                def dphi(X,Y,Z,w=None, scale=None, beta=None):
                    if scale is None: scale = 5
                    if w is None: w=1
                    if beta in None: beta=0
                    C=X+Y*1j
                    A = C / scale + w +1
                    dCout= 1/np.sqrt(-4 * w + A*A)*(1/scale)
                    norm = np.sqrt(np.real(dCout)*np.real(dCout) +np.imag(dCout)*np.imag(dCout))
                    zeros = np.zeros(X.shape)
                    ones = np.ones(X.shape)
                    v1 = [np.imag(dCout) / norm, np.real(dCout) / norm, zeros]
                    v2 = [v1[1], -v1[0], zeros]
                    v3 = [zeros, zeros, ones]
                    return v1, v2, v3

                @change_w_scale(w=w[k], scale=scale)
                def phiInv(U,V,W,w=None,scale=None):
                    if scale is None: scale = 5
                    if w is None: w=1
                    C = U + V * 1j
                    Cout = np.exp(C)
                    result = scale*(Cout-1 + w*(1/Cout-1))
                    return np.real(result), np.imag(result), W

                @L1L2L3_drt_w_scale(drt=drt[j],w=w[k],scale=scale)
                @np.vectorize
                def L1L2L3(X,Y,Z,w=None,scale=None,drt=None):
                    if scale is None: scale = 5
                    #if drt is None: drt=1.5 #drt is the radial coordinate where we transition from tang to rad
                    if w is None: w=1
                    l1 = 99.9E-4
                    l2 = 0.1E-4
                    l3 = 0.00

                    C = X + Y * 1j
                    A = C / scale + w+1
                    Cout = np.log(0.5*(A+np.sqrt(A*A-4*w)))
                    U= np.real(Cout)

                    if U<drt:
                        L1=l1
                        L2=l2
                        L3=l3
                    else:
                         L1 = l2
                         L2 = l1
                         L3 = l3
                    return L1,L2,L3


                Uparams=domainParams(0,0.3,-np.pi/6,np.pi/6,0,1,deltas=[0.05,0.05,0.05])

                X, Y, Z = phiInv(Uparams.A,Uparams.B,Uparams.C)


                uu=0.3
                u=0
                vv=np.pi / 6
                v=-np.pi / 6
                N=50
                delta=(uu-u)/(N-1)
                L1 = 99.9E-4
                L2 = 0.1E-4
                L3 = 0.00


                Uparams = domainParams(u, uu, v, vv, 0, 1,deltas=[delta,delta,1])
                #evaluate deltas for native space
                X, Y, Z = phiInv(Uparams.A, Uparams.B, Uparams.C)
                Nx = (np.nanmax(X) - np.nanmin(X)) /res[i] + 1
                Uparams = domainParams(u, uu, v, vv, 0, 1*res[i],deltas=[delta,delta,res[i]])
                base="/home/uzair/PycharmProjects/Unfolding/data/diffusionSimulationsAlignment_res-"
                path=base+str(int(res[i]*10000))+"mm_drt-"+str(int(drt[j]*100))+"_w-"+str(int(w[k]*100))+"_beta-"+"/"
                if not os.path.exists(path):
                    os.mkdir(path)

                bvals = "/home/uzair/PycharmProjects/Unfolding/data/101006/Diffusion/Diffusion/bvals"
                bvecs = "/home/uzair/PycharmProjects/Unfolding/data/101006/Diffusion/Diffusion/bvecs"

                sim=simulateDiffusion.simulateDiffusion(phi,dphi,phiInv,Uparams,L1L2L3,bvals=bvals,bvecs=bvecs,N0=Nx)
                sim.simulate(path)




