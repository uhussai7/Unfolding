import numpy as np
import simulateDiffusion
from coordinates import domainParams
import os
import matplotlib.pyplot as plt
import nibabel as nib

# class simulationFuncion:
#     def __int__(self,phi,phiInv,dphi):


#phi is the mapping from native to unfold dphi the derivative and phiInv the inverse
def phi(X,Y,Z):
    C=X+Y*1j
    w = 10
    Cout=np.log((1 / 2) * (C + np.sqrt(C * C - 4 * w)))
    return np.real(Cout), np.imag(Cout), Z

def dphi(X,Y,Z):
    C=X+Y*1j
    w = 10
    dCout= 1/np.sqrt(-4 * w + C*C)
    norm = np.sqrt(np.real(dCout)*np.real(dCout) +np.imag(dCout)*np.imag(dCout))
    zeros = np.zeros(X.shape)
    ones = np.ones(X.shape)
    v1 = [np.imag(dCout) / norm, np.real(dCout) / norm, zeros]
    v2 = [v1[1], -v1[0], zeros]
    v3 = [zeros, zeros, ones]
    return v1, v2, v3

def phiInv(X,Y,Z):
    C = X + Y * 1j
    w = 10
    Cout = np.exp(C)
    Cout = Cout + w/Cout
    return np.real(Cout), np.imag(Cout), Z

@np.vectorize
def L1L2L3(X,Y,Z):
    l1 = 99.9E-4
    l2 = 0.1E-4
    l3 = 0.00
    C=X+Y*1j
    w = 10
    Cout = np.log((1 / 2) * (C + np.sqrt(C * C - 4 * w)))
    U=np.real(Cout)
    V= np.imag(Cout)
    W= Z
    if U<1.3:
        L1=l1
        L2=l2
        L3=l3
    else:
         L1 = l2
         L2 = l1
         L3 = l3
    return L1,L2,L3



#we have to specify the unfolded space params (this the domain of our
# coordinates)
ress=['close_high/', 'close_medium/', 'close_low/']
Ns = [40,30,20]

for r in range(0,3):
    base="/home/uzair/PycharmProjects/Unfolding/data/diffusionSimulations"
    path=base+ress[r]
    if not os.path.exists(path):
        os.mkdir(path)
    uu=1.75
    u=1.17
    vv=np.pi / 5
    v=-np.pi / 5
    N=50
    delta=(uu-u)/(N-1)
    Uparams = domainParams(u, uu, v, vv, 0, 1,deltas=[delta,delta,1])
    #evaluate deltas for native space
    X, Y, Z = phiInv(Uparams.A, Uparams.B, Uparams.C)
    Nx =Ns[r]#24 #this number is important it determines the size of native space
    dx = (np.nanmax(X) - np.nanmin(X)) / (Nx - 1)
    Uparams = domainParams(u, uu, v, vv, 0, 1*dx,deltas=[delta,delta,dx])



    # bvals = "K:\\Datasets\\HCP_diffusion\\101006\\Diffusion\\Diffusion\\bvals"
    # bvecs = "K:\\Datasets\\HCP_diffusion\\101006\\Diffusion\\Diffusion\\bvecs"
    bvals = "/home/uzair/PycharmProjects/Unfolding/data/101006/Diffusion/Diffusion/bvals"
    bvecs = "/home/uzair/PycharmProjects/Unfolding/data/101006/Diffusion/Diffusion/bvecs"

    L1 = 99.9E-4
    L2 = 0.1E-4
    L3 = 0.00



    sim=simulateDiffusion.simulateDiffusion(phi,dphi,phiInv,Uparams,L1L2L3,bvals=bvals,bvecs=bvecs,N0=Nx)
    #sim.simulate('K:\\Datasets\\diffusionSimulations\\')
    sim.simulate(path)
    #sim.plotTangVecs()

    #we have to make some masks for
    cortical_layers_nii=nib.Nifti1Image(np.digitize(sim.U_nii.get_fdata(),
                                np.append(np.linspace(u,uu+1e-3,5),
                                1000)),sim.U_nii.affine)
    radtang_nii=nib.Nifti1Image(np.digitize(sim.U_nii.get_fdata(),
                        np.append(np.linspace(u,1.5+1e-3,2),
                        1000)),sim.U_nii.affine)
    sixths_nii=nib.Nifti1Image(np.digitize(sim.V_nii.get_fdata(),
                        np.append(np.linspace(v,vv+1e-3,7),
                        1000)),sim.U_nii.affine)
    halfs_nii= nib.Nifti1Image(np.digitize(sim.V_nii.get_fdata(),
                       np.append(np.linspace(v,vv+1e-3,3),
                        1000)),sim.U_nii.affine)

    mask_names=['cortical_layers.nii.gz', 'radtang.nii.gz', 'sixths.nii.gz', 'halfs.nii.gz']
    masks=[cortical_layers_nii, radtang_nii, sixths_nii, halfs_nii]
    for m in range(0,4):
        nib.save(masks[m],path+mask_names[m])



