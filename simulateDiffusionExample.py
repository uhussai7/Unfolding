import numpy as np
import simulateDiffusion
from coordinates import domainParams


#phi is the mapping from native to unfold dphi the derivative and phiInv the inverse
def phi(X,Y,Z):
    C=X+Y*1j
    w=1
    Cout=np.log((1 / 2) * (C + np.sqrt(C * C - 4 * w)))
    return np.real(Cout), np.imag(Cout), Z

def dphi(X,Y,Z):
    C=X+Y*1j
    w=1
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
    w = 1
    Cout = np.exp(C)
    Cout = Cout + w/Cout
    return np.real(Cout), np.imag(Cout), Z


#we have to specify the unfolded space params (this the domain of our
# coordinates)

Uparams = domainParams(0.3, 1.2, -np.pi / 2, np.pi / 2, 0, 1,#3*(1.2-0.3)/(50-1),
                       dims=[50, 50, 4])

# bvals = "K:\\Datasets\\HCP_diffusion\\101006\\Diffusion\\Diffusion\\bvals"
# bvecs = "K:\\Datasets\\HCP_diffusion\\101006\\Diffusion\\Diffusion\\bvecs"
bvals = "/home/uzair/PycharmProjects/Unfolding/data/101006/Diffusion/Diffusion/bvals"
bvecs = "/home/uzair/PycharmProjects/Unfolding/data/101006/Diffusion/Diffusion/bvecs"

L1 = 99.9E-4
L2 = 0.1E-4
L3 = 0.00
sim=simulateDiffusion.simulateDiffusion(phi,dphi,phiInv,Uparams,L1,L2,L3,bvals=bvals,bvecs=bvecs)
#sim.simulate('K:\\Datasets\\diffusionSimulations\\')
sim.simulate("/home/uzair/PycharmProjects/Unfolding/data/diffusionSimulations/")
sim.plotTangVecs()



