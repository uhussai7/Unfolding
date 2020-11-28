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

def change_w_scale(w,scale):
    def wrap(func):
        def inner(X,Y,Z,w=w,scale=scale):
            return func(X,Y,Z,w=w,scale=scale)
        return inner
    return wrap

    #Parameters to vary are drt (rad tang transistion) and resolution
res=[1.75,1.5,1.25,1.00,0.75]
drt=np.linspace(0.1,0.25,5)
w=np.linspace(0.9,0.99,4)
scale=50
for i in range(0,len(res)):
    print(i)
    for j in range(0,len(drt)):
        for k in range(0,len(w)):
            #phi is the mapping from native to unfold dphi the derivative and phiInv the inverse
            @change_w_scale(w=w[k],scale=scale)
            def phi(X,Y,Z,w=None,scale=None):
                if scale is None: scale=5
                if w is None: w=1
                C=X+Y*1j
                A=C/scale + w+1
                Cout=np.log(0.5*(A+np.sqrt(A*A-4*w)))
                return np.real(Cout), np.imag(Cout), Z

            @change_w_scale(w=w[k], scale=scale)
            def dphi(X,Y,Z,w=None, scale=None):
                if scale is None: scale = 5
                if w is None: w=1
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
            base="/home/uzair/PycharmProjects/Unfolding/data/diffusionSimulations_res-"
            path=base+str(int(res[i]*100))+"mm_drt-"+str(int(drt[j]*100))+"+w-"+str(int(w[k]*100))+"/"
            if not os.path.exists(path):
                os.mkdir(path)

            bvals = "/home/uzair/PycharmProjects/Unfolding/data/101006/Diffusion/Diffusion/bvals"
            bvecs = "/home/uzair/PycharmProjects/Unfolding/data/101006/Diffusion/Diffusion/bvecs"

            sim=simulateDiffusion.simulateDiffusion(phi,dphi,phiInv,Uparams,L1L2L3,bvals=bvals,bvecs=bvecs,N0=Nx)
            sim.simulate(path)

# #lets do some visualization to make sure the tranformation are working as intended
# #Nparams=domainParams(0,1,0,1,0,1,deltas=[0.1,0.1,0.1])
# fig= plt.figure()
# axes=plt.gca()
# axes.set_xlim([X.min(),X.max()])
# axes.set_ylim([Y.min(),Y.max()])
# axes.axis('equal')
# for u in range(0,X.shape[0]):
#     ux_const = X[u, :, 0]
#     uy_const = Y[u, :, 0]
#     for v in range(0, X.shape[1]):
#         vx_const = X[:, v, 0]
#         vy_const = Y[:, v, 0]
#         plt.plot(ux_const,uy_const,color='blue')
#         if (u>0.5*(X.shape[0])):
#             plt.plot(vx_const, vy_const,color='red')



# #Parameters to vary are drt (rad tang transistion) and resolution
# res=[0.2,.175,.15,.125,.100]
# #drt=[1.0,1.2,1.4,1.6]
# drt=[1.31]
#
# for i in range(0,len(res)):
#     for j in range(0,len(drt)):

#
#
#         @L1L2L3_drt(drt=1.6)
#         @np.vectorize
#         def L1L2L3(X, Y, Z, w=None, scale=None, drt=None):
#             if scale is None: scale = 1
#             if drt is None: drt = 1.5  # drt is the radial coordinate where we transition from tang to rad
#             if w is None: w = 1
#             l1 = 99.9E-4
#             l2 = 0.1E-4
#             l3 = 0.00
#
#             C = X + Y * 1j
#             A = C / scale + w - 1
#             dCout = 1 / np.sqrt(-4 * w + A * A) * (1 / scale)
#             U = np.real(dCout)
#
#             if U < drt:
#                 L1 = l1
#                 L2 = l2
#                 L3 = l3
#             else:
#                 L1 = l2
#                 L2 = l1
#                 L3 = l3
#             return L1, L2, L3
#
#         sim=simulateDiffusion.simulateDiffusion(phi,dphi,phiInv,Uparams,L1L2L3,bvals=bvals,bvecs=bvecs,N0=Nx)
#         sim.simulate(path=path)



# #we have to specify the unfolded space params (this the domain of our
# # coordinates)
# ress=['close_high/', 'close_medium/', 'close_low/']
# Ns = [40,30,20]
#
# for r in range(0,3):
#     base="/home/uzair/PycharmProjects/Unfolding/data/diffusionSimulations"
#     path=base+ress[r]
#     if not os.path.exists(path):
#         os.mkdir(path)
#     uu=1.75
#     u=1.17
#     vv=np.pi / 5
#     v=-np.pi / 5
#     N=50
#     delta=(uu-u)/(N-1)
#     Uparams = domainParams(u, uu, v, vv, 0, 1,deltas=[delta,delta,1])
#     #evaluate deltas for native space
#     X, Y, Z = phiInv(Uparams.A, Uparams.B, Uparams.C)
#     Nx =Ns[r]#24 #this number is important it determines the size of native space
#     dx = (np.nanmax(X) - np.nanmin(X)) / (Nx - 1)
#     Uparams = domainParams(u, uu, v, vv, 0, 1*dx,deltas=[delta,delta,dx])
#
#
#
#     # bvals = "K:\\Datasets\\HCP_diffusion\\101006\\Diffusion\\Diffusion\\bvals"
#     # bvecs = "K:\\Datasets\\HCP_diffusion\\101006\\Diffusion\\Diffusion\\bvecs"
#     bvals = "/home/uzair/PycharmProjects/Unfolding/data/101006/Diffusion/Diffusion/bvals"
#     bvecs = "/home/uzair/PycharmProjects/Unfolding/data/101006/Diffusion/Diffusion/bvecs"
#
#     L1 = 99.9E-4
#     L2 = 0.1E-4
#     L3 = 0.00
#
#
#
#     sim=simulateDiffusion.simulateDiffusion(phi,dphi,phiInv,Uparams,L1L2L3,bvals=bvals,bvecs=bvecs,N0=Nx)
#     #sim.simulate('K:\\Datasets\\diffusionSimulations\\')
#     sim.simulate(path)
#     #sim.plotTangVecs()
#
#     #we have to make some masks for
#     cortical_layers_nii=nib.Nifti1Image(np.digitize(sim.U_nii.get_fdata(),
#                                 np.append(np.linspace(u,uu+1e-3,5),
#                                 1000)),sim.U_nii.affine)
#     radtang_nii=nib.Nifti1Image(np.digitize(sim.U_nii.get_fdata(),
#                         np.append(np.linspace(u,1.5+1e-3,2),
#                         1000)),sim.U_nii.affine)
#     sixths_nii=nib.Nifti1Image(np.digitize(sim.V_nii.get_fdata(),
#                         np.append(np.linspace(v,vv+1e-3,7),
#                         1000)),sim.U_nii.affine)
#     halfs_nii= nib.Nifti1Image(np.digitize(sim.V_nii.get_fdata(),
#                        np.append(np.linspace(v,vv+1e-3,3),
#                         1000)),sim.U_nii.affine)
#
#     mask_names=['cortical_layers.nii.gz', 'radtang.nii.gz', 'sixths.nii.gz', 'halfs.nii.gz']
#     masks=[cortical_layers_nii, radtang_nii, sixths_nii, halfs_nii]
#     for m in range(0,4):
#         nib.save(masks[m],path+mask_names[m])



