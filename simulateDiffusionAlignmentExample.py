import numpy as np
import simulateDiffusion
from coordinates import domainParams
import os
import matplotlib.pyplot as plt
import nibabel as nib
import unfoldSubject
import shutil
from dipy.io.image import load_nifti, save_nifti
from dipy.io.gradients import read_bvals_bvecs
from dipy.core.gradients import gradient_table
import dipy.reconst.dti as dti
from dipy.tracking import utils
from dipy.tracking.local_tracking import LocalTracking
from dipy.tracking.streamline import Streamlines
from dipy.tracking.stopping_criterion import (ThresholdStoppingCriterion, BinaryStoppingCriterion)
from dipy.direction import peaks_from_model
from dipy.data import default_sphere
from dipy.reconst.shm import CsaOdfModel
import copy
import coordinates
from unfoldTracking import tracking
from unfoldTracking import unfoldStreamlines
from coordinates import getPointsData
from dipy.tracking.utils import length
from dipy.io import streamline
import sys
import cmath


class tractographyExperiments:
    def __init__(self,npath,upath):
        self.npath=npath
        self.upath=upath
        self.Uparams=[]


    def simulate(self,w,drt,scale,res,beta):

        def L1L2L3_drt_w_scale(drt,w,scale,beta):
            def wrap(func):
                def inner(X,Y,Z,drt=drt,w=w,scale=scale,beta=beta):
                    return func(X,Y,Z,drt=drt,w=w,scale=scale,beta=beta)
                return inner
            return wrap

        def change_w_scale(w,scale,beta):
            def wrap(func):
                def inner(X,Y,Z,w=w,scale=scale,beta=beta):
                    return func(X,Y,Z,w=w,scale=scale,beta=beta)
                return inner
            return wrap


        @change_w_scale(w=w,scale=scale,beta=None)
        def phi(X,Y,Z,w=None,scale=None,beta=None):
            if scale is None: scale=5
            if w is None: w=1
            C=X+Y*1j
            #A=C/scale + w+1
            #Cout=np.log(0.5*(A+np.sqrt(A*A-4*w)))
            #A=C/scale
            #Cout=np.log(0.5*(A+np.sqrt(A*A-4*w)))
            Cout=np.power(C,1./w)
            return np.real(Cout), np.imag(Cout), Z

        @change_w_scale(w=w, scale=scale,beta=None)
        def phiInv(U,V,W,w=None,scale=None,beta=None):
            if scale is None: scale = 5
            if w is None: w=1
            C = U + V * 1j
            #Cout = np.exp(C)
            #result = scale*(Cout-1 + w*(1/Cout-1))
            #result = scale*(Cout+ w*(1/Cout))
            result = np.power(C,w)
            return np.real(result), np.imag(result), W


        @change_w_scale(w=w, scale=scale,beta=beta)
        def dphi(X,Y,Z,w=None, scale=None, beta=None):
            if scale is None: scale = 5
            if w is None: w=1
            if beta is None: beta=0
            #rot=cmath.rect(1,beta)
            C=X+Y*1j
            #C=C*rot
            # A = C / scale + w +1
            # dCout= 1/np.sqrt(-4 * w + A*A)*(1/scale)
            dCout = (1/w)*np.power(C,1/w-1) 
            norm = np.sqrt(np.real(dCout)*np.real(dCout) +np.imag(dCout)*np.imag(dCout))
            zeros = np.zeros(X.shape)
            ones = np.ones(X.shape)
            v1 = [np.imag(dCout) / norm, np.real(dCout) / norm, zeros]
            v2 = [v1[1], -v1[0], zeros]
            v3 = [zeros, zeros, ones]
            return v1, v2, v3

        
        @L1L2L3_drt_w_scale(drt=drt,w=w,scale=scale,beta=None)
        @np.vectorize
        def L1L2L3(X,Y,Z,w=None,scale=None,drt=None,beta=None):
            if scale is None: scale = 5
            #if drt is None: drt=1.5 #drt is the radial coordinate where we transition from tang to rad
            if w is None: w=1
            if beta is None: beta=0

            l1 = 0.01
            l2 = l1/100
            l3 = 0.00
            rot = cmath.rect(1, beta)
            C = X + Y * 1j
            C = C * rot
            A = C / scale + w+1
            #Cout = np.log(0.5*(A+np.sqrt(A*A-4*w)))
            Cout = np.log(0.5*(A+np.sqrt(A*A-4*w)))
            U= np.real(Cout)

            #if U<drt:
            L1=l1
            L2=l2
            L3=l3
            #else:
            #    L1 = l1
            #    L2 = l2
            #    L3 = l3
            return L1,L2,L3

        @L1L2L3_drt_w_scale(drt=drt,w=w,scale=scale,beta=None)
        @np.vectorize
        def windowFunction(X,Y,Z,w=None,scale=None,drt=None,beta=None):
            if scale is None: scale = 5
            #if drt is None: drt=1.5 #drt is the radial coordinate where we transition from tang to rad
            if w is None: w=1
            if beta is None: beta=0

            uu = 1/1.75
            u = 0.02
            #x_h= 0.5*
            x_h= 0.5*(phiInv(uu,0,0,w=w)[0]+phiInv(u,0,0,w=w)[0])
    
            drt= phi(x_h,0,0,beta=beta)[0]




            rot = cmath.rect(1, beta)
            C = X + Y * 1j
            C = C * rot
            A = C / scale + w+1
            #Cout = np.log(0.5*(A+np.sqrt(A*A-4*w)))
            Cout=np.power(C,1/w)
            U= np.real(Cout)

            # if U<drt:
            #     out=1
            # else:
            #     out=0

            lk=60
            out=1/(1+np.exp(-lk*(U-drt)))

            return out


        uu = 1/1.75
        u = 0.02
        vv = np.pi / 4
        v = -np.pi / 4
        N = 50
        delta = (uu - u) / (N - 1)

        x_h= 0.5*(phiInv(uu,0,0,w=w)[0]+phiInv(u,0,0,w=w)[0])
    
        drt= phi(x_h,0,0,beta=beta)[0]


        Uparams = domainParams(u, uu, v, vv, 0, 1, deltas=[delta, delta, 1])

        # evaluate deltas for native space
        X, Y, Z = phiInv(Uparams.A, Uparams.B, Uparams.C)
        Nx = (np.nanmax(X) - np.nanmin(X)) / res + 1
        Ny = (np.nanmax(Y) - np.nanmin(Y)) / res + 1
        Uparams = domainParams(u, uu, v, vv, 0, 1 * res, deltas=[delta, delta, res])
        if not os.path.exists(self.npath):
            os.mkdir(self.npath)

        bvals = "/home/u2hussai/scratch/unfoldingSimulations/data/sampleBvalBvecs/bvals"
        bvecs = "/home/u2hussai/scratch/unfoldingSimulations/data/sampleBvalBvecs/bvecs"

        self.Uparams=Uparams
        sim = simulateDiffusion.simulateDiffusion(phi, dphi, phiInv, Uparams, L1L2L3, windowFunction, bvals=bvals, bvecs=bvecs, Nx=Nx,Ny=Ny)
        sim.simulate(self.npath)

        #make some masks
        #radtang
        radtang_nii=copy.deepcopy(sim.U_nii.get_fdata())
        radtang_nii[sim.U_nii.get_fdata() >= drt] = 1
        radtang_nii[sim.U_nii.get_fdata() < drt] = 2
        radtang_nii=nib.Nifti1Image(radtang_nii,sim.U_nii.affine)
        nib.save(radtang_nii, self.npath+'radtang.nii.gz')
        #updown
        updown_nii = copy.deepcopy(sim.U_nii.get_fdata())
        updown_nii[sim.V_nii.get_fdata() >= 0] = 1
        updown_nii[sim.V_nii.get_fdata() < 0] = 2
        updown_nii = nib.Nifti1Image(updown_nii, sim.U_nii.affine)
        nib.save(updown_nii, self.npath + 'updown.nii.gz')
        #halfdrt
        halfdrt_nii = copy.deepcopy(sim.U_nii.get_fdata())
        halfdrt_nii[sim.U_nii.get_fdata() < drt] = 1
        halfdrt_nii[sim.U_nii.get_fdata() < drt/2] = 2
        halfdrt_nii[sim.U_nii.get_fdata() >= drt] = np.NaN
        halfdrt_nii = nib.Nifti1Image(halfdrt_nii, sim.U_nii.affine)
        nib.save(halfdrt_nii, self.npath + 'halfdrt.nii.gz')
        #angles
        angles_nii = copy.deepcopy(sim.U_nii.get_fdata())
        max_V = np.nanmax(sim.V_nii.get_fdata())
        min_V = np.nanmin(sim.V_nii.get_fdata())
        N_angles=19
        delta_angles=(max_V-min_V)/(N_angles-1)
        for a in range(0,N_angles):
            angle=max_V-delta_angles*a
            condition=sim.V_nii.get_fdata()<=(angle+delta_angles)
            angles_nii[condition]=N_angles-a
        angles_nii = nib.Nifti1Image(angles_nii, sim.U_nii.affine)
        nib.save(angles_nii, self.npath + 'angles.nii.gz')
        #seeds
        radtang=radtang_nii.get_fdata()
        angles=angles_nii.get_fdata()
        seeds_nii=copy.deepcopy(sim.U_nii.get_fdata())
        seeds_nii[:]=np.NaN
        seeds_nii[(radtang==2) & (angles==1)]=1
        seeds_nii=nib.Nifti1Image(seeds_nii,sim.U_nii.affine)
        nib.save(seeds_nii, self.npath + 'seeds.nii.gz')
        #zero graddev
        #shp=radtang_nii.get_fdata().shape
        #graddev_nii=np.zeros(shp[0],shp[1],shp[2],9)
        #graddev_nii=nib.Nifti1Image(graddev_nii,sim.U_nii.affine)
        #nib.save(graddev_nii,self.npath+'grad_dev.nii.gz')



    def unfold(self):

        sub = unfoldSubject.unfoldSubject()
        sub.loadCoordinates(path=self.npath, prefix="")
        sub.coords.computeGradDev()
        sub.loadDiffusion(self.npath)
        sub.pushToUnfold(type='diffusion')
        upath = self.npath + "Unfolded/"
        if not os.path.exists(upath):
            os.mkdir(upath)
        shutil.copyfile(self.npath + "bvals", upath + "bvals")
        shutil.copyfile(self.npath + "bvecs", upath + "bvecs")
        nib.save(sub.diffUnfold.vol, upath + 'data.nii.gz')
        nib.save(sub.coords.gradDevUVW_nii, upath + 'grad_dev.nii.gz')
        nib.save(sub.coords.gradDevXYZ_nii, upath + 'grad_devXYZ.nii.gz')
        temp_mask = sub.diffUnfold.mask.get_fdata()
        #temp_mask[:, :, -1] = np.NaN
        temp_mask = nib.Nifti1Image(temp_mask, sub.diffUnfold.mask.affine)
        nib.save(sub.diffUnfold.mask, upath + 'nodif_brain_mask.nii.gz')


    def track(self,ang_thr=None,default_sphere=default_sphere):
        default_sphere = default_sphere.subdivide()
        default_sphere = default_sphere.subdivide()
        default_sphere = default_sphere.subdivide()
        default_sphere.vertices = np.append(default_sphere.vertices, [[1, 0, 0]])
        default_sphere.vertices = np.append(default_sphere.vertices, [[-1, 0, 0]])
        default_sphere.vertices = np.append(default_sphere.vertices, [[0, 1, 0]])
        default_sphere.vertices = np.append(default_sphere.vertices, [[0, -1, 0]])
        default_sphere.vertices = default_sphere.vertices.reshape([-1, 3])

        # this needs to be moved into tracking class (in unfoldTracking)
        def loc_track(path, default_sphere, coords=None, npath=None, UParams=None,ang_thr=None):
            data, affine = load_nifti(path + 'data.nii.gz')
            data[np.isnan(data) == 1] = 0
            mask, affine = load_nifti(path + 'nodif_brain_mask.nii.gz')
            mask[np.isnan(mask) == 1] = 0
            mask[:, :, 1:] = 0
            stopper = copy.deepcopy(mask)
            #stopper[:, :, :] = 1
            gtab = gradient_table(path + 'bvals', path + 'bvecs')


            csa_model = CsaOdfModel(gtab, smooth=1, sh_order=12)
            peaks = peaks_from_model(csa_model, data, default_sphere,
                                     relative_peak_threshold=0.8,
                                     min_separation_angle=45,
                                     mask=mask)
            if ang_thr is not None:
                peaks.ang_thr = ang_thr
            if os.path.exists(path + 'grad_dev.nii.gz'):
                gd, affine_g = load_nifti(path + 'grad_dev.nii.gz')
                nmask, naffine = load_nifti(npath + 'nodif_brain_mask.nii.gz')
                seedss, saffine = load_nifti(npath + 'seeds.nii.gz')
                nmask[np.isnan(nmask) == 1] = 0
                nmask[:, :, 1:] = 0
                seedss[np.isnan(seedss) == 1]=0
                seedss[:, :, 1:] = 0
                #seedss = copy.deepcopy(nmask)
                
                
                seedss = utils.seeds_from_mask(seedss, naffine, [2, 2, 2])
                useed = []
                UParams = coords.Uparams
                for seed in seedss:
                    us = coords.rFUa_xyz(seed[0], seed[1], seed[2])
                    vs = coords.rFVa_xyz(seed[0], seed[1], seed[2])
                    ws = coords.rFWa_xyz(seed[0], seed[1], seed[2])
                    condition = us >= UParams.min_a and us <= UParams.max_a and vs >= UParams.min_b and vs <= UParams.max_b \
                                and ws >= UParams.min_c and ws <= UParams.max_c
                    if condition == True:
                        useed.append([float(us), float(vs), float(ws)])
                seeds = np.asarray(useed)

            else:
                gd = None
                seedss = copy.deepcopy(mask)
                seedss, saffine = load_nifti(npath + 'seeds.nii.gz')
                seedss[np.isnan(seedss) == 1]=0
                seedss[:, :, 1:] = 0
                seeds = utils.seeds_from_mask(seedss, affine, [2, 2, 2])

            stopping_criterion = BinaryStoppingCriterion(stopper)
            tracked = tracking(peaks, stopping_criterion, seeds, affine, graddev=gd, sphere=default_sphere)
            tracked.localTracking()
            return tracked

        # this needs to be moved to unfoldStreamlines (in unfoldTracking)
        def unfold2nativeStreamlines(tracking, coords):
            points, X = getPointsData(coords.X_uvwa_nii)
            points, Y = getPointsData(coords.Y_uvwa_nii)
            points, Z = getPointsData(coords.Z_uvwa_nii)
            allLines = tracking.streamlines.get_data()
            x = coords.rFX_uvwa(allLines[:, 0], allLines[:, 1], allLines[:, 2])
            y = coords.rFY_uvwa(allLines[:, 0], allLines[:, 1], allLines[:, 2])
            z = coords.rFZ_uvwa(allLines[:, 0], allLines[:, 1], allLines[:, 2])
            #x = coords.FX_uvwa(allLines[:, 0], allLines[:, 1], allLines[:, 2])
            #y = coords.FY_uvwa(allLines[:, 0], allLines[:, 1], allLines[:, 2])
            #z = coords.FZ_uvwa(allLines[:, 0], allLines[:, 1], allLines[:, 2])
            allLines = np.asarray([x, y, z]).T
            pointsPerLine = tracking.NpointsPerLine
            streamlines = []
            first = 0
            for i in range(0, len(pointsPerLine) - 1):
                templine = []
                points = allLines[first:first + pointsPerLine[i]]
                for p in range(0, pointsPerLine[i]):
                    if( np.isnan(np.sum(points[p]))==0):
                        templine.append(points[p])
                if (len(templine) > 1):  # and len(templine) == pointsPerLine[i]):
                    streamlines.append(templine)
                first = first + pointsPerLine[i]
            return streamlines

            # path="/home/uzair/PycharmProjects/Unfolding/data/diffusionSimulations_nonConformal_scale/"

        ntracking = loc_track(self.npath, default_sphere,ang_thr=ang_thr,npath=self.npath)
        coords = coordinates.coordinates(self.npath, '')
        utracking = loc_track(self.upath, default_sphere, coords=coords, npath=self.npath,
                              UParams=self.Uparams, ang_thr=ang_thr)
        u2n_streamlines = unfold2nativeStreamlines(utracking, coords)
        streamline.save_vtk_streamlines(ntracking.streamlines,
                                        self.npath + "native_streamlines.vtk")
        streamline.save_vtk_streamlines(utracking.streamlines,
                                        self.upath + "unfold_streamlines.vtk")
        streamline.save_vtk_streamlines(u2n_streamlines,
                                        self.npath + "from_unfold_streamlines.vtk")





#Parameters to vary are drt (rad tang transistion) and resolution

scale=1
res=np.linspace(0.03,0.13,16)
drt=[0.5*(1/1.75 - 0.07)]#np.linspace(0.1,0.3,16)
ang_thr=np.linspace(20,90,16)
w=np.linspace(1.0,1.99,16)
beta=np.linspace(0,np.pi/4,5)



i=int(sys.argv[1]) #res
j=int(sys.argv[2]) #w

#k=int(sys.argv[3]) #ang_thr
for k in range(0,16):#len(ang_thr)):
    for l in range(0,1):#len(w)):
        for b in range(0,1):#len(beta)):

            base = "/home/u2hussai/scratch/simulations/diffusionSimulations_LambdaStep-k-60_scale-50_res-"
            npath=base + str(int(res[i] * 1000)) + "_um-drt-" +str(int(drt[l] * 100)) +"_w-"+\
                str(int(round(w[j],5)*1000))+"_angthr-" + str(int(ang_thr[k])) + "_beta-"+str(int(beta[b]*100))+"/"
            upath=npath+"Unfolded/"
            tracking_experiment=tractographyExperiments(npath,upath)
            tracking_experiment.simulate(w[j],drt[l],scale,res[i],beta[b])
            tracking_experiment.unfold()
            tracking_experiment.track(ang_thr[k])
