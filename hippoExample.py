import numpy as np
import simulateDiffusion
from coordinates import domainParams
import os
import matplotlib.pyplot as plt
import matplotlib
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
from dipy.tracking.utils import target
import sys

class tractographyExperiments:
    def __init__(self,npath,upath):
        self.npath=npath
        self.upath=upath
        self.Uparams=[]

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
        def loc_track(path, default_sphere,seedss,stopper=None,mask=None, coords=None, npath=None, UParams=None,
                      ang_thr=None):
            data, affine = load_nifti(path + 'data.nii.gz')
            data[np.isnan(data) == 1] = 0
            if mask is None:
                mask, affine = load_nifti(path + 'nodif_brain_mask.nii.gz')
                mask[np.isnan(mask) == 1] = 0
            #mask[:, :, 1:] = 0

            #stopper[:, :, :] = 1
            gtab = gradient_table(path + 'bvals', path + 'bvecs')

            csa_model = CsaOdfModel(gtab, smooth=1, sh_order=12)
            peaks = peaks_from_model(csa_model, data, default_sphere,
                                     relative_peak_threshold=0.99,
                                     min_separation_angle=45,
                                     mask=mask)
            if stopper is None:
                stopping_criterion = ThresholdStoppingCriterion(peaks.gfa, .015)

            if ang_thr is not None:
                peaks.ang_thr = ang_thr
            if os.path.exists(path + 'grad_dev.nii.gz'):
                gd, affine_g = load_nifti(path + 'grad_dev.nii.gz')
                nmask, naffine = load_nifti(npath + 'nodif_brain_mask.nii.gz')
                nmask[np.isnan(nmask) == 1] = 0
                #nmask[:, :, 1:] = 0
                sseeds = utils.seeds_from_mask(seedss, naffine, [2, 2, 2])
                print(len(sseeds))
                useed = []
                UParams = coords.Uparams
                for seed in sseeds:
                    us = coords.rFUa_xyz(seed[0], seed[1], seed[2])
                    vs = coords.rFVa_xyz(seed[0], seed[1], seed[2])
                    ws = coords.rFWa_xyz(seed[0], seed[1], seed[2])
                    condition = us >= UParams.min_a and us <= UParams.max_a and vs >= UParams.min_b and vs <= UParams.max_b \
                                and ws >= UParams.min_c and ws <= UParams.max_c
                    if condition == True:
                        useed.append([float(us), float(vs), float(ws)])
                seeds = np.asarray(useed)
                plt.interactive('off')
                #matplotlib.use('Agg')
                #fig, ax =plt.subplots()
                #plt.scatter(seeds[:,0],seeds[:,1])
                #plt.savefig('seeds.png')

            else:
                gd = None
                #seedss = copy.deepcopy(mask)
                seeds = utils.seeds_from_mask(seedss, affine, [2, 2, 2])

            #stopping_criterion = BinaryStoppingCriterion(stopper)
            tracked = tracking(peaks, stopping_criterion, seeds, affine, graddev=gd, sphere=default_sphere)
            tracked.localTracking()
            return tracked

        # this needs to be moved to unfoldStreamlines (in unfoldTracking)
        #def unfold2nativeStreamlines(tracking, coords):
        def unfold2nativeStreamlines(tracking, coords):
            allLines_p = tracking.streamlines.get_data()
            batchsize=5000
            allLines=copy.deepcopy(allLines_p)
            #for ll in range(0,allLines_p.shape[0],batchsize):
            someint=int(allLines_p.shape[0]/1)
            #someint = int(len(allLines_p))
            for ll in range(0,someint,batchsize):
                print(ll)
                allLines[ll:ll+batchsize, 0] = coords.rFX_uvwa(allLines_p[ll:ll+batchsize, 0], allLines_p[ll:ll+batchsize, 1], allLines_p[ll:ll+batchsize, 2])
                allLines[ll:ll+batchsize, 1] = coords.rFY_uvwa(allLines_p[ll:ll+batchsize, 0], allLines_p[ll:ll+batchsize, 1], allLines_p[ll:ll+batchsize, 2])
                allLines[ll:ll+batchsize, 2] = coords.rFZ_uvwa(allLines_p[ll:ll+batchsize, 0], allLines_p[ll:ll+batchsize, 1], allLines_p[ll:ll+batchsize, 2])
                #allLines[ll:ll+batchsize, 0] = coords.FX_uvwa(allLines_p[ll:ll+batchsize, 0], allLines_p[
                # ll:ll+batchsize, 1], allLines_p[ll:ll+batchsize, 2])
                #allLines[ll:ll+batchsize, 1] = coords.FY_uvwa(allLines_p[ll:ll+batchsize, 0], allLines_p[
                # ll:ll+batchsize, 1], allLines_p[ll:ll+batchsize, 2])
                #allLines[ll:ll+batchsize, 2] = coords.FZ_uvwa(allLines_p[ll:ll+batchsize, 0], allLines_p[
                # ll:ll+batchsize, 1], allLines_p[ll:ll+batchsize, 2])

            pointsPerLine = tracking.NpointsPerLine
            streamlines = []
            first = 0
            for i in range(0, len(pointsPerLine) - 1):
                templine = []
                points = allLines[first:first + pointsPerLine[i]]
                for p in range(0, pointsPerLine[i]):
                    if( np.isnan(np.sum(points[p]))==0):
                        templine.append(points[p])
                if (len(templine) > 1 and len(templine) == pointsPerLine[i] ):
                    streamlines.append(templine)
                first = first + pointsPerLine[i]
            return streamlines



            # path="/home/uzair/PycharmProjects/Unfolding/data/diffusionSimulations_nonConformal_scale/"

        # load subfields
        subfieldpath = 'data/oldUnfold/subfields/'
        sub, aaffine = load_nifti(subfieldpath + 'sub5.nii.gz')
        CA1, dump = load_nifti(subfieldpath + 'sub4.nii.gz')
        CA2, dump = load_nifti(subfieldpath + 'sub3.nii.gz')
        CA3, dump = load_nifti(subfieldpath + 'sub2.nii.gz')
        CA4, dump = load_nifti(subfieldpath + 'sub1.nii.gz')

        AP, dump = load_nifti('data/oldUnfold/U.nii.gz')
        PD, dump = load_nifti('data/oldUnfold/V.nii.gz')
        IO, dump = load_nifti('data/oldUnfold/W.nii.gz')


        AP[np.isnan(AP) == 1] = 0
        PD[np.isnan(PD) == 1] = 0
        IO[np.isnan(IO) == 1] = 0

        subs_array=[sub,CA1,CA2,CA3,CA4]
        #subs_strings=['sub','CA1','CA2','CA3','CA4']
        subs_strings = ['CA1','CA3']

        bins = np.linspace(0, 1, 4)


        # def get_hippo_tracts(i,CA1,CA3,affine,npath,upath,unfld):
        #     CA1[(AP >= bins[i + 1]) | (AP <= bins[i])] = 0
        #     CA3[(AP >= bins[i + 1]) | (AP <= bins[i])] = 0
        #     both_mask=CA1+CA3
        #     # seed from CA1 to CA3
        #
        #     if unfld == 0:
        #         CA1CA3 = loc_track(npath, default_sphere, CA1, ang_thr=ang_thr)
        #         CA3CA1 = loc_track(npath, default_sphere, CA3, ang_thr=ang_thr)
        #         return Streamlines(target(np.concatenate((CA1CA3, CA3CA1)), affine, both_mask))
        #     else:
        #         CA1CA3 = loc_track(upath, default_sphere, CA1, npath=npath, ang_thr=ang_thr)
        #         CA3CA1 = loc_track(upath, default_sphere, CA3, npath=npath, ang_thr=ang_thr)
        #         CA1CA3 = unfold2nativeStreamlines(CA1CA3 , coords)
        #         CA3CA1 = unfold2nativeStreamlines(CA3CA1 , coords)
        #         return Streamlines(target( np.concatenate((CA1CA3.streamlines,CA3CA1.streamlines)),affine,both_mask))

        #for s,subfield in enumerate(subs_array):
        for i in range(0,3):

            #seedss[np.isnan(seedss) == 1] = 0
            # seedss[:]=0

            #ntracking = loc_track(self.npath, default_sphere,seedss,self.npath,ang_thr=ang_thr)

            seedCA1 = copy.deepcopy(CA1)
            seedCA1[(AP >= bins[i + 1]) | (AP <= bins[i])] = 0
            seedCA3 = copy.deepcopy(CA3)
            seedCA3[(AP >= bins[i + 1]) | (AP <= bins[i])] = 0

            binmask=copy.deepcopy(AP)
            binmask[binmask>0]=1
            binmask[(AP >= bins[i + 1]) | (AP <= bins[i])] = 0

            both_mask = seedCA1 + seedCA3

            CA1CA3 = loc_track(npath, default_sphere, seedCA1,mask=binmask, ang_thr=ang_thr)
            CA3CA1 = loc_track(npath, default_sphere, seedCA3,mask=binmask, ang_thr=ang_thr)
            ntracking= Streamlines(target(np.concatenate((CA1CA3.streamlines, CA3CA1.streamlines)), aaffine, both_mask))

            #ntracking= get_hippo_tracts(i,CA1,CA3,aaffine,self.npath,self.upath,0)

            coords = coordinates.coordinates(self.npath, '')
            print(coords.mean_u)
            print(coords.mean_v)
            print(coords.mean_w)
            binmask=copy.deepcopy(coords.X_uvwa_nii.get_fdata())
            deltabin=int(binmask.shape[0]/3)
            binmask[np.isnan(binmask)==0]=2
            binmask[i*deltabin:(i+1)*deltabin,:,:]=1
            binmask[binmask==2]=0

            CA1CA3 = loc_track(upath, default_sphere, seedCA1,mask=binmask, coords=coords, npath=npath, ang_thr=ang_thr)
            CA3CA1 = loc_track(upath, default_sphere, seedCA3,mask=binmask, coords=coords, npath=npath, ang_thr=ang_thr)
            CA1CA3 = unfold2nativeStreamlines(CA1CA3, coords)
            CA3CA1 = unfold2nativeStreamlines(CA3CA1, coords)

            u2n_streamlines = Streamlines(target(np.concatenate((CA1CA3, CA3CA1)), aaffine,
                                           both_mask))
            #utracking = get_hippo_tracts(i,CA1,CA3,aaffine,self.npath,self.upath,1)
            #utracking = loc_track(self.upath, default_sphere,seedss, coords=coords, npath=self.npath,
            #                      UParams=self.Uparams, ang_thr=ang_thr)

            #u2n_streamlines = unfold2nativeStreamlines(utracking, coords)

            native_streamlines="native_streamline_%s_bin-%d.vtk" % ('CA1CA3e',i)
            unfold_streamlines= "unfold_streamlines_%s_bin-%d.vtk" % ('CA1CA3e', i)
            from_unfold_streamlines = "from_unfold_streamlines_%s_bin-%d.vtk" % ('CA1CA3e', i)

            streamline.save_vtk_streamlines(ntracking,
                                            self.npath + native_streamlines)
            #streamline.save_vtk_streamlines(utracking,
            #                                self.upath + unfold_streamlines)
            streamline.save_vtk_streamlines(u2n_streamlines,
                                            self.npath + from_unfold_streamlines)

            # streamline.save_vtk_streamlines(ntracking.streamlines,
            #                                 self.npath + native_streamlines)
            # streamline.save_vtk_streamlines(utracking.streamlines,
            #                                 self.upath + unfold_streamlines)
            # streamline.save_vtk_streamlines(u2n_streamlines,
            #                                 self.npath + from_unfold_streamlines)


base = "data/oldUnfold/"
npath=base
upath=npath+"Unfolded/"
tracking_experiment=tractographyExperiments(npath,upath)
#tracking_experiment.unfold()
tracking_experiment.track(60)
