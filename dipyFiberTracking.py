import numpy as np
from dipy.io.image import load_nifti, save_nifti
from dipy.io.gradients import read_bvals_bvecs
from dipy.core.gradients import gradient_table
import dipy.reconst.dti as dti
from dipy.tracking import utils
from dipy.tracking.local_tracking import LocalTracking
from dipy.tracking.streamline import Streamlines
from dipy.tracking.stopping_criterion import ThresholdStoppingCriterion
from dipy.direction import peaks_from_model
from dipy.data import default_sphere
from dipy.viz import window, actor, has_fury
from dipy.viz import colormap
import matplotlib.pyplot as plt
from dipy.reconst.shm import CsaOdfModel
import copy
from dipy.reconst.dti import quantize_evecs
from scipy.spatial import KDTree
import copy
from dipy.reconst.csdeconv import (ConstrainedSphericalDeconvModel,
                                   auto_response_ssst)
from dipy.direction import DeterministicMaximumDirectionGetter
import coordinates
from unfoldTracking import tracking
from scipy.interpolate import griddata
from unfoldTracking import unfoldStreamlines
from coordinates import getPointsData
from dipy.tracking.utils import length
import os
from dipy.io import streamline
import nibabel as nib

#subdivide sphere
default_sphere=default_sphere.subdivide()
default_sphere=default_sphere.subdivide()
default_sphere=default_sphere.subdivide()
default_sphere.vertices=np.append(default_sphere.vertices,[[1,0,0]])
default_sphere.vertices=np.append(default_sphere.vertices,[[-1,0,0]])
default_sphere.vertices=np.append(default_sphere.vertices,[[0,1,0]])
default_sphere.vertices=np.append(default_sphere.vertices,[[0,-1,0]])
default_sphere.vertices=default_sphere.vertices.reshape([-1,3])

#this needs to be moved into tracking class (in unfoldTracking)
def loc_track(path,default_sphere):
    data, affine = load_nifti(path + 'data.nii.gz')
    data[np.isnan(data) == 1] = 0
    mask, affine = load_nifti(path + 'nodif_brain_mask.nii.gz')
    mask[np.isnan(mask) == 1] = 0
    #mask[:,:,2:]=0
    gtab = gradient_table(path + 'bvals', path + 'bvecs')
    if os.path.exists(path + 'grad_dev.nii.gz'):
        gd, affine_g = load_nifti(path + 'grad_dev.nii.gz')
    else:
        gd=None
    csa_model = CsaOdfModel(gtab, smooth=1, sh_order=12)
    peaks = peaks_from_model(csa_model, data, default_sphere,
                             relative_peak_threshold=0.99,
                             min_separation_angle=25,
                             mask=mask)
    seedss = copy.deepcopy(mask)
    seeds = utils.seeds_from_mask(seedss, affine, [2, 2, 2])
    stopping_criterion = ThresholdStoppingCriterion(mask, 0)
    tracked=tracking(peaks, stopping_criterion, seeds, affine, graddev=gd, sphere=default_sphere)
    tracked.localTracking()
    return tracked

#this needs to be moved to unfoldStreamlines (in unfoldTracking)
def unfold2nativeStreamlines(tracking,coords):
    points, X = getPointsData(coords.X_uvwa_nii)
    points, Y = getPointsData(coords.Y_uvwa_nii)
    points, Z = getPointsData(coords.Z_uvwa_nii)
    allLines = tracking.streamlines.get_data()
    x=coords.FX_uvwa(allLines[:,0],allLines[:,1],allLines[:,2])
    y=coords.FY_uvwa(allLines[:,0],allLines[:,1],allLines[:,2])
    z=coords.FZ_uvwa(allLines[:,0],allLines[:,1],allLines[:,2])
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


#tracking for the hippocampus
base="/home/uzair/PycharmProjects/Unfolding/data/oldUnfold/DiffusionCropped/Native_hiRes/Crop/L/cropped/"
path=base
print('doing native tracking')
#ntracking = loc_track(path ,default_sphere)
# print('loading coordinates')
# coords = coordinates.coordinates(path,'')
# print('doing unfold tracking')
# utracking = loc_track(path+'Unfolded/',default_sphere)
# u2n_streamlines=unfold2nativeStreamlines(utracking,coords)
#
# mask=nib.load(base+'nodif_brain_mask.nii.gz')
# maskk=mask.get_fdata()
# u2n_streamlines_f=[]
# for line in u2n_streamlines:
#     templine=[]
#     for p in line:
#         inds=coordinates.toInds(mask,[p])
#         inds=inds.round().astype(int)
#         if np.isnan(maskk[inds[0][0],inds[0][1],inds[0][2]])==0:
#             templine.append(p)
#     if len(templine)>2:
#         u2n_streamlines_f.append(templine)
#
# print('saving')
# streamline.save_vtk_streamlines(ntracking.streamlines,
#                                 path+"native_streamlines.vtk")
# streamline.save_vtk_streamlines(utracking.streamlines,
#                                 path + "Unfolded/unfold_streamlines.vtk")
# streamline.save_vtk_streamlines(u2n_streamlines,
#                     path + "from_unfold_streamlines.vtk")

#loop over all states and save the streamlines
# scale=50
# res=[1.75,1.5,1.25,1.00,0.75]
# res=np.asarray(res)
# res=scale*res/100
# drt=np.linspace(0.1,0.25,5)
# w=np.linspace(0.9,0.99,4)
# for i in range(2,3):#len(res)):
#     print(i)
#     for j in range(2,3):#len(drt)):
#         for k in range(3,4):#len(w)):
#             base = "/home/uzair/PycharmProjects/Unfolding/data/diffusionSimulations_res-"
#             path = base + str(int(res[i] * 10000)) + "mm_drt-" + str(int(drt[j] * 100)) + "+w-" + str(
#                 int(w[k] * 100)) + "/"
#             #path="/home/uzair/PycharmProjects/Unfolding/data/diffusionSimulations_nonConformal_scale/"
#             ntracking = loc_track(path ,default_sphere)
#             coords = coordinates.coordinates(path,'')
#             utracking = loc_track(path+'Unfolded/',default_sphere)
#             u2n_streamlines=unfold2nativeStreamlines(utracking,coords)
#             streamline.save_vtk_streamlines(ntracking.streamlines,
#                                             path+"native_streamlines.vtk")
#             streamline.save_vtk_streamlines(utracking.streamlines,
#                                             path + "Unfolded/unfold_streamlines.vtk")
#             streamline.save_vtk_streamlines(u2n_streamlines,
#                                 path + "from_unfold_streamlines.vtk")


def plot_streamlines(streamlines):
    if has_fury:
        # Prepare the display objects.
        color = colormap.line_colors(streamlines)

        streamlines_actor = actor.line(streamlines,
                                       colormap.line_colors(streamlines))

        # Create the 3D display.
        scene = window.Scene()
        scene.add(streamlines_actor)

        # Save still images for this static example. Or for interactivity use
        window.show(scene)


# #loadfiles
# #native
# npath="/home/uzair/PycharmProjects/Unfolding/data/diffusionSimulations_res-100mm_drt-160/"
# ndata,naffine=load_nifti(npath+'data.nii.gz')
# ndata[np.isnan(ndata)==1]=0
# nmask,affine1 = load_nifti(npath+'nodif_brain_mask.nii.gz')
# nmask[np.isnan(nmask)==1]=0
# #nmask[:]=1
# #nmask[:,:,1:]=0
# #coords = coordinates.coordinates(npath,'')
# nbvals, nbvecs = read_bvals_bvecs(npath+'bvals', npath+'bvecs')
# ngtab = gradient_table(npath+'bvals', npath+'bvecs')
# csa_model = CsaOdfModel(ngtab, sh_order=10)
# npeaks = peaks_from_model(csa_model, ndata, default_sphere,
#                               relative_peak_threshold=0.99,
#                               min_separation_angle=25,
#                               mask=nmask)
# seedss=copy.deepcopy(nmask)
# seeds=utils.seeds_from_mask(seedss,naffine,[2,2,2])
# stopping_criterion=ThresholdStoppingCriterion(nmask,0)
# ntracking=tracking(npeaks,stopping_criterion,seeds,naffine,sphere=default_sphere)
# ntracking.localTracking()
# #ntracking.plot()
#
# #unfold
# upath="/home/uzair/PycharmProjects/Unfolding/data/diffusionSimulations"+trail +"Unfolded/"
# udata,uaffine=load_nifti(upath+'diffunfold.nii.gz')
# udata[np.isnan(udata)==1]=0
# umask,affine1 = load_nifti(upath+'nodif_brain_mask.nii.gz')
# umask[np.isnan(umask)==1]=0
# umask[:]=1
# #nmask[:,:,1:]=0
#
#
# ubvals, ubvecs = read_bvals_bvecs(upath+'bvals', upath+'bvecs')
# ugtab = gradient_table(upath+'bvals', upath+'bvecs')
# gd,affine_g=load_nifti(upath+'grad_devUVW.nii.gz')
# csa_model = CsaOdfModel(ugtab, sh_order=10)
# upeaks = peaks_from_model(csa_model, udata, default_sphere,
#                               relative_peak_threshold=0.99,
#                               min_separation_angle=25,
#                               mask=umask)
# useedss=copy.deepcopy(umask)
# useeds=utils.seeds_from_mask(useedss,uaffine,[2,2,2])
# ustopping_criterion=ThresholdStoppingCriterion(umask,0)
# utracking=tracking(upeaks,ustopping_criterion,useeds,uaffine,graddev=gd, sphere=default_sphere)
# utracking.localTracking()
# #utracking.plot()
#
#
# a_streams=unfoldStreamlines(ntracking.streamlines,
#                             utracking.streamlines,
#                             ntracking.NpointsPerLine,
#                             utracking.NpointsPerLine,
#                             coords)
#
#
#
# points,X=getPointsData(coords.X_uvwa_nii)
# points,Y=getPointsData(coords.Y_uvwa_nii)
# points,Z=getPointsData(coords.Z_uvwa_nii)
#
#
# allLines= utracking.streamlines.get_data()
# coords.meanArcLength()
#
# x=coords.rFX_uvwa(allLines[:,0],allLines[:,1],allLines[:,2])
# y=coords.rFY_uvwa(allLines[:,0],allLines[:,1],allLines[:,2])
# z=coords.rFZ_uvwa(allLines[:,0],allLines[:,1],allLines[:,2])
#
# allLines=np.asarray([x,y,z]).T
#
#
# pointsPerLine=utracking.NpointsPerLine
# streamlines=[]
# first=0
# for i in range(0,len(pointsPerLine)-1):
#     templine=[]
#     points = allLines[first:first +pointsPerLine[i]]
#     for p in range(0,pointsPerLine[i]):
#          #if( np.isnan(np.sum(points[p]))==0):
#         templine.append(points[p])
#     if(len(templine)>1):# and len(templine) == pointsPerLine[i]):
#         streamlines.append(templine)
#     first=first+pointsPerLine[i]
#
#
# if has_fury:
#     # Prepare the display objects.
#     color = colormap.line_colors(streamlines)
#
#     streamlines_actor = actor.line(streamlines,
#                                    colormap.line_colors(streamlines))
#
#     # Create the 3D display.
#     scene = window.Scene()
#     scene.add(streamlines_actor)
#
#     # Save still images for this static example. Or for interactivity use
#     window.show(scene)
#
#
# from dipy.tracking.utils import target
#
# U,affine_g=load_nifti(npath+'U.nii.gz')
# V,affine_g=load_nifti(npath+'V.nii.gz')
# source=copy.deepcopy(U)
# source[:]=False
# condition= ((U <= 1.2) & (V <= -0.5))
# source[condition]=True
#
# #
# radtang,affine_g=load_nifti(npath+'radtang.nii.gz')
# rad=copy.deepcopy(radtang)
# tang=copy.deepcopy(radtang)
#
# condition= ((U <= 1.26))
#
# rad[:]=False
# tang[:]=False
# rad[radtang==1]=True
# tang[radtang==2]=True
# tang[condition]=True
#
# n_filter_generator=target(ntracking.streamlines,affine_g,source)
# u_filter_generator=target(streamlines,affine_g,source)
#
# n_filter_streamlines=Streamlines(n_filter_generator)
# u_filter_streamlines=Streamlines(u_filter_generator)
#
# n_radtang_streamlines=Streamlines(target(ntracking.streamlines,affine_g,tang))
# n_radtang_streamlines=Streamlines(target(n_radtang_streamlines,affine_g,rad))
#
# def plot(streamlines):
#     if has_fury:
#         # Prepare the display objects.
#         color = colormap.line_colors(streamlines)
#
#         streamlines_actor = actor.line(streamlines,
#                                        colormap.line_colors(streamlines))
#
#         # Create the 3D display.
#         scene = window.Scene()
#         scene.add(streamlines_actor)
#
#         # Save still images for this static example. Or for interactivity use
#         window.show(scene)
#
# plot(n_filter_streamlines)
# plot(u_filter_streamlines)