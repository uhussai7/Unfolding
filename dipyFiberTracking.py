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
import matplotlib.pyplot as plt
from dipy.reconst.shm import CsaOdfModel
import copy
from dipy.reconst.csdeconv import ConstrainedSphericalDeconvModel
from dipy.direction import DeterministicMaximumDirectionGetter
import coordinates
from unfoldTracking import tracking
from scipy.interpolate import griddata
from unfoldTracking import unfoldStreamlines
from coordinates import getPointsData
from dipy.tracking.utils import length
import os
from dipy.io import streamline
import sys

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
    mask[:,:,2:]=0
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
    x=coords.rFX_uvwa(allLines[:,0],allLines[:,1],allLines[:,2])
    y=coords.rFY_uvwa(allLines[:,0],allLines[:,1],allLines[:,2])
    z=coords.rFZ_uvwa(allLines[:,0],allLines[:,1],allLines[:,2])
    allLines = np.asarray([x, y, z]).T
    pointsPerLine = tracking.NpointsPerLine
    streamlines = []
    first = 0
    for i in range(0, len(pointsPerLine) - 1):
        templine = []
        points = allLines[first:first + pointsPerLine[i]]
        for p in range(0, pointsPerLine[i]):
            #if( np.isnan(np.sum(points[p]))==0):
            templine.append(points[p])
        if (len(templine) > 1):  # and len(templine) == pointsPerLine[i]):
            streamlines.append(templine)
        first = first + pointsPerLine[i]
    return streamlines


#loop over all states and save the streamlines
i=int(sys.argv[1])
j=int(sys.argv[2])
k=int(sys.argv[3])
scale=100
res=[1.75,1.5,1.25,1.00,0.75]
res=np.asarray(res)
res=scale*res/100
drt=np.linspace(0.1,0.25,5)
w=np.linspace(0.9,0.99,4)

base = "/home/u2hussai/Unfolding/data/diffusionSimulations_res-"
path = base + str(int(res[i] * 10000)) + "mm_drt-" + str(int(drt[j] * 100)) + "+w-" + str(int(w[k] * 100)) + "/"
ntracking = loc_track(path ,default_sphere)
coords = coordinates.coordinates(path,'')
utracking = loc_track(path+'Unfolded/',default_sphere)
u2n_streamlines=unfold2nativeStreamlines(utracking,coords)
streamline.save_vtk_streamlines(ntracking.streamlines,path+"native_streamlines.vtk")
streamline.save_vtk_streamlines(utracking.streamlines,path + "Unfolded/unfold_streamlines.vtk")
streamline.save_vtk_streamlines(u2n_streamlines,path + "from_unfold_streamlines.vtk")


