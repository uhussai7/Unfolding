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
from dipy.tracking.utils import target
import os
from dipy.io import streamline
from dipy.tracking.streamline import Streamlines
from scipy.interpolate import interp1d


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



def rad_tang_native(streamlines,drt,U,affine):
    eps=drt*0.3
    tang = Streamlines(target(streamlines,affine,U>(drt+eps),include=False))
    tang = Streamlines(target(tang,affine,U<(drt+eps)))
    rad = Streamlines(target(streamlines, affine, U > (drt + eps)))
    return rad, tang

def inflection(streamlines,drt,U,affine):
    eps=drt*0.3
    #tang = Streamlines(target(streamlines,affine,U>(drt+eps),include=False))
    tang = Streamlines(target(streamlines,affine,U<(drt+eps)))
    tang = Streamlines(target(tang, affine, U > (drt + eps)))
    return tang

def tangent(streamlines):
    vx_bundle=[]
    vy_bundle=[]
    for line in streamlines:
        vx=[]
        vy=[]
        lines=np.asarray(line)
        for i in line.shape[0]:
            if i<line.shape[0]-1:
                vx.append(line[i+1,0] -line[i,0])
                vy.append(line[i+1, 1] -line[i, 1])
            else:
                vx.append(line[i, 0] - line[i-1, 0])
                vy.append(line[i, 1] - line[i-1, 1])
        vx_bundle.append(vx)
        vy_bundle.append(vy)
    return vx_bundle, vy_bundle

def max_x_y(streamlines,min_length,x_min,Npoints):
    max_xy=[]
    lengths=list(length(streamlines))
    streamlines_out=[]
    for l in range(0,len(streamlines)):
        line=np.asarray(streamlines[l])
        max_arg=np.argmax(line[:, 0])
        if max_arg<(line.shape[0]-5) and lengths[l]>min_length and line[max_arg, 0]>x_min and line.shape[0]>Npoints:
            max_xy.append([line[max_arg, 0], line[max_arg, 1]])
            streamlines_out.append(streamlines[l])
    return np.asarray(max_xy), streamlines_out

#loop over all states and save the streamlines
scale=100
res=[1.75,1.5,1.25,1.00]#,0.75]
res=np.asarray(res)
res=scale*res/100
drt=np.linspace(0.1,0.25,5)
w=np.linspace(0.9,0.99,4)


for i in range(0,len(res)):
    print(i)
    for j in range(0,len(drt)):
        for k in range(3,len(w)):
            base = "/home/uzair/Datasets/diffusionSimulations/data/diffusionSimulations_res-"
            path = base + str(int(res[i] * 10000)) + "mm_drt-" + str(int(drt[j] * 100)) + "+w-" + str(
                int(w[k] * 100)) + "/"


            nstreamlines=streamline.load_vtk_streamlines(path+'native_streamlines.vtk')
            fustreamlines = streamline.load_vtk_streamlines(path + 'from_unfold_streamlines.vtk')
            ustreamlines=streamline.load_vtk_streamlines(path+'Unfolded/unfold_streamlines.vtk')
            U, affine = load_nifti(path + 'U.nii.gz')
            V, affine = load_nifti(path + 'V.nii.gz')
            mask,affine=load_nifti(path+'nodif_brain_mask.nii.gz')
            one_slice=copy.deepcopy(mask)
            one_slice[:,:,1:]=0
            one_slice[np.isnan(one_slice)==1]=0
            nstreamlines=Streamlines(target(nstreamlines,affine,one_slice))



nrad,ntang=rad_tang_native(nstreamlines,drt[j],U,affine)
furad,futang=rad_tang_native(fustreamlines,drt[j],U,affine)


#Asymmetry plot
import matplotlib
nmax_x_y, nmax_x_y_streamlines = max_x_y(ntang,1.1,-0.5,5)
fumax_x_y,fumax_x_y_streamlines = max_x_y(futang,1.1,-0.5,5)

font = {'family' : 'sans-serif',
        'weight':1,
        'size'   : 22}

matplotlib.rc('font', **font)
fig, axs = plt.subplots(2)
axs[0].set_xlim(-0.5,1.25)
axs[0].set_ylim(-1.2,1.2)
axs[1].set_xlim(-0.5,1.25)
axs[1].set_ylim(-1.2,1.2)
axs[0].set_aspect('equal')
axs[1].set_aspect('equal')
#plt.setp(axs,fontsize=10)
for line in nmax_x_y_streamlines:
    axs[0].plot(line[:,0],line[:,1],color='red',alpha=0.05)
for line in fumax_x_y_streamlines:
    axs[1].plot(line[:,0],line[:,1],color='blue',alpha=0.05)

axs[0].scatter(nmax_x_y[:,0],nmax_x_y[:,1],color='red',s=100,alpha=1)
axs[1].scatter(fumax_x_y[:,0],fumax_x_y[:,1],color='blue',s=100,alpha=1)
