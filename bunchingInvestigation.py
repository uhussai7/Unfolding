import numpy as np
import simulateDiffusion
from coordinates import domainParams
import os
import matplotlib.pyplot as plt
import nibabel as nib
import unfoldSubject
import shutil

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
scale=0.05
res=[1.75,1.5,1.25,1.00,0.75]
res=np.asarray(res)
res=scale*res/100
drt=np.linspace(0.1,0.25,5)
w=np.linspace(0.9,0.99,4)
i=0
j=0
k=-1
# for i in range(2,3):#len(res)):
#     print(i)
#     for j in range(2,3):#len(drt)):
#         for k in range(3,4):#len(w)):
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
base="/home/uzair/PycharmProjects/Unfolding/data/diffusionSimulationsBunching_res-"
path=base+str(int(res[i]*10000))+"mm_drt-"+str(int(drt[j]*100))+"+w-"+str(int(w[k]*100))+"/"
if not os.path.exists(path):
    os.mkdir(path)

bvals = "/home/uzair/PycharmProjects/Unfolding/data/101006/Diffusion/Diffusion/bvals"
bvecs = "/home/uzair/PycharmProjects/Unfolding/data/101006/Diffusion/Diffusion/bvecs"

sim=simulateDiffusion.simulateDiffusion(phi,dphi,phiInv,Uparams,L1L2L3,bvals=bvals,bvecs=bvecs,N0=Nx)
sim.simulate(path)


sub=unfoldSubject.unfoldSubject()
#sub.loadCoordinates(path="K:\\Datasets\\diffusionSimulations\\",prefix="")
sub.loadCoordinates(path=path,prefix="")
sub.coords.computeGradDev()
sub.loadDiffusion(path)
sub.pushToUnfold(type='diffusion')
#sub.diffNoGradDev()
upath=path+"Unfolded/"
if not os.path.exists(upath):
    os.mkdir(upath)
shutil.copyfile(path+"bvals",upath+"bvals")
shutil.copyfile(path + "bvecs", upath + "bvecs")
nib.save(sub.diffUnfold.vol,upath+'data.nii.gz')
nib.save(sub.coords.gradDevUVW_nii,upath+'grad_dev.nii.gz')
nib.save(sub.coords.gradDevXYZ_nii,upath+'grad_devXYZ.nii.gz')
temp_mask=sub.diffUnfold.mask.get_fdata()
temp_mask[:,:,-1]=np.NaN
temp_mask=nib.Nifti1Image(temp_mask,sub.diffUnfold.mask.affine)
nib.save(sub.diffUnfold.mask,upath+'nodif_brain_mask.nii.gz')
#


import numpy as np
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
def loc_track(path,default_sphere, coords=None,npath=None,UParams=None):
    data, affine = load_nifti(path + 'data.nii.gz')
    data[np.isnan(data) == 1] = 0
    mask, affine = load_nifti(path + 'nodif_brain_mask.nii.gz')
    mask[np.isnan(mask) == 1] = 0
    mask[:,:,1:]=0
    stopper=copy.deepcopy(mask)
    stopper[:,:,:]=1
    gtab = gradient_table(path + 'bvals', path + 'bvecs')

    csa_model = CsaOdfModel(gtab, smooth=1, sh_order=12)
    peaks = peaks_from_model(csa_model, data, default_sphere,
                             relative_peak_threshold=0.99,
                             min_separation_angle=25,
                             mask=mask)
    peaks.ang_thr=90
    if os.path.exists(path + 'grad_dev.nii.gz'):
        gd, affine_g = load_nifti(path + 'grad_dev.nii.gz')
        nmask, naffine = load_nifti(npath + 'nodif_brain_mask.nii.gz')
        nmask[np.isnan(nmask) == 1] = 0
        nmask[:, :, 1:] = 0
        seedss = copy.deepcopy(nmask)
        seedss = utils.seeds_from_mask(seedss, naffine, [2, 2, 2])
        useed = []
        UParams=coords.Uparams
        for seed in seedss:
            us = coords.rFUa_xyz(seed[0], seed[1], seed[2])
            vs = coords.rFVa_xyz(seed[0], seed[1], seed[2])
            ws = coords.rFWa_xyz(seed[0], seed[1], seed[2])
            condition = us >= UParams.min_a and us <= UParams.max_a and vs >= UParams.min_b and vs <= UParams.max_b \
                        and ws >= UParams.min_c and ws <= UParams.max_c
            print(condition)
            if condition==True:
                useed.append([float(us), float(vs), float(ws)])
                #print(useed)
        seeds = np.asarray(useed)

    else:
        gd=None
        seedss = copy.deepcopy(mask)
        seeds = utils.seeds_from_mask(seedss, affine, [2, 2, 2])


    stopping_criterion = BinaryStoppingCriterion(stopper)
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



            #path="/home/uzair/PycharmProjects/Unfolding/data/diffusionSimulations_nonConformal_scale/"
ntracking = loc_track(path ,default_sphere)
coords = coordinates.coordinates(path,'')
utracking = loc_track(path+'Unfolded/',default_sphere,coords=coords,npath=path,UParams=Uparams)
u2n_streamlines=unfold2nativeStreamlines(utracking,coords)
streamline.save_vtk_streamlines(ntracking.streamlines,
                                path+"native_streamlines.vtk")
streamline.save_vtk_streamlines(utracking.streamlines,
                                path + "Unfolded/unfold_streamlines.vtk")
streamline.save_vtk_streamlines(u2n_streamlines,
                    path + "from_unfold_streamlines.vtk")


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


#make native to unfold seeds
mask, affine = load_nifti(path + 'nodif_brain_mask.nii.gz')
mask[np.isnan(mask) == 1] = 0
mask[:,:,1:]=0
seedss = copy.deepcopy(mask)
seeds = utils.seeds_from_mask(seedss, affine, [2, 2, 2])
useed=[]
for seed in seeds:
    us=coords.rFUa_xyz(seed[0],seed[1],seed[2])
    vs = coords.rFVa_xyz(seed[0], seed[1], seed[2])
    ws = coords.rFWa_xyz(seed[0], seed[1], seed[2])
    useed.append([us,vs,ws])
useed=np.asarray(useed)