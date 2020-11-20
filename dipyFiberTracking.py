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

#subdivide sphere
default_sphere=default_sphere.subdivide()
default_sphere=default_sphere.subdivide()
default_sphere=default_sphere.subdivide()
default_sphere.vertices=np.append(default_sphere.vertices,[[1,0,0]])
default_sphere.vertices=np.append(default_sphere.vertices,[[-1,0,0]])
default_sphere.vertices=np.append(default_sphere.vertices,[[0,1,0]])
default_sphere.vertices=np.append(default_sphere.vertices,[[0,-1,0]])
default_sphere.vertices=default_sphere.vertices.reshape([-1,3])


#loadfiles
#native
npath="/home/uzair/PycharmProjects/Unfolding/data/diffusionSimulations_nonConformal/"
ndata,naffine=load_nifti(npath+'data.nii.gz')
ndata[np.isnan(ndata)==1]=0
nmask,affine1 = load_nifti(npath+'nodif_brain_mask.nii.gz')
nmask[np.isnan(nmask)==1]=0
nmask[:]=1
#nmask[:,:,1:]=0
coords = coordinates.coordinates(npath,'')
nbvals, nbvecs = read_bvals_bvecs(npath+'bvals', npath+'bvecs')
ngtab = gradient_table(npath+'bvals', npath+'bvecs')
csa_model = CsaOdfModel(ngtab, sh_order=10)
npeaks = peaks_from_model(csa_model, ndata, default_sphere,
                              relative_peak_threshold=0.99,
                              min_separation_angle=25,
                              mask=nmask)
seedss=copy.deepcopy(nmask)
seeds=utils.seeds_from_mask(seedss,naffine,[1,1,1])
stopping_criterion=ThresholdStoppingCriterion(nmask,0)
ntracking=tracking(npeaks,stopping_criterion,seeds,naffine,sphere=default_sphere)
ntracking.localTracking()
#ntracking.plot()

#unfold
upath="/home/uzair/PycharmProjects/Unfolding/data/diffusionSimulations_nonConformal/Unfolded/"
udata,uaffine=load_nifti(upath+'diffunfold.nii.gz')
udata[np.isnan(udata)==1]=0
umask,affine1 = load_nifti(upath+'nodif_brain_mask.nii.gz')
umask[np.isnan(umask)==1]=0
umask[:]=1
#nmask[:,:,1:]=0


ubvals, ubvecs = read_bvals_bvecs(upath+'bvals', upath+'bvecs')
ugtab = gradient_table(upath+'bvals', upath+'bvecs')
gd,affine_g=load_nifti(upath+'grad_devUVW.nii.gz')
csa_model = CsaOdfModel(ugtab, sh_order=10)
upeaks = peaks_from_model(csa_model, udata, default_sphere,
                              relative_peak_threshold=0.99,
                              min_separation_angle=25,
                              mask=umask)
useedss=copy.deepcopy(umask)
useeds=utils.seeds_from_mask(useedss,uaffine,[1,1,1])
ustopping_criterion=ThresholdStoppingCriterion(umask,0)
utracking=tracking(upeaks,ustopping_criterion,useeds,uaffine,graddev=gd, sphere=default_sphere)
utracking.localTracking()
#utracking.plot()


a_streams=unfoldStreamlines(ntracking.streamlines,
                            utracking.streamlines,
                            ntracking.NpointsPerLine,
                            utracking.NpointsPerLine,
                            coords)



points,X=getPointsData(coords.X_uvwa_nii)
points,Y=getPointsData(coords.Y_uvwa_nii)
points,Z=getPointsData(coords.Z_uvwa_nii)



# points=np.asarray([coords.Ua,
#                    coords.Va,
#                    coords.Wa]).transpose()

allLines= utracking.streamlines.get_data()
# x = griddata(points, coords.X, allLines)
# y = griddata(points, coords.Y, allLines)
# z = griddata(points, coords.Z, allLines)

# x = griddata(points, X, allLines)
# y = griddata(points, Y, allLines)
# z = griddata(points, Z, allLines)

# coords.nativeUnfold()
# x=coords.FX_uvwa(allLines)
# y=coords.FY_uvwa(allLines)
# z=coords.FZ_uvwa(allLines)
coords.meanArcLength()

x=coords.rFX_uvwa(allLines[:,0],allLines[:,1],allLines[:,2])
y=coords.rFY_uvwa(allLines[:,0],allLines[:,1],allLines[:,2])
z=coords.rFZ_uvwa(allLines[:,0],allLines[:,1],allLines[:,2])

allLines=np.asarray([x,y,z]).T


pointsPerLine=utracking.NpointsPerLine
streamlines=[]
first=0
for i in range(0,len(pointsPerLine)-1):
    templine=[]
    points = allLines[first:first +pointsPerLine[i]]
    for p in range(0,pointsPerLine[i]):
         #if( np.isnan(np.sum(points[p]))==0):
        templine.append(points[p])
    if(len(templine)>1):# and len(templine) == pointsPerLine[i]):
        streamlines.append(templine)
    first=first+pointsPerLine[i]

# fig=plt.figure()
# ax=fig.add_subplot(111,projection='3d')
# for l in range(0,len(streamlines)):
#     line=np.asarray(streamlines[l])
#     ax.plot(line[:,0],line[:,1],line[:,2])

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



#a_streams.moveStreamlines2Native()


# points = np.asarray([coords.Ua,
#                      coords.Va,
#                      coords.Wa]).transpose()
# #
# #
# allLines=utracking.streamlines.get_data()
# x = griddata(points, coords.X, allLines)
# y = griddata(points, coords.Y, allLines)
# z = griddata(points, coords.Z, allLines)
# #
# # test=utracking.streamlines.get_data()
#
# p=utracking.streamlines[1500]
# x = griddata(points, coords.X, test)
# y = griddata(points, coords.Y, test)
# z = griddata(points, coords.Z, test)

#number of points in each lines

#
# streamlines=[]
# templine=[]
# for l in range(0,len(utracking.streamlines)):
#     line=utracking.streamlines[l]
#     print(l)
#     if len(line)>60:
#         x = griddata(points, coords.X, line)
#         y = griddata(points, coords.Y, line)
#         z = griddata(points, coords.Z, line)
#         templine.append([x, y, z])
#         #streamlines.append(templine)
#
#
# inds=np.asarray(np.where(coords.Ua_xyz_nii))



# #path="/home/uzair/PycharmProjects/Unfolding/data/diffusionSimulations/"
# path="/home/uzair/PycharmProjects/Unfolding/data/diffusionSimulations/Unfolded/"
# #data,affine=load_nifti(path+'data.nii.gz')
# data,affine=load_nifti(path+'diffunfold.nii.gz')
# data[np.isnan(data)==1]=0
# mask,affine1 = load_nifti(path+'nodif_brain_mask.nii.gz')
# mask[:,:,1:]=0
# seedss=copy.deepcopy(mask)
# mid0=round(mask.shape[0]/2)
# mid1=round(mask.shape[1]/2)
# #seedss[:]=0
# seedss[0:4,20:,:]=1
# bvals, bvecs = read_bvals_bvecs(path+'bvals', path+'bvecs')
# gtab = gradient_table(bvals, bvecs)
# gd,affine_g=load_nifti(path+'grad_devUVW.nii.gz')
# gd = gd.reshape(gd.shape[0:3]+ (3, 3), order='F')
# for i in range(gd.shape[0]):
#     for j in range(gd.shape[1]):
#         for k in range(gd.shape[2]):
#             gd[i,j,k,:,:]=gd[i,j,k,:,:]+np.eye(3)
#
# dti_V1,affine1=load_nifti(path+'dti_V1.nii.gz')
#
# # tenmodel=dti.TensorModel(gtab)
# # tenfit=tenmodel.fit(data,mask)
#
#
#
# default_sphere=default_sphere.subdivide()
# default_sphere=default_sphere.subdivide()
# default_sphere=default_sphere.subdivide()
#
# default_sphere.vertices=np.append(default_sphere.vertices,[[1,0,0]])
# default_sphere.vertices=np.append(default_sphere.vertices,[[-1,0,0]])
# default_sphere.vertices=np.append(default_sphere.vertices,[[0,1,0]])
# default_sphere.vertices=np.append(default_sphere.vertices,[[0,-1,0]])
#
# default_sphere.vertices=default_sphere.vertices.reshape([-1,3])
#
#
# csa_model = CsaOdfModel(gtab, sh_order=10)
# csa_peaks = peaks_from_model(csa_model, data, default_sphere,
#                              relative_peak_threshold=1,
#                              min_separation_angle=40,
#                              mask=mask)
#
#
# response, ratio = auto_response_ssst(gtab, data, roi_radii=10, fa_thr=0.7)
# csd_model = ConstrainedSphericalDeconvModel(gtab, response=response,sh_order=6)
# csd_fit = csd_model.fit(data, mask=mask)
#
# # dg = DeterministicMaximumDirectionGetter.from_shcoeff(csd_fit.shm_coeff,
# #                                                       max_angle=30.,
# #                                                       sphere=default_sphere)
#
#
# #adjust peaks with graddev
# #1. make kdtree for sphere
# #2. take peak vec and push it with jacobian matrix
# #3. find closest vector on sphere after pushing
# #sphereTree=KDTree(default_sphere.vertices)
#
#
# new_peak_dirs=np.einsum('ijkab,ijkvb->ijkva',gd,csa_peaks.peak_dirs)
#
# #new_peak_dirs=csa_peaks.peak_dirs
#
# # new_peak_dirs=dti_V1
#
# fig=plt.figure()
# ax=fig.add_subplot(111,projection='3d')
# for l in range(0,len(streamlines)):
#     line=np.asarray(streamlines[l])
#     ax.plot(line[:,0],line[:,1],line[:,2])

# #ax.scatter(new_peak_dirs[:,0],new_peak_dirs[:,1],new_peak_dirs[:,2])
# A=np.arange(0,new_peak_dirs.shape[0])
# B=np.arange(0,new_peak_dirs.shape[1])
# C=np.arange(0,new_peak_dirs.shape[2])
# A,B,C = np.meshgrid(A,B,C,indexing='ij')
# ax.set_zlim(-10,10)
# ax.quiver(A,B,C,new_peak_dirs[:,:,:,0,0],new_peak_dirs[:,:,:,0,1],new_peak_dirs[:,:,:,0,2],length=10)
#
# #
# #
# #dis,inds=sphereTree.query(new_peak_dirs.reshape([-1,3]),1)
# #inds[inds==len(default_sphere.vertices)]=-1
# #
# #
# #inds= inds.reshape(csa_peaks.peak_indices.shape)
#
# #inds=np.random.random(csa_peaks.peak_indices.shape)*(len(default_sphere.vertices)-1)
# #inds=inds.round()
#
# #csa_peaks.peak_indices=
# #csa_peaks.peak_dirs[:]=1
#
# #csa_peaks.initial_direction=0
# #csa_peaks.peak_dirs=peak_dirs
# # #csa_peaks.peak_indices=copy.deepcopy(inds)
#
# #dti_peaks = peaks_from_model(tenmodel, data,default_sphere,
#                              # relative_peak_threshold=0.8,
#                              # min_separation_angle=45)
#                              #
#
# peak_indices=np.zeros(csa_peaks.peak_indices.shape)
# peak_indices=peak_indices.reshape([-1,5])
# new_peak_dirs=new_peak_dirs.reshape([-1,5,3])
# # new_peak_dirs=dti_V1.reshape([-1,3])
# # #
# for i in range(0,peak_indices.shape[0]):
#     for k in range(0,5):
#         peak_indices[i,k]=default_sphere.find_closest(new_peak_dirs[i,k,:])
# #
# # #csa_peaks.peak_dirs=new_peak_dirs.reshape(csa_peaks.peak_dirs.shape)
# csa_peaks.peak_indices=peak_indices.reshape(csa_peaks.peak_indices.shape)
# # #csa_peaks.peak_values=csa_peaks.peak_indices
#
# stopping_criterion=ThresholdStoppingCriterion(mask,0)
# seeds=utils.seeds_from_mask(seedss,affine,[2,2,2])
#
# streamlines_generator=LocalTracking(csa_peaks,stopping_criterion,seeds,affine,step_size=affine[0,0]/2,maxlen=5000)
# streamlines = Streamlines(streamlines_generator)
#
#


#
#
# # lines=np.asarray(streamlines)
# #
# # fig = plt.figure()
# # ax = plt.axes(projection='3d')
# # ax.plot3D(lines[:,0,0],lines[:,0,1],lines[:,0,2])
#
# # for stream in streamlines:
# #     print(len(stream))
# #
# # def initial_direction(arg):
# #     out=peaks.inital_direction(arg)
# #     print(arg)
# #     print(out)
# #     return(out)

