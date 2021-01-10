import coordinates
import matplotlib.pyplot as plt
import numpy as np
import nibabel as nib
import simulateDiffusion
import diffusion
import unfoldSubject
from pyshtools.shtools import SHExpandLSQ
from pyshtools.shtools import MakeGridPoint
from pyshtools.shtools import MakeGridPointC
import os
import shutil
from dipy.align.reslice import reslice
from dipy.data import get_fnames
import copy
from nibabel.processing import resample_from_to




#get all the images to reasonable resolution
#load, resize and save coordinates
# base="/home/uzair/PycharmProjects/Unfolding/data/oldUnfold/DiffusionCropped/Native_hiRes/Crop/L/cropped/"
# U_nii=nib.load(base+'U.nii.gz')
# V_nii=nib.load(base+'V.nii.gz')
# W_nii=nib.load(base+'W.nii.gz')
# data_nii=nib.load(base+'data.nii.gz')
#
# temp=copy.deepcopy( data_nii.get_fdata()[:,:,:,0])
# temp=nib.Nifti1Image(temp,data_nii.affine)
#
# U_rsmpl_nii=resample_from_to(U_nii,temp,order=0)
# V_rsmpl_nii=resample_from_to(V_nii,temp,order=0)
# W_rsmpl_nii=resample_from_to(W_nii,temp,order=0)
#
# nib.save(U_rsmpl_nii,base+'U.nii.gz')
# nib.save(V_rsmpl_nii,base+'V.nii.gz')
# nib.save(W_rsmpl_nii,base+'W.nii.gz')


#need to mask diffusion data
# base="/home/uzair/PycharmProjects/Unfolding/data/oldUnfold/DiffusionCropped/Native_hiRes/Crop/L/cropped/"
# U_nii=nib.load(base+'U.nii.gz')
# V_nii=nib.load(base+'V.nii.gz')
# W_nii=nib.load(base+'W.nii.gz')
# data_nii=nib.load(base+'data.nii.gz')
#
# U=U_nii.get_fdata()
# V=V_nii.get_fdata()
# W=W_nii.get_fdata()
# U[U==0]=np.NaN
# V[np.isnan(U)==1]=np.NaN
# W[np.isnan(U)==1]=np.NaN
# mask=copy.deepcopy(U)
# mask[np.isnan(mask)==0]=1
#
# data=copy.deepcopy(data_nii.get_fdata())
# for i in range(0,data.shape[-1]):
#     temp=copy.deepcopy(data[:,:,:,i])
#     temp[mask!=1]=np.NaN
#     data[:,:,:,i]=temp
#
# nib.save(nib.Nifti1Image(U,U_nii.affine),base+'U.nii.gz')
# nib.save(nib.Nifti1Image(V,U_nii.affine),base+'V.nii.gz')
# nib.save(nib.Nifti1Image(W,U_nii.affine),base+'W.nii.gz')
# nib.save(nib.Nifti1Image(mask,U_nii.affine),base+'nodif_brain_mask.nii.gz')
# nib.save(nib.Nifti1Image(data,U_nii.affine),base+'data.nii.gz')



#unfold the hippocampus
sub=unfoldSubject.unfoldSubject()
base="/home/uzair/PycharmProjects/Unfolding/data/oldUnfold/DiffusionCropped/Native_hiRes/Crop/L/cropped/"
sub.loadCoordinates(path=base,prefix='')
print('computing gradev')
sub.coords.computeGradDev()
# print('loading diffusion')
# sub.loadDiffusion(base)
# print('unfolding diffusion')
# sub.pushToUnfold(type='diffusion')
path=base
upath=path+"Unfolded/"
# if not os.path.exists(upath):
#     os.mkdir(upath)
# shutil.copyfile(path+"bvals",upath+"bvals")
# shutil.copyfile(path + "bvecs", upath + "bvecs")
# nib.save(sub.diffUnfold.vol,upath+'data.nii.gz')
nib.save(sub.coords.gradDevUVW_nii,upath+'grad_dev.nii.gz')
nib.save(sub.coords.gradDevXYZ_nii,upath+'grad_devXYZ.nii.gz')
temp_mask=sub.diffUnfold.mask.get_fdata()
temp_mask=nib.Nifti1Image(temp_mask,sub.diffUnfold.mask.affine)
nib.save(sub.diffUnfold.mask,upath+'nodif_brain_mask.nii.gz')
#
temp=copy.deepcopy(sub.coords.gradDevUVW_nii.get_fdata()[:,:,:,:])
temp=np.sum(temp,-1)
temp[np.isnan(temp)==0]=1
temp=nib.Nifti1Image(temp,sub.coords.gradDevUVW_nii.affine)
nib.save(temp,base+'Unfolded/nodif_brain_mask.nii.gz')


# scale=0.05
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
#     int(w[k] * 100)) + "/"
#
#             #path="/home/uzair/PycharmProjects/Unfolding/data/diffusionSimulations_nonConformal/"
#             sub=unfoldSubject.unfoldSubject()
#             #sub.loadCoordinates(path="K:\\Datasets\\diffusionSimulations\\",prefix="")
#             sub.loadCoordinates(path=path,prefix="")
#             sub.coords.computeGradDev()
#             sub.loadDiffusion(path)
#             sub.pushToUnfold(type='diffusion')
#             #sub.diffNoGradDev()
#             upath=path+"Unfolded/"
#             if not os.path.exists(upath):
#                 os.mkdir(upath)
#             shutil.copyfile(path+"bvals",upath+"bvals")
#             shutil.copyfile(path + "bvecs", upath + "bvecs")
#             nib.save(sub.diffUnfold.vol,upath+'data.nii.gz')
#             nib.save(sub.coords.gradDevUVW_nii,upath+'grad_dev.nii.gz')
#             nib.save(sub.coords.gradDevXYZ_nii,upath+'grad_devXYZ.nii.gz')
#             temp_mask=sub.diffUnfold.mask.get_fdata()
#             temp_mask[:,:,-1]=np.NaN
#             temp_mask=nib.Nifti1Image(temp_mask,sub.diffUnfold.mask.affine)
#             nib.save(sub.diffUnfold.mask,upath+'nodif_brain_mask.nii.gz')
                        # nib.save(temp_mask,upath+'nodif_brain_mask.nii.gz')
                        # nib.save(sub.coords.X_uvwa_nii,upath+'X.nii.gz')
                        # nib.save(sub.coords.Y_uvwa_nii,upath+'Y.nii.gz')
                        # nib.save(sub.coords.Z_uvwa_nii,upath+'Z.nii.gz')


#nib.save(sub.diffUnfoldNograd.vol,upath+'diff_nograd.nii.gz')


# U_nii=sub.coords.U_xyz_nii.get_fdata()
# V_nii=sub.coords.V_xyz_nii.get_fdata()
# W_nii=sub.coords.W_xyz_nii.get_fdata()
#
# U_nii=sub.coords.mean_u * (U_nii- np.nanmin(U_nii)) / np.nanmax(U_nii)
# V_nii=sub.coords.mean_v * (V_nii- np.nanmin(V_nii)) / np.nanmax(V_nii)
# W_nii=sub.coords.mean_u * (W_nii- np.nanmin(W_nii)) / np.nanmax(W_nii)
#
# U_nii=nib.Nifti1Image(U_nii,sub.coords.U_xyz_nii.affine)
# V_nii=nib.Nifti1Image(V_nii,sub.coords.V_xyz_nii.affine)
# W_nii=nib.Nifti1Image(W_nii,sub.coords.W_xyz_nii.affine)
# npath="/home/uzair/PycharmProjects/Unfolding/data/diffusionSimulations/"
# nib.save(U_nii,npath+'Ua.nii.gz')
# nib.save(V_nii,npath+'Va.nii.gz')
# nib.save(W_nii,npath+'Wa.nii.gz')


#coords=coordinates.coordinates(path="K:\\Datasets\\diffusionSimulations\\",prefix="")
#coords.computeGradDev()
#
# coords.meanArcLength()
#
# nib.save(coords.X_uvw_nii,"X_uvw.nii.gz")
#
#
# plt.imshow(coords.X_uvw_nii.get_data()[:,:,3])

# diffvol=diffusion.diffVolume()
# diffvol.getVolume(folder="K:\\Datasets\\HCP_diffusion\\101006\\Diffusion\\Diffusion\\")
# diffvol.shells()



# b = np.arange(0,len(sim.bvals[1]))
# x = np.linspace(sim.Nparams.max_a,sim.Nparams.max_a,sim.Nparams.Na)
# y = np.linspace(sim.Nparams.max_b, sim.Nparams.max_b, sim.Nparams.Nb)
# z = np.linspace(sim.Nparams.max_c, sim.Nparams.max_c, sim.Nparams.Nc)
# X,Y,Z,B = np.meshgrid(x,y,z,b,indexing='ij')



##-------spherical harmonic testing-------##
#extract signal at point
# def cart2latlon(x,y,z):
#     R=x*x+y*y+z*z
#     lat = np.arcsin(z / R)
#     lon = np.arctan2(y, x)
#     return lat, lon
#
# S=sub.diff.vol.get_fdata()[10,10,2,sub.diff.inds[1]]
# bvecs=np.asarray(sub.diff.bvecs_hemi_cart[1])
# x=bvecs[:,0]
# y=bvecs[:,1]
# z=bvecs[:,2]
#
# lat,lon=cart2latlon(x,y,z)
# lmax=100
# cilm=SHExpandLSQ(S,lat,lon,lmax)
# sample=np.zeros(len(lat))
# for i in range(0,len(lat)):
#     sample[i]=MakeGridPoint(cilm[0],lat[i],lon[i])

