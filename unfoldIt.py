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

#ress=['close_high/', 'close_medium/', 'close_low/']
ress=['_nonConformal/']
for r in range(0,len(ress)):
    base="/home/uzair/PycharmProjects/Unfolding/data/diffusionSimulations"
    path=base+ress[r]
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
    nib.save(sub.diffUnfold.vol,upath+'diffunfold.nii.gz')
    nib.save(sub.coords.gradDevUVW_nii,upath+'grad_devUVW.nii.gz')
    nib.save(sub.coords.gradDevXYZ_nii,upath+'grad_devXYZ.nii.gz')
    temp_mask=sub.diffUnfold.mask.get_fdata()
    temp_mask[:,:,-1]=np.NaN
    temp_mask=nib.Nifti1Image(temp_mask,sub.diffUnfold.mask.affine)
    #nib.save(sub.diffUnfold.mask,upath+'nodif_brain_mask.nii.gz')
    nib.save(temp_mask,upath+'nodif_brain_mask.nii.gz')
    nib.save(sub.coords.X_uvwa_nii,upath+'X.nii.gz')
    nib.save(sub.coords.Y_uvwa_nii,upath+'Y.nii.gz')
    nib.save(sub.coords.Z_uvwa_nii,upath+'Z.nii.gz')


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
