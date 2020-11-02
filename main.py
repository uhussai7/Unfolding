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

sub=unfoldSubject.unfoldSubject()
#sub.loadCoordinates(path="K:\\Datasets\\diffusionSimulations\\",prefix="")
sub.loadCoordinates(path="/home/uzair/PycharmProjects/Unfolding/data/diffusionSimulations/",prefix="")
sub.coords.computeGradDev()
sub.loadDiffusion("/home/uzair/PycharmProjects/Unfolding/data/diffusionSimulations/")

sub.diffUnfold=sub.pushToUnfold(sub.diff.vol,type='diffusion')

sub.diffNoGradDev()

upath="/home/uzair/PycharmProjects/Unfolding/data/diffusionSimulations/Unfolded/"
nib.save(sub.diffUnfold.vol,upath+'diffunfold.nii.gz')
nib.save(sub.coords.gradDevUVW_nii,upath+'grad_devUVW.nii.gz')
nib.save(sub.coords.gradDevXYZ_nii,upath+'grad_devXYZ.nii.gz')
nib.save(sub.diffUnfold.mask,upath+'nodif_brain_mask.nii.gz')
nib.save(sub.coords.X_uvw_nii,upath+'X.nii.gz')
nib.save(sub.coords.Y_uvw_nii,upath+'Y.nii.gz')
nib.save(sub.coords.Z_uvw_nii,upath+'Z.nii.gz')

diff_nograd_nii=nib.Nifti1Image(sub.diff_nograd,sub.diffUnfold.vol.affine)
nib.save(diff_nograd_nii,upath+'diff_nograd.nii.gz')
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

