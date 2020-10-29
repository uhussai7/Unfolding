import coordinates
import matplotlib.pyplot as plt
import numpy as np
import nibabel as nib
import simulateDiffusion
import diffusion
import unfoldSubject

sub=unfoldSubject.unfoldSubject()
sub.loadCoordinates(path="K:\\Datasets\\diffusionSimulations\\",prefix="")
sub.coords.computeGradDev()
sub.loadDiffusion("K:\\Datasets\\diffusionSimulations\\")

sub.diffUnfold=sub.pushToUnfold(sub.diff.vol,type='diffusion')

sub.diffNoGradDev()

nib.save(sub.diffUnfold.vol,'diffunfold.nii.gz')
nib.save(sub.coords.gradDevUVW_nii,'grad_dev.nii.gz')
nib.save(sub.coords.gradDevXYZ_nii,'grad_devXYZ.nii.gz')

diff_nograd_nii=nib.Nifti1Image(sub.diff_nograd,sub.diffUnfold.vol.affine)
nib.save(diff_nograd_nii,'diff_nograd.nii.gz')
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

