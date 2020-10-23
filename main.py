import coordinates
import matplotlib.pyplot as plt
import numpy as np
import nibabel as nib
import simulateDiffusion
import diffusion

# coords=coordinates.coordinates()
#
# coords.loadCoordinates(path="K:\\Datasets\\sampleNiftiCoordinates\\",prefix="")
#
# coords.initialize()
# coords.meanArcLength()
#
# nib.save(coords.X_uvw_nii,"X_uvw.nii.gz")
#
#
# plt.imshow(coords.X_uvw_nii.get_data()[:,:,3])

# diffvol=diffusion.diffVolume()
# diffvol.getVolume(folder="K:\\Datasets\\HCP_diffusion\\101006\\Diffusion\\Diffusion\\")
# diffvol.shells()

bvals = "K:\\Datasets\\HCP_diffusion\\101006\\Diffusion\\Diffusion\\bvals"
bvecs = "K:\\Datasets\\HCP_diffusion\\101006\\Diffusion\\Diffusion\\bvecs"
sim=simulateDiffusion.simulateDiffusion(1,1,1,bvals=bvals,bvecs=bvecs)
sim.simulate('K:\\Datasets\\diffusionSimulations\\')
#sim.plotTangVecs()


sim.diffusionSignal()

b = np.arange(0,len(sim.bvals[1]))
x = np.linspace(sim.Nparams.max_a,sim.Nparams.max_a,sim.Nparams.Na)
y = np.linspace(sim.Nparams.max_b, sim.Nparams.max_b, sim.Nparams.Nb)
z = np.linspace(sim.Nparams.max_c, sim.Nparams.max_c, sim.Nparams.Nc)
X,Y,Z,B = np.meshgrid(x,y,z,b,indexing='ij')

