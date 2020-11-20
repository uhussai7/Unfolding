import nipype.interfaces.camino as cmon
import nibabel as nib



upath="/home/uzair/PycharmProjects/Unfolding/data/diffusionSimulations/Unfolded/"

fsl2scheme=cmon.FSL2Scheme()
fsl2scheme.inputs.bvec_file=upath+'bvecs'
fsl2scheme.inputs.bval_file=upath+'bvals'
fsl2scheme.output_spec('A.scheme')
fsl2scheme.run()

data_nii=nib.load(upath+'diff_nograd.nii.gz')
data_vox=cmon.Image2Voxel(data_nii)


fit = cmon.ModelFit()
fit.model = 'dt'
