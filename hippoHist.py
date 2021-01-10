import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import polar
import copy
import matplotlib.image as mpimg
from matplotlib.offsetbox import TextArea, DrawingArea, OffsetImage, AnnotationBbox


# #take graham graddev and make a rotation only one
# graddev_nii=nib.load('./data/oldUnfold_graham/oldUnfold/Unfolded/grad_dev.nii.gz')
# sz=graddev_nii.get_fdata().shape
# #graddev = np.asarray(graddev_nii.get_fdata().reshape(-1,9,order='F'))
# graddev = np.asarray(graddev_nii.get_fdata().reshape((sz[0:3])+(3, 3),order='F'))
# graddev=graddev.reshape(-1,3,3,order='F')
#
# temp_graddev = copy.deepcopy(graddev)
# graddev[:] = np.NaN
# for g in range(0, graddev.shape[0]):
#     m = temp_graddev[g, :, :] + np.eye(3)
#     if (np.isnan(sum(sum(m))) == 0):
#         # print(sum(sum(m)))
#         rot, deform = polar(m)
#         graddev[g, :, :] = rot - np.eye(3)
#
# graddev=graddev.reshape((sz[0:3])+(3, 3),order='F')
# graddev=graddev.reshape((sz[0:3])+(9,),order='F')
# graddev=nib.Nifti1Image(graddev,graddev_nii.affine)
# nib.save(graddev,'./data/oldUnfold_graham/oldUnfold/Unfolded/grad_dev_rot.nii.gz')



dti_FA_n=nib.load('./data/oldUnfold/DiffusionCropped/Native_hiRes/Crop/L/cropped/dti_FA.nii.gz')
dti_FA_u_full=nib.load('./data/oldUnfold/DiffusionCropped/Native_hiRes/Crop/L/cropped/Unfolded/dti_FA.nii.gz')
dti_FA_u_rot=nib.load('./data/oldUnfold/DiffusionCropped/Native_hiRes/Crop/L/cropped/Unfolded/dti_rot_FA.nii.gz')


fa_n=dti_FA_n.get_fdata()[dti_FA_n.get_fdata()>0]
fa_u_full=dti_FA_u_full.get_fdata()[dti_FA_u_full.get_fdata()>0]
fa_u_rot=dti_FA_u_rot.get_fdata()[dti_FA_u_rot.get_fdata()>0]

plt.figure()
plt.title('FA histogram')
plt.xlabel('FA')
plt.hist(fa_n,40,histtype='step',linewidth=3, density=True,alpha=0.99,label='Cartesian')
plt.hist(fa_u_full,40,histtype='step',linewidth=3,density=True,alpha=0.99,label='Conformal')
plt.hist(fa_u_rot,40,histtype='step',linewidth=3,density=True,alpha=0.99,label='Conformal, rotation only')
plt.legend()


fig, ax = plt.subplots(figsize=(16,10))
ax.axis('off')

FA_img=mpimg.imread('./data/plots/FA.png')
NAT_img=mpimg.imread('./data/plots/hippoNative.png')

nat_ax=fig.add_axes([0.05,0.55,0.92,0.4])
nat_ax.axis('off')
nat_ax.imshow(NAT_img)
ax.text(0.135,0.95,'a)',fontsize=22,transform=fig.transFigure)
ax.text(0.455,0.95,'b)',fontsize=22,transform=fig.transFigure)
ax.text(0.675,0.95,'c)',fontsize=22,transform=fig.transFigure)

imax=fig.add_axes([0.08,0.05,0.62,0.45])
imsh=imax.imshow(FA_img,extent=(-0., FA_img.shape[1], -0., FA_img.shape[0]),origin='upper')
imax.axis('off')
imax.axes.text(FA_img.shape[1]/2,-75,'Proximal',fontsize=25,horizontalalignment='center')
imax.axes.text(FA_img.shape[1]/2,FA_img.shape[0]+25,'Distal',fontsize=25,horizontalalignment='center')
imax.axes.text(-25,FA_img.shape[0]/2,'Posterior',fontsize=25,horizontalalignment='center',rotation=90,
               rotation_mode='anchor')
imax.axes.text(FA_img.shape[1]+75,FA_img.shape[0]/2,'Anterior',fontsize=25,horizontalalignment='center',rotation=90,
               rotation_mode='anchor')


histax=fig.add_axes([0.78,0.11,0.2,0.33])
histax.hist(fa_n,40,histtype='step',linewidth=3, density=True,alpha=0.99,label='Cartesian')
histax.hist(fa_u_full,40,histtype='step',linewidth=3,density=True,alpha=0.99,label='Conformal')
histax.hist(fa_u_rot,40,histtype='step',linewidth=3,density=True,alpha=0.99,label='Conformal, rotaion only')
ax.text(0.05,0.45,'d)',fontsize=22,transform=fig.transFigure)
ax.text(0.76,0.46,'e)',fontsize=22,transform=fig.transFigure)
histax.tick_params(axis='both',labelsize=20)
histax.set_title('FA histogram',fontsize=22)
histax.set_xlabel('FA',fontsize=22)
