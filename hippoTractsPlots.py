import numpy as np
from dipy.viz import window, actor, has_fury
from dipy.viz import colormap
import matplotlib.pyplot as plt
from dipy.tracking.utils import length
from dipy.io import streamline
from dipy.tracking.metrics import mean_curvature
import matplotlib.image as mpimg
from PIL import Image
import nibabel as nib










#---------------------------------------Previous Plots------------------------------------#
# import numpy as np
# from dipy.viz import window, actor, has_fury
# from dipy.viz import colormap
# import matplotlib.pyplot as plt
# from dipy.tracking.utils import length
# from dipy.io import streamline
# from dipy.tracking.metrics import mean_curvature
# import matplotlib.image as mpimg
# from PIL import Image
# import nibabel as nib
#
# def plot_streamlines(streamlines):
#     if has_fury:
#         # Prepare the display objects.
#         color = colormap.line_colors(streamlines)
#
#         streamlines_actor = actor.line(streamlines,
#                                        colormap.line_colors(streamlines))
#
#         # Create the 3D display.
#         scene = window.Scene()
#         scene.add(streamlines_actor)
#
#         # Save still images for this static example. Or for interactivity use
#         window.show(scene)
#
# nlines=streamline.load_vtk_streamlines('./data/oldUnfold_graham/oldUnfold/native_streamlines.vtk')
# ulines=streamline.load_vtk_streamlines('./data/oldUnfold_graham/oldUnfold/from_unfold_streamlines.vtk')
# inuliens=streamline.load_vtk_streamlines('./data/oldUnfold_graham/oldUnfold/Unfolded/unfold_streamlines.vtk')
#
# nlens=list(length(nlines))
# ulens=list(length(ulines))
#
# nlens=np.asarray(nlens)
# ulens=np.asarray(ulens)
#
#
# plt.figure()
# plt.hist(nlens[(nlens>1) & (nlens<40) ],75,alpha=0.4,density=True)
# plt.hist(ulens[(ulens>1) & (ulens<40) ],75,alpha=0.4,density=True)
#
#
# ncurv=[]
# for line in nlines:
#     if len(line)>6:
#         curv = mean_curvature(line)
#         if curv<2:
#             ncurv.append(curv)
#
# ucurv=[]
# for line in ulines:
#     if len(line)>6:
#         curv = mean_curvature(line)
#         if curv < 2:
#             ucurv.append(curv)
#
# plt.figure()
# plt.hist(ncurv,50,alpha=0.4,density=True)
# plt.hist(ucurv,50,alpha=0.4,density=True)
#
# plot_streamlines(nlines)
# plot_streamlines(ulines)
#
#
#
# #combine the images into something nice
# # n1hippo=mpimg.imread('./data/plots/nativeHippoTracts.png')
# # n2hippo=mpimg.imread('./data/plots/nativeHippoTracts1.png')
# # un1hippo=mpimg.imread('./data/plots/unfldnativeHippoTracts.png')
# # un2hippo=mpimg.imread('./data/plots/unfldnativeHippoTracts1.png')
#
#
# nativehippo=Image.open('./data/plots/hippoNative.png')
# n1hippo=Image.open('./data/plots/nativeHippoTracts.png')
# n2hippo=Image.open('./data/plots/nativeHippoTracts1.png')
# un1hippo=Image.open('./data/plots/unfldnativeHippoTracts.png')
# un2hippo=Image.open('./data/plots/unfldnativeHippoTracts1.png')
#
#
#
# n2hippo=n2hippo.resize(n1hippo.size)
# un1hippo=un1hippo.resize(n1hippo.size)
# un2hippo=un2hippo.resize(n1hippo.size)
#
# fontsize=14
# fig,ax=plt.subplots(figsize=(8,11))
# ax.axis('off')
# padding=0.05
# width=(1-0.25)/4
# ax1=fig.add_axes([0.05,0.45,width,0.45])
# ax2=fig.add_axes([width+0.1,0.45,width,0.45])
# ax3=fig.add_axes([2*width+0.15,0.45,width,0.45])
# ax4=fig.add_axes([3*width+0.2,0.45,width,0.45])
#
# ax1.axis('off')
# ax2.axis('off')
# ax3.axis('off')
# ax4.axis('off')
#
# ax1.text(-n1hippo.size[1]*0.08,0,'a)',fontsize=16)
# ax2.text(-n1hippo.size[1]*0.08,0,'b)',fontsize=16)
# ax3.text(-n1hippo.size[1]*0.08,0,'c)',fontsize=16)
# ax4.text(-n1hippo.size[1]*0.08,0,'d)',fontsize=16)
#
# ax1.set_title('Cartesian')
# ax2.set_title('Cartesian')
# ax3.set_title('Harmonic')
# ax4.set_title('Harmonic')
#
# ax1.imshow(n1hippo)
# ax2.imshow(n2hippo)
# ax3.imshow(un1hippo)
# ax4.imshow(un2hippo)
#
# height=(0.45-0.05)/2
# ax5=fig.add_axes([0.05,0.05,0.6,height])
# ax6=fig.add_axes([0.05,0.1+height,0.6,height])
#
#
#
# u1hippo=Image.open('./data/plots/unfldHippo1.png')
# u2hippo=Image.open('./data/plots/unfldHippo2.png')
# u2hippo=u2hippo.resize(u1hippo.size)
#
# ax5.imshow(u1hippo)
# ax6.imshow(u2hippo)
#
# ax5.text(-u1hippo.size[1]*0.15,0,'g)',fontsize=fontsize)
# ax6.text(-u1hippo.size[1]*0.15,0,'e)',fontsize=fontsize)
#
#
#
# ydelta=30
# xdelta=30
# ax5.text(u1hippo.size[0]/2,u1hippo.size[1]+ydelta,'Proximal',horizontalAlignment='center',verticalAlignment='center',
#          fontsize=fontsize)
# ax5.text(u1hippo.size[0]/2,-ydelta,'Distal',horizontalAlignment='center',verticalAlignment='center',fontsize=fontsize)
# ax5.text(-xdelta,u1hippo.size[1]/2,'Posterior',horizontalAlignment='center',verticalAlignment='center',
#          fontsize=fontsize,rotation=90,
#          rotation_mode='anchor')
# ax5.text(u1hippo.size[0]+xdelta,u1hippo.size[1]/2,'Anterior',horizontalAlignment='center',verticalAlignment='center',
#          fontsize=fontsize,rotation=90,
#          rotation_mode='anchor')
#
# ax6.text(u1hippo.size[0]/2,u1hippo.size[1]+ydelta,'Proximal',horizontalAlignment='center',verticalAlignment='center',
#          fontsize=fontsize)
# ax6.text(u1hippo.size[0]/2,-ydelta,'Distal',horizontalAlignment='center',verticalAlignment='center',fontsize=fontsize)
# ax6.text(-xdelta,u1hippo.size[1]/2,'Anterior',horizontalAlignment='center',verticalAlignment='center',
#          fontsize=fontsize,rotation=90,
#          rotation_mode='anchor')
# ax6.text(u1hippo.size[0]+xdelta,u1hippo.size[1]/2,'Posterior',horizontalAlignment='center',verticalAlignment='center',
#          fontsize=fontsize,rotation=90,
#          rotation_mode='anchor')
#
# ax5.axis('off')
# ax6.axis('off')
#
#
#
# #ax7=fig.add_axes([0.721,0.15,0.25,0.25])
# ax7=fig.add_axes([0.735,0.3,0.21,0.18])
# ax7.hist(nlens[(nlens>1) & (nlens<40) ],50,alpha=0.4,density=True,label='Cartesian coords.')
# ax7.hist(ulens[(ulens>1) & (ulens<40) ],50,alpha=0.4,density=True,label='Harmonic coords.')
# ax7.legend()
# ax7.set_title('Length histogram')
# ax7.text(-0.15*ax7.get_xlim()[1],1.02*ax7.get_ylim()[1],'f)',fontsize=fontsize)
#
# ax8=fig.add_axes([0.735,0.05,0.21,0.18])
# ax8.hist(ncurv,50,alpha=0.4,density=True,label='Cartesian coords.')
# ax8.hist(ucurv,50,alpha=0.4,density=True,label='Harmonic coords.')
# ax8.legend()
# ax8.text(-0.22*ax8.get_xlim()[1],1.02*ax8.get_ylim()[1],'h)',fontsize=fontsize)
# ax8.set_title('Mean Curvature histogram')
#
#
#
# #make a complete hippocmapus iamge
# nativehippo=Image.open('./data/plots/hippoNative.png')
# n1hippo=Image.open('./data/plots/nativeHippoTracts.png')
# n2hippo=Image.open('./data/plots/nativeHippoTracts1.png')
# un1hippo=Image.open('./data/plots/unfldnativeHippoTracts.png')
# un2hippo=Image.open('./data/plots/unfldnativeHippoTracts1.png')
# udyads=Image.open('./data/plots/FA.png')
#
# padx=0.05
# pady=0.05
# fig,ax=plt.subplots(figsize=(8,11))
# ax.axis('off')
# ax_T2=fig.add_axes([0+padx,0.75+pady,1-padx,0.25-2*pady])
# ax_T2.imshow(nativehippo)
#
#
# width=(1-0.25)/4
# ax1=fig.add_axes([0.05,0.5+pady,width,0.25-pady])
# ax2=fig.add_axes([width+0.1,0.5+pady,width,0.25-pady])
# ax3=fig.add_axes([2*width+0.15,0.5+pady,width,0.25-pady])
# ax4=fig.add_axes([3*width+0.2,0.5+pady,width,0.25-pady])
#
# ax1.text(-n1hippo.size[1]*0.08,0,'a)',fontsize=16)
# ax2.text(-n1hippo.size[1]*0.08,0,'b)',fontsize=16)
# ax3.text(-n1hippo.size[1]*0.08,0,'c)',fontsize=16)
# ax4.text(-n1hippo.size[1]*0.08,0,'d)',fontsize=16)
#
# ax1.set_title('Cartesian')
# ax2.set_title('Cartesian')
# ax3.set_title('Harmonic')
# ax4.set_title('Harmonic')
#
# ax1.imshow(n1hippo)
# ax2.imshow(n2hippo)
# ax3.imshow(un1hippo)
# ax4.imshow(un2hippo)
#
# u1hippo=Image.open('./data/plots/unfldHippo1.png')
# u2hippo=Image.open('./data/plots/unfldHippo2.png')
# u2hippo=u2hippo.resize(u1hippo.size)
# udyads=udyads.resize(u1hippo.size)
#
#
# height=(0.5)/3
# pady=0.05
# ax5=fig.add_axes([0.05,0+pady,0.6,height-pady])
# ax6=fig.add_axes([0.05,height,0.6,height-pady])
# ax7=fig.add_axes([0.05,2*height,0.6,height-pady])
#
# ax7.imshow(udyads)
# ax6.imshow(u1hippo)
# ax5.imshow(u2hippo)
#
#
# ax7=fig.add_axes([0.735,0+pady,0.21,height-pady])
# ax7.hist(nlens[(nlens>1) & (nlens<40) ],50,alpha=0.4,density=True,label='Cartesian coords.')
# ax7.hist(ulens[(ulens>1) & (ulens<40) ],50,alpha=0.4,density=True,label='Harmonic coords.')
# ax7.legend()
# ax7.set_title('Length histogram')
# ax7.text(-0.15*ax7.get_xlim()[1],1.02*ax7.get_ylim()[1],'f)',fontsize=fontsize)
#
# ax8=fig.add_axes([0.735,height,0.21,height-pady])
# ax8.hist(ncurv,50,alpha=0.4,density=True,label='Cartesian coords.')
# ax8.hist(ucurv,50,alpha=0.4,density=True,label='Harmonic coords.')
# ax8.legend()
# ax8.text(-0.22*ax8.get_xlim()[1],1.02*ax8.get_ylim()[1],'h)',fontsize=fontsize)
# ax8.set_title('Mean Curvature histogram')
#
# dti_FA_n=nib.load('./data/oldUnfold/DiffusionCropped/Native_hiRes/Crop/L/cropped/dti_FA.nii.gz')
# dti_FA_u_full=nib.load('./data/oldUnfold/DiffusionCropped/Native_hiRes/Crop/L/cropped/Unfolded/dti_FA.nii.gz')
# dti_FA_u_rot=nib.load('./data/oldUnfold/DiffusionCropped/Native_hiRes/Crop/L/cropped/Unfolded/dti_rot_FA.nii.gz')
#
#
# fa_n=dti_FA_n.get_fdata()[dti_FA_n.get_fdata()>0]
# fa_u_full=dti_FA_u_full.get_fdata()[dti_FA_u_full.get_fdata()>0]
# fa_u_rot=dti_FA_u_rot.get_fdata()[dti_FA_u_rot.get_fdata()>0]
#
# histax=fig.add_axes([0.735,2*height,0.21,height-pady])
# histax.hist(fa_n,40,histtype='step',linewidth=3, density=True,alpha=0.99,label='Cartesian')
# histax.hist(fa_u_full,40,histtype='step',linewidth=3,density=True,alpha=0.99,label='Conformal')
# histax.hist(fa_u_rot,40,histtype='step',linewidth=3,density=True,alpha=0.99,label='Conformal, rotaion only')
# #ax.text(0.05,0.45,'d)',fontsize=22,transform=fig.transFigure)
# #ax.text(0.76,0.46,'e)',fontsize=22,transform=fig.transFigure)
# #histax.tick_params(axis='both',labelsize=20)
# histax.set_title('FA histogram',fontsize=fontsize)
# #histax.set_xlabel('FA',fontsize=22)
#
