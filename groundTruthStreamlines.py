from nibabel import Nifti1Image
from nibabel import load
import numpy as np
from dipy.io.image import load_nifti, save_nifti
from dipy.tracking import utils
from dipy.viz import window, actor, has_fury
from dipy.viz import colormap
from unfoldTracking import trueTracts
import matplotlib.pyplot as plt
from dipy.io.streamline import (load_vtk_streamlines,save_vtk_streamlines)
import unfoldTracking
from dipy.tracking.utils import target
from dipy.tracking.streamline import Streamlines
from dipy.tracking.distances import approx_polygon_track
from dipy.tracking.metrics import set_number_of_points
from dipy.tracking.streamline import select_random_set_of_streamlines
from unfoldTracking import connectedInds
import copy
from coordinates import toInds
import sys
from PIL import Image

def phiInv(U, V, W, w=None, scale=None):
    if scale is None: scale = 5
    if w is None: w = 1
    C = U + V * 1j
    Cout = np.exp(C)
    result = scale * (Cout - 1 + w * (1 / Cout - 1))
    return np.real(result), np.imag(result), W


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

def plot_mask(mask_nii,inds):
    data=mask_nii.get_fdata()
    data[:]=0
    for ind in inds:
        data[ind[0],ind[1],ind[2]]=1
    plt.imshow(data[:,:,0])

def plot_masks(mask_nii,inds1,inds2):
    data=mask_nii.get_fdata()
    data[:]=0
    for ind in inds1:
        data[ind[0],ind[1],ind[2]]=1
    plt.imshow(data[:,:,0],alpha=0.2)
    data[:]=0
    for ind in inds2:
        data[ind[0],ind[1],ind[2]]=1
    plt.imshow(data[:,:,0],alpha=0.2)


def plot_lines(lines,color='blue',alpha=0.03):
    plt.figure()
    plt.axis('equal')
    for line in lines:
        plt.plot(line[:,0],line[:,1], color=color,alpha=alpha)


class lines:

    def __init__(self,filepath):

        self.lines = load_vtk_streamlines(filepath)

        # these will be from a tangential seed
        self.seedlines = []

        self.lines_crsovr = []
        self.lines_crsovr_fail = []


        # sens spec
        self.sens = []
        self.spec = []



class linesFromSims:
    def __init__(self, npath, nlines, ulines):

        self.mask_nii = load(npath + 'nodif_brain_mask.nii.gz')
        self.radtang_nii = load(npath + 'radtang.nii.gz')
        self.halfdrt_nii = load(npath + 'halfdrt.nii.gz')
        self.angles_nii = load(npath + 'angles.nii.gz')
        self.updown_nii = load(npath + 'updown.nii.gz')

        self.mask = self.mask_nii.get_fdata()
        self.mask[:, :, 1:] = np.NaN
        self.radtang = self.radtang_nii.get_fdata()
        self.radtang[:, :, 1:] = np.NaN
        self.halfdrt = self.halfdrt_nii.get_fdata()
        self.halfdrt[:, :, 1:] = np.NaN
        self.angles = self.angles_nii.get_fdata()
        self.angles[:, :, 1:] = np.NaN
        self.updown = self.updown_nii.get_fdata()
        self.updown[:, :, 1:] = np.NaN

        self.nlines=nlines
        self.ulines=ulines


    def filter(self,lines):
        # lines that go though seed region
        testmask=copy.deepcopy(self.mask)

        #these are the raw lines from the seed
        testmask[:] = 0
        testmask[(self.radtang == 2) & (self.angles == 1)] = 1
        lines.seedlines = Streamlines(target(lines.lines, self.mask_nii.affine, testmask))

        # seeds
        testmask[:] = 0
        testmask[(self.radtang == 2) & (self.angles == 1)] = 1
        lines.seedlines = Streamlines(target(lines.lines, self.mask_nii.affine, testmask))


        # lines that cross over successfully
        testmask[:] = 0
        testmask[(self.radtang == 2) & (self.updown == 1)] = 1
        lines.lines_crsovr = Streamlines(target(lines.seedlines, self.mask_nii.affine, testmask))
        lines.lines_crsovr = Streamlines(target(lines.lines_crsovr, self.mask_nii.affine, (self.radtang == 1),
                                           include=False))

        #lines that fail to cross over
        lines.lines_crsovr_fail = Streamlines(target(lines.seedlines, self.mask_nii.affine, testmask, include=False))



    def lineCount(self,lines):
        sz=self.mask.shape
        tp_inds_linear=np.zeros(sz[0]*sz[1]*sz[2])
        for line in lines:
            inds=np.asarray(toInds(self.mask_nii,line).round().astype(int))
            lin_inds=np.ravel_multi_index([inds[:, 0], inds[:, 1], inds[:, 2]], (sz[0], sz[1], sz[2]))
            tp_inds_linear[lin_inds]=tp_inds_linear[lin_inds]+1
        return  tp_inds_linear


    # def symmetry(self):
    #     #makesymmetry masks
    #     right=
    def thresholdSenSpec(self,lines, threshold):

        #occupation by threshold
        inds=self.lineCount(lines.seedlines)
        inds=inds.reshape(self.mask.shape)
        test=inds

        tp = len(np.where(((test>=threshold) & (self.radtang==2))==True)[0])
        fn = len(np.where(((test < threshold) & (self.radtang == 2)) == True)[0])
        p = len(np.where(self.radtang==2)[0])

        fp = len(np.where(((test>=threshold) & (self.radtang==1))==True)[0])
        tn = len(np.where(((test < threshold) & (self.radtang == 1)) == True)[0])
        n = len(np.where(self.radtang==1)[0])


        if p==0:
            sens=np.NaN
        else:
            sens=tp/p

        if n ==0:
            spec=np.NaN
        else:
            spec=tn/n

        return sens, spec


scale=50
res=np.linspace(1.7,1.00,8)
drt=np.linspace(0.1,0.3,5)
ang_thr=np.linspace(20,90,8)
w=np.linspace(0,0.99,4)

# i_io=int(sys.argv[1]) #res
# j_io=int(sys.argv[2]) #drt
# k_io=int(sys.argv[3]) #ang_thr

# i_io=6
# j_io=1
# k_io=3
# l_io=-1
#
# base = "/home/uzair/PycharmProjects/Unfolding/data/diffusionSimulations_scale_50_res-"
# npath = base + str(int(res[i_io] * 1000)) + "um_drt-" + str(int(drt[j_io] * 100)) + "_w-" + \
#         str(int(round(w[l_io], 2) * 100)) + "_angthr-" + str(int(ang_thr[k_io])) + "/"
#
# nlines=lines(npath+'native_streamlines.vtk')
# ulines=lines(npath+'from_unfold_streamlines.vtk')
# simlines=linesFromSims(npath,nlines,ulines)
#
# simlines.filter(simlines.nlines)
# simlines.filter(simlines.ulines)

#plots for native space
# plot_lines(simlines.nlines.lines)
# plot_lines(simlines.nlines.seedlines)
# plot_lines(simlines.nlines.lines_crsovr)
# plot_lines(simlines.nlines.lines_crsovr_fail)

#plots for unfolded space
# plot_lines(simlines.ulines.lines)
# plot_lines(simlines.ulines.seedlines)
# plot_lines(simlines.ulines.lines_crsovr)
# plot_lines(simlines.ulines.lines_crsovr_fail)


#numbers in each plot
# crs_over_proportion = np.zeros([8,5,8,2])
# crs_over_fail_proportion = np.zeros([8,5,8,2])
#
# for i_io in range(0,len(res)):
#     print("i",i_io)
#     for j_io in range(0,len(drt)):
#         print("i j", i_io, j_io)
#         for k_io in range(0, len(ang_thr)):
#             for l_io in range(-1,0):#len(w)):
#                 print(l_io)
#                 base = "/home/uzair/PycharmProjects/Unfolding/data/diffusionSimulations_scale_50_res-"
#                 npath = base + str(int(res[i_io] * 1000)) + "um_drt-" + str(int(drt[j_io] * 100)) + "_w-" + \
#                         str(int(round(w[l_io], 2) * 100)) + "_angthr-" + str(int(ang_thr[k_io])) + "/"
#
#                 nlines = lines(npath + 'native_streamlines.vtk')
#                 ulines = lines(npath + 'from_unfold_streamlines.vtk')
#                 simlines = linesFromSims(npath, nlines, ulines)
#
#                 simlines.filter(simlines.nlines)
#                 simlines.filter(simlines.ulines)
#
#                 crs_over_proportion[i_io,j_io,k_io,0]=len(simlines.nlines.lines_crsovr)/len(simlines.nlines.seedlines)
#                 crs_over_proportion[i_io,j_io,k_io,1]=len(simlines.ulines.lines_crsovr)/len(simlines.ulines.seedlines)
#
#                 crs_over_fail_proportion[i_io,j_io,k_io,0]=len(simlines.nlines.lines_crsovr_fail)/len(
#                     simlines.nlines.seedlines)
#                 crs_over_fail_proportion[i_io,j_io,k_io,1]=len(simlines.ulines.lines_crsovr_fail)/len(
#                     simlines.ulines.seedlines)
#
#
#
# #crossover plot
# p=1
# fig, ax = plt.subplots(8,5)
# fig.set_figheight(18)
# fig.set_figwidth(10)
# fig.subplots_adjust(wspace=1)
# fig.subplots_adjust(hspace=1)
#
# for i in range(0,len(res)):
#     for j in range(0,len(drt)):
#         plotname=str((round(res[i],1))) + "mm drt" + str(round(drt[j] * 10,1))
#         ##plt.figure(p)
#         ax[i,j].set_title(plotname)
#         if( i==0 and j==0):
#             #ax[i,j].set_ylabel('Sensitivity')
#             #ax[i, j].set_ylabel('Specificity')
#             ax[i, j].set_xlabel('Angle Thres.')
#
#         ax[i,j].set_xlim([20,95])
#         ax[i, j].set_ylim([-0.1, 1.1])
#         y=crs_over_proportion[i,j,:,0]
#         x=ang_thr
#         ax[i,j].plot(x,y,color='blue')
#         y=crs_over_proportion[i,j,:,1]
#         ax[i,j].plot(x,y,color='orange')
#
# #crossover fail plot
# p=1
# fig, ax = plt.subplots(8,5)
# fig.set_figheight(18)
# fig.set_figwidth(10)
# fig.subplots_adjust(wspace=1)
# fig.subplots_adjust(hspace=1)
#
# for i in range(0,len(res)):
#     for j in range(0,len(drt)):
#         plotname=str((round(res[i],1))) + "mm drt" + str(round(drt[j] * 10,1))
#         ##plt.figure(p)
#         ax[i,j].set_title(plotname)
#         if( i==0 and j==0):
#             #ax[i,j].set_ylabel('Sensitivity')
#             #ax[i, j].set_ylabel('Specificity')
#             ax[i, j].set_xlabel('Angle Thres.')
#
#         ax[i,j].set_xlim([20,95])
#         ax[i, j].set_ylim([-0.1, 1.1])
#         y=crs_over_fail_proportion[i,j,:,0]
#         x=ang_thr
#         ax[i,j].plot(x,y,color='blue')
#         y=crs_over_fail_proportion[i,j,:,1]
#         ax[i,j].plot(x,y,color='orange')
#
# #where are the failures?
# nfig, nax = plt.subplots()
# ufig, uax = plt.subplots()
# nax.axis('equal')
# uax.axis('equal')
# for i_io in range(0,len(res)):
#     print("i",i_io)
#     for j_io in range(0,len(drt)):
#         print("i j", i_io, j_io)
#         for k_io in range(0, len(ang_thr)):
#             for l_io in range(-1,0):#len(w)):
#                 print(l_io)
#                 base = "/home/uzair/PycharmProjects/Unfolding/data/diffusionSimulations_scale_50_res-"
#                 npath = base + str(int(res[i_io] * 1000)) + "um_drt-" + str(int(drt[j_io] * 100)) + "_w-" + \
#                         str(int(round(w[l_io], 2) * 100)) + "_angthr-" + str(int(ang_thr[k_io])) + "/"
#
#                 nlines = lines(npath + 'native_streamlines.vtk')
#                 ulines = lines(npath + 'from_unfold_streamlines.vtk')
#                 simlines = linesFromSims(npath, nlines, ulines)
#
#                 simlines.filter(simlines.nlines)
#                 simlines.filter(simlines.ulines)
#
#                 for line in simlines.nlines.lines_crsovr_fail:
#                     nax.plot(line[:,0],line[:,1],alpha=0.075,color="blue")
#
#                 for line in simlines.ulines.lines_crsovr_fail:
#                     uax.plot(line[:,0],line[:,1],alpha=0.075,color="orange")
#
# #where are the winners?
# nfigw, naxw = plt.subplots()
# ufigw, uaxw = plt.subplots()
# naxw.axis('equal')
# uaxw.axis('equal')
# for i_io in range(0,len(res)):
#     print("i",i_io)
#     for j_io in range(0,len(drt)):
#         print("i j", i_io, j_io)
#         for k_io in range(0, len(ang_thr)):
#             for l_io in range(-1,0):#len(w)):
#                 print(l_io)
#                 base = "/home/uzair/PycharmProjects/Unfolding/data/diffusionSimulations_scale_50_res-"
#                 npath = base + str(int(res[i_io] * 1000)) + "um_drt-" + str(int(drt[j_io] * 100)) + "_w-" + \
#                         str(int(round(w[l_io], 2) * 100)) + "_angthr-" + str(int(ang_thr[k_io])) + "/"
#
#                 nlines = lines(npath + 'native_streamlines.vtk')
#                 ulines = lines(npath + 'from_unfold_streamlines.vtk')
#                 simlines = linesFromSims(npath, nlines, ulines)
#
#                 simlines.filter(simlines.nlines)
#                 simlines.filter(simlines.ulines)
#
#                 for line in simlines.nlines.lines_crsovr:
#                     naxw.plot(line[:,1],line[:,2],alpha=0.075,color="blue")
#
#                 for line in simlines.ulines.lines_crsovr:
#                     uaxw.plot(line[:,1],line[:,2],alpha=0.075,color="orange")

#threshold sen spec
# sens=np.zeros([8,5,8,2])
# spec=np.zeros([8,5,8,2])
# threshold=20
# for i_io in range(0,len(res)):
#     print("i",i_io)
#     for j_io in range(0,len(drt)):
#         print("i j", i_io, j_io)
#         for k_io in range(0, len(ang_thr)):
#             for l_io in range(-1,0):#len(w)):
#                 print(l_io)
#                 base = "/home/uzair/PycharmProjects/Unfolding/data/diffusionSimulations_scale_50_res-"
#                 npath = base + str(int(res[i_io] * 1000)) + "um_drt-" + str(int(drt[j_io] * 100)) + "_w-" + \
#                         str(int(round(w[l_io], 2) * 100)) + "_angthr-" + str(int(ang_thr[k_io])) + "/"
#
#                 nlines = lines(npath + 'native_streamlines.vtk')
#                 ulines = lines(npath + 'from_unfold_streamlines.vtk')
#                 simlines = linesFromSims(npath, nlines, ulines)
#
#                 simlines.filter(simlines.nlines)
#                 simlines.filter(simlines.ulines)
#
#                 sens[i_io,j_io,k_io,0],spec[i_io,j_io,k_io,0]= simlines.thresholdSenSpec(simlines.nlines,threshold)
#                 sens[i_io, j_io, k_io, 1], spec[i_io, j_io, k_io, 1] = simlines.thresholdSenSpec(
#                     simlines.ulines, threshold)


#plot of threshold sens
# p=1
# fig, ax = plt.subplots(8,5)
# fig.set_figheight(18)
# fig.set_figwidth(10)
# fig.subplots_adjust(wspace=1)
# fig.subplots_adjust(hspace=1)
#
# for i in range(0,len(res)):
#     for j in range(0,len(drt)):
#         plotname=str((round(res[i],1))) + "mm drt" + str(round(drt[j] * 10,1))
#         ##plt.figure(p)
#         ax[i,j].set_title(plotname)
#         if( i==0 and j==0):
#             #ax[i,j].set_ylabel('Sensitivity')
#             ax[i, j].set_ylabel('Specificity')
#             ax[i, j].set_xlabel('Angle Thres.')
#
#         ax[i,j].set_xlim([20,95])
#         ax[i, j].set_ylim([-0.1, 1.1])
#         y=spec[i,j,:,0]
#         x=ang_thr
#         ax[i,j].plot(x,y,color='blue')
#         y=spec[i,j,:,1]
#         ax[i,j].plot(x,y,color='orange')

# roc curve calculation for typical simulations angglethreshold
#threshold sen spec
# sens=np.zeros([8,5,8,2])
# spec=np.zeros([8,5,8,2])
# thres=np.linspace(0,100,10)
# i_io=3
# j_io=2
# l_io=-1
# spec_roc=np.zeros([10,8,2])
# sens_roc=np.zeros([10,8,2])
# for ttt in range(0,len(thres)):
#     for k_io in range(0, len(ang_thr)):
#         print(ttt,k_io)
#         base = "/home/uzair/PycharmProjects/Unfolding/data/diffusionSimulations_scale_50_res-"
#         npath = base + str(int(res[i_io] * 1000)) + "um_drt-" + str(int(drt[j_io] * 100)) + "_w-" + \
#                 str(int(round(w[l_io], 2) * 100)) + "_angthr-" + str(int(ang_thr[k_io])) + "/"
#
#         nlines = lines(npath + 'native_streamlines.vtk')
#         ulines = lines(npath + 'from_unfold_streamlines.vtk')
#         simlines = linesFromSims(npath, nlines, ulines)
#
#         simlines.filter(simlines.nlines)
#         simlines.filter(simlines.ulines)
#
#         sens_roc[ttt,k_io,0],spec_roc[ttt,k_io,0]= simlines.thresholdSenSpec(simlines.nlines,thres[ttt])
#         sens_roc[ttt, k_io, 1], spec_roc[ttt, k_io, 1] = simlines.thresholdSenSpec(simlines.ulines, thres[ttt])

#plot the roc curve
# fig, ax=plt.subplots(8,1)
# for k_io in range(0, len(ang_thr)):
#     print(ttt, k_io)
#     y = sens_roc[:, k_io, 0]
#     x = 1-spec_roc[:, k_io, 0]
#     ax[k_io].set_xlim([0, 1])
#     ax[k_io].set_ylim([0, 1])
#     ax[k_io].axis('equal')
#     ax[k_io].plot(x,y,color='blue')
#     y = sens_roc[:, k_io, 1]
#     x = 1-spec_roc[:, k_io, 1]
#     ax[k_io].plot(x,y,color='orange')

n_master_img=np.zeros([19,32])
u_master_img=np.zeros([19,32])
#seed denstity maps for each simulations
for i_io in range(0,len(res)):
    print("i",i_io)
    for j_io in range(0,len(drt)):
        print("i j", i_io, j_io)
        for k_io in range(0, len(ang_thr)):
            for l_io in range(-1,0):#len(w)):
                print(l_io)

                base = "/home/uzair/PycharmProjects/Unfolding/data/diffusionSimulations_scale_50_res-"
                npath = base + str(int(res[i_io] * 1000)) + "um_drt-" + str(int(drt[j_io] * 100)) + "_w-" + \
                        str(int(round(w[l_io], 2) * 100)) + "_angthr-" + str(int(ang_thr[k_io])) + "/"

                nlines = lines(npath + 'native_streamlines.vtk')
                ulines = lines(npath + 'from_unfold_streamlines.vtk')

                simlines = linesFromSims(npath, nlines, ulines)

                simlines.filter(simlines.nlines)
                simlines.filter(simlines.ulines)

                snap=simlines.lineCount(simlines.nlines.seedlines)
                snap=snap.reshape(simlines.mask.shape)
                nsnap= snap[:,:,0]

                #temp=Image.fromarray(snap)
                #n_master_img=n_master_img+np.asarray( temp.resize(n_master_img.shape[::-1]))

                #plt.imshow(snap)
                #plt.savefig(npath+'native_density_map.png')
                #plt.close()

                snap = simlines.lineCount(simlines.ulines.seedlines)
                snap = snap.reshape(simlines.mask.shape)
                usnap = snap[:, :, 0]

                residue=usnap-nsnap
                tangmask=simlines.radtang==2

                plt.title('unfld-native')
                plt.imshow(residue)
                plt.colorbar()
                plt.imshow(tangmask[:,:,0],alpha=0.5)
                plt.savefig(npath + 'residue_map.png')
                plt.close()

                #plt.imshow(snap)
                #plt.savefig(npath + 'unfold_density_map.png')
                #plt.close()
                #temp=Image.fromarray(snap)
                #u_master_img=u_master_img+np.asarray( temp.resize(u_master_img.shape[::-1]))


# #this is to make outline of rapunzel
# uu=0.3
# u=0
# vv=np.pi / 6
# v=-np.pi / 6
#
# constline=np.ones(10)
#
# uline=np.linspace(u,uu,10)
# vline=np.linspace(v,vv,10)
# u_const=constline*u
# uu_const=constline*uu
# v_const=constline*v
# vv_const=constline*vv
#
#
# figo, axo = plt.subplots()
# axo.axis('equal')
# x,y,z=phiInv(uline,v_const,0,0.99,scale=scale)
# nax.plot(x,y,color='black')
# x,y,z=phiInv(uline,vv_const,0,0.99,scale=scale)
# nax.plot(x,y,color='black')
# x,y,z=phiInv(u_const,vline,0,0.99,scale=scale)
# nax.plot(x,y,color='black')
# x,y,z=phiInv(uu_const,vline,0,0.99,scale=scale)
# nax.plot(x,y,color='black')
#
# x,y,z=phiInv(uline,v_const,0,0.99,scale=scale)
# uax.plot(x,y,color='black')
# x,y,z=phiInv(uline,vv_const,0,0.99,scale=scale)
# uax.plot(x,y,color='black')
# x,y,z=phiInv(u_const,vline,0,0.99,scale=scale)
# uax.plot(x,y,color='black')
# x,y,z=phiInv(uu_const,vline,0,0.99,scale=scale)
# uax.plot(x,y,color='black')



#plt.savefig(base+plotname)
#plt.savefig(base+'alldata.png')
#plt.close()


# def truePositiveRate(mask_nii,gtinds,inds):
#     #have to find overlap of gtinds(ground truth) and inds
#     if (len(inds) or len(gtinds))<4:
#         return np.NaN
#     gtindss = set(np.ravel_multi_index(np.array(gtinds).T,mask_nii.get_fdata().shape))
#     indss = set(np.ravel_multi_index(np.array(inds).T, mask_nii.get_fdata().shape))
#     overlap=gtindss.intersection(indss)
#     #if(len(overlap)/len(gtinds)==1):
#     #    return np.NaN
#     return(len(overlap)/len(gtinds))
#
# def falsePostiveRate(mask_nii,gtinds,inds):
#     if (len(inds) or len(gtinds)) <4:
#         return np.NaN
#     negatives=len(np.where(np.isnan(mask_nii.get_fdata())!=1)[0])
#     gtindss = set(np.ravel_multi_index(np.array(gtinds).T, mask_nii.get_fdata().shape))
#     negatives=negatives-len(gtindss)
#     indss = set(np.ravel_multi_index(np.array(inds).T, mask_nii.get_fdata().shape))
#     falsepositives=len(indss.difference(gtindss))
#     #if(falsepositives/negatives==1 or falsepositives/negatives==0 ):
#     #    return np.NaN
#     return(falsepositives/negatives)
#
# def trueNegativeRate(mask_nii,gtinds,inds):
#     return (1-falsePostiveRate(mask_nii,gtinds,inds))
#
#
# def sensSpec(mask_nii,tt,lines,inds,stopval):
#
#     tang_sens=[]
#     tang_spec=[]
#     rad_sens=[]
#     rad_spec=[]
#     tang_lines=[]
#     rad_lines=[]
#     residue=[]
#     for l in range(0,len(lines)):
#         #for p in range(0,int(0.2*len(lines[l]))):
#             #print(lines[l][p])
#         tang_trueinds,templine=tt.connectedInds(mask_nii,stopval,lines[l][0],const_coord='u')
#          #   if len(tang_trueinds) ==0:
#           #      break
#         # #for p in range(0, int(len(lines[l]))):
#         # rad_trueinds,templine=tt.connectedInds(mask_nii,1,lines[l][0],const_coord='v')
#         # if len(rad_trueinds) ==0:
#         #     break
#         # #if len(tang_trueinds)>3: # and
#         # #if (np.isnan(len(tang_trueinds))==0):
#         # tang_tpr = truePositiveRate(mask_nii, tang_trueinds, inds[l])
#         # tang_tnr = trueNegativeRate(mask_nii, tang_trueinds, inds[l])
#         #     #    tang_sens.append(tang_tpr)
#         #     #    tang_spec.append(tang_tnr)
#         #     #if (len(rad_trueinds) > 3 and
#         #     #if (np.isnan(len(rad_trueinds)) == 0):
#         # rad_tpr = truePositiveRate(mask_nii, rad_trueinds, inds[l])
#         # rad_tnr = trueNegativeRate(mask_nii, rad_trueinds, inds[l])
#         # #rad_sens.append(rad_tpr)
#         # #rad_spec.append(rad_tnr)
#         # if (tang_tpr >= rad_tpr):
#         #     tang_sens.append(tang_tpr)
#         #     tang_spec.append(tang_tnr)
#         #     tang_lines.append(lines[l])
#         # elif (tang_tpr < rad_tpr):
#         #     rad_sens.append(rad_tpr)
#         #     rad_spec.append(rad_tnr)
#         #     rad_lines.append(lines[l])
#         # else:
#         #     residue.append(lines[l])
#         tang_tpr = truePositiveRate(mask_nii, tang_trueinds, inds[l])
#         tang_tnr = trueNegativeRate(mask_nii, tang_trueinds, inds[l])
#         if np.isnan(tang_tpr)==0:
#             tang_sens.append(tang_tpr)
#         if np.isnan(tang_tnr) == 0:
#             tang_spec.append(tang_tnr)
#
#     tang_sens = np.asarray(tang_sens)
#     tang_spec = np.asarray(tang_spec)
#     rad_sens = np.asarray(rad_sens)
#     rad_spec = np.asarray(rad_spec)
#
#
#     #return tang_sens[tang_sens>0.3], tang_spec[tang_sens>0.3], rad_sens[rad_sens > 0.3], rad_spec[rad_sens > 0.3]
#     return tang_sens, tang_spec, rad_sens, rad_spec, tang_lines, rad_lines, residue
#
# def nonbuggytarget(lines,mask,affine,include=True):
#     lines_out=[]
#     for ii in range(0,len(lines)):
#         line=lines[ii]
#         try:
#             temp=Streamlines(target(line,affine,mask,include))
#         except ValueError:
#             print('Buggy ValueError at line number ', ii)
#             pass
#         else:
#             if(len(temp)>0):
#                 lines_out.append(temp)
#     return lines_out
#
# def lesspoints(lines):
#     #line_out= [approx_polygon_track(line,0.06) for line in lines]
#     #return select_random_set_of_streamlines(lines,2000)
#     return lines
# def loadmasks(npath):
#     mask_nii = load(npath+'nodif_brain_mask.nii.gz')
#     radtang_nii = load(npath + 'radtang.nii.gz')
#     halfdrt_nii = load(npath + 'halfdrt.nii.gz')
#     angles_nii = load(npath + 'angles.nii.gz')
#     updown_nii=load(npath + 'updown.nii.gz')
#     return mask_nii, radtang_nii, halfdrt_nii, angles_nii, updown_nii
#
# def nicelines(npath,lines_in):
#     mask_nii = load(npath + 'nodif_brain_mask.nii.gz')
#     radtang_nii = load(npath + 'radtang.nii.gz')
#     # halfdrt_nii = load(npath + 'halfdrt.nii.gz')
#     # angles_nii = load(npath + 'angles.nii.gz')
#     # updown_nii = load(npath + 'updown.nii.gz')
#     length=list(utils.length(lines_in))
#     length=np.asarray(length)
#     lines=[]
#     for ll in range(0,len(length)):
#         if length[ll]>10*mask_nii.affine[0,0]:
#             lines.append(lines_in[ll])
#
#     testmask = copy.deepcopy(mask_nii.get_fdata())
#
#     testmask[:] = 0
#     testmask[radtang_nii.get_fdata() == 2] = 1
#     lines_tang = nonbuggytarget(lines, testmask, mask_nii.affine)
#
#
# def coarse(npath,lines):
#     mask_nii = load(npath + 'nodif_brain_mask.nii.gz')
#     radtang_nii = load(npath + 'radtang.nii.gz')
#     halfdrt_nii = load(npath + 'halfdrt.nii.gz')
#     angles_nii = load(npath + 'angles.nii.gz')
#     updown_nii = load(npath + 'updown.nii.gz')
#
#
#     mask=mask_nii.get_fdata()
#     mask[:,:,1:]=np.NaN
#     radtang=radtang_nii.get_fdata()
#     radtang[:,:,1:]=np.NaN
#     halfdrt=halfdrt_nii.get_fdata()
#     halfdrt[:,:,1:]=np.NaN
#     angles=angles_nii.get_fdata()
#     angles[:,:,1:]=np.NaN
#     updown=updown_nii.get_fdata()
#     updown[:,:,1:]=np.NaN
#
#     testmask = copy.deepcopy(mask_nii.get_fdata())
#
#     #lines that go though seed region
#     testmask[:]=0
#     testmask[(radtang==2) & ((angles==1))] = 1
#     seedlines=Streamlines(target(lines,mask_nii.affine,testmask))
#
#     #lines from seeds that are true positives
#     testmask[:] = 0
#     testmask[(radtang==2) & (updown==1)] = 1
#     #testmask[(radtang== 2)] = 1
#     tplines = Streamlines(target(seedlines, mask_nii.affine, testmask ))
#     tplines = Streamlines(target(tplines, mask_nii.affine, (radtang == 1),include=False))
#
#     #get streamlines that start in seeds but do not pass through true region
#     # lines from seeds that are true positives
#     #testmask[:] = 0
#     #testmask[(radtang == 1) & (updown==1)] = 1
#     fplines=Streamlines(target(seedlines,mask_nii.affine, testmask,include=False))
#
#     #plt.figure()
#     seedinds=[]
#     for seedline in seedlines:
#         seedline=np.asarray(seedline)
#         cinds=connectedInds(seedline,mask_nii)
#         for cind in cinds:
#             ind=np.array(cind)
#             if testmask[ind[0],ind[1],ind[2]]==1:
#                 seedinds.append(cind) if cind not in seedinds else seedinds
#         #plt.plot(seedline[:,0],seedline[:,1],color='blue',alpha=0.2)
#
#     #plt.figure()
#     tpinds=[]
#     testmask[:]=0
#     testmask[(radtang==2) & (updown==1)] = 1
#     for tpline in tplines:
#         tpline=np.asarray(tpline)
#         cinds = connectedInds(tpline, mask_nii)
#         for cind in cinds:
#             ind=np.array(cind)
#             if testmask[ind[0],ind[1],ind[2]]==1:
#                tpinds.append(cind) if cind not in tpinds else tpinds
#         #plt.plot(tpline[:,0],tpline[:,1])
#
#     #plt.figure()
#     testmask[:] = 0
#     testmask[(radtang == 1) & (updown == 2)] = 1
#     fpinds=[]
#     for fpline in fplines:
#         fpline = np.asarray(fpline)
#         cinds = connectedInds(fpline, mask_nii)
#         for cind in cinds:
#             ind=np.array(cind)
#             if testmask[ind[0],ind[1],ind[2]]==1:
#                 fpinds.append(cind) if cind not in fpinds else fpinds
#         #plt.plot(fpline[:, 0], fpline[:, 1])
#
#     #gtp=len(np.where(radtang_nii.get_fdata()==2)[0])
#     gtp = len(np.where((radtang==2) & (updown==1))[0])
#     gtn=len(np.where((radtang == 1) & (updown==2))[0])
#     tp=len(tpinds)
#     fp=len(fpinds)
#
#     if tp/gtp >=1:
#         sens =1
#     else:
#         sens = tp/gtp
#     if gtn <=0:
#         spec=np.NaN
#     else:
#         spec=1 - fp/gtn
#
#     return sens, spec, seedlines,seedinds, tplines,tpinds, fplines,fpinds
#
#
#
# scale=75
# res=np.linspace(1.7,1.00,8)
# drt=np.linspace(0.1,0.3,5)
# ang_thr=np.linspace(20,90,8)
# w=np.linspace(0,0.99,4)
#
# # i_io=int(sys.argv[1]) #res
# # j_io=int(sys.argv[2]) #drt
# # k_io=int(sys.argv[3]) #ang_thr
#
# i_io=6
# j_io=1
# k_io=3
#
# allsens=np.zeros([8,5,8,4,2])
# allspec=np.zeros([8,5,8,4,2])
#
# for i_io in range(0,len(res)):
#     print("i",i_io)
#     for j_io in range(0,len(drt)):
#         print("i j", i_io, j_io)
#         for k_io in range(0, len(ang_thr)):
#             for l_io in range(-1,0):#len(w)):
#                 print(l_io)
#
#                 base = "/home/uzair/PycharmProjects/Unfolding/data/diffusionSimulations_scale_50_res-"
#                 npath = base + str(int(res[i_io] * 1000)) + "um_drt-" + str(int(drt[j_io] * 100)) + "_w-" + \
#                         str(int(round(w[l_io], 2) * 100)) + "_angthr-" + str(int(ang_thr[k_io])) + "/"
#
#
#                 print('Loading...')
#                 #load native and unfolded strealines
#                 nlines=lesspoints(load_vtk_streamlines(npath+'native_streamlines.vtk'))
#                 ulines=lesspoints(load_vtk_streamlines(npath+'from_unfold_streamlines.vtk'))
#
#                 nsens,nspec,nseedlines,nseedinds, ntplines,ntpinds, nfplines,nfpinds=coarse(npath,nlines)
#                 usens,uspec,useedlines,useedinds, utplines,utpinds, ufplines,ufpinds=coarse(npath,ulines)
#
#
#                 allsens[i_io, j_io, k_io, l_io, 0] = nsens
#                 allsens[i_io, j_io, k_io, l_io, 1] = usens
#
#                 allspec[i_io, j_io, k_io, l_io, 0] = nspec
#                 allspec[i_io, j_io, k_io, l_io, 1] = uspec
#
#
#                 #print('n mean', allmeans[k_io, l_io, 0])
#                 #print('u mean', allmeans[k_io, l_io, 1])
#
#     #
#     #
#     #
#     # np.save(npath+'ntang_sens.npy',ntang_sens)
#     # np.save(npath+'ntang_spec.npy',ntang_spec)
#     # np.save(npath+'nrad_sens.npy',nrad_sens)
#     # np.save(npath+'nrad_spec.npy',nrad_spec)
#     #
#     # np.save(npath+'utang_sens.npy',utang_sens)
#     # np.save(npath+'utang_spec.npy',utang_spec)
#     # np.save(npath+'urad_sens.npy',urad_sens)
#     # np.save(npath+'urad_spec.npy',urad_spec)
#     #
#     # np.save(npath+'ntang_lines.npy',ntang_lines)
#     # np.save(npath+'nrad_lines.npy',nrad_lines)
#     # np.save(npath+'nresidue.npy',nresidue)
#     # np.save(npath+'utang_lines.npy',utang_lines)
#     # np.save(npath+'urad_lines.npy',urad_lines)
#     # np.save(npath+'uresidue.npy',uresidue)
#     #
#     #
#     # plt.hist(ntang_sens,50,density=True,alpha=0.5,color='blue')
#     # plt.hist(utang_sens,50,density=True,alpha=0.5,color='orange')
#     # plt.savefig(npath+'tang_sens_hist.png')
#     #
#     # plt.figure(2)
#     # plt.hist(ntang_spec,50,density=True,alpha=0.5,color='blue')
#     # plt.hist(utang_spec,50,density=True,alpha=0.5,color='orange')
#     # plt.savefig(npath+'tang_spec_hist.png')
#     #
#     # plt.figure(3)
#     # plt.hist(nrad_sens,50,density=True,alpha=0.5,color='blue')
#     # plt.hist(urad_sens,50,density=True,alpha=0.5,color='orange')
#     # plt.savefig(npath+'rad_sens_hist.png')
#     #
#     # plt.figure(4)
#     # plt.hist(nrad_spec,50,density=True,alpha=0.5,color='blue')
#     # plt.hist(urad_spec,50,density=True,alpha=0.5,color='orange')
#     # plt.savefig(npath+'rad_spec_hist.png')
#     #
#     # plt.close()
#     # print(npath)
#
# import matplotlib
#
# font = {'family' : 'sans-serif',
#         'weight':1,
#         'size'   : 13}
#
# matplotlib.rc('font', **font)
#
# p=1
# fig, ax = plt.subplots(8,5)
# fig.set_figheight(18)
# fig.set_figwidth(10)
# fig.subplots_adjust(wspace=1)
# fig.subplots_adjust(hspace=1)
# for i in range(0,len(res)):
#     for j in range(0,len(drt)):
# #base = "/home/u2hussai/scratch/"
#         plotname=str((round(res[i],1))) + "mm drt" + str(round(drt[j] * 10,1))
# ##plt.figure(p)
#         ax[i,j].set_title(plotname)
#         if( i==0 and j==0):
#             #ax[i,j].set_ylabel('Sensitivity')
#             ax[i, j].set_ylabel('Specificity')
#             ax[i, j].set_xlabel('Angle Thres.')
#
#         ax[i,j].set_xlim([20,95])
#         #ax[i, j].set_ylim([0.2, 1.0])
#         y=allsens[i,j,:,-1,0]
#         x=ang_thr
#         ax[i,j].plot(x,y,color='blue')
#         y=allsens[i,j,:,-1,1]
#         ax[i,j].plot(x,y,color='orange')
#
# #plt.savefig(base+plotname)
# #plt.savefig(base+'alldata.png')
# #plt.close()
#
# # seeds=[]
# # testlines=[]
# # for line in ulines:
# #     seeds.append(line[0])
# #     temp, templine=tt.connectedInds(radtang_nii,2,line[0],const_coord='u')
# #     testlines.append(templine)
# #
# # for ss in range(0,len(seeds)):
# #     seed=seeds[ss]
# #     plt.scatter(seed[0],seed[1])
# #     if(len(testlines[ss])>0):
# #         thisline=np.asarray(testlines[ss])
# #         plt.plot(thisline[:,0],thisline[:,1])
# # plt.axis('equal')