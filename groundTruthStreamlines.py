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
from dipy.tracking.streamline import select_random_set_of_streamlines

import copy
import sys


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


def truePositiveRate(mask_nii,gtinds,inds):
    #have to find overlap of gtinds(ground truth) and inds
    if (len(gtinds))<1:
        return np.NaN
    gtindss = set(np.ravel_multi_index(np.array(gtinds).T,mask_nii.get_fdata().shape))
    indss = set(np.ravel_multi_index(np.array(inds).T, mask_nii.get_fdata().shape))
    overlap=gtindss.intersection(indss)
    #if(len(overlap)/len(gtinds)==1):
    #    return np.NaN
    return(len(overlap)/len(gtinds))

def falsePostiveRate(mask_nii,gtinds,inds):
    if (len(gtinds)) <1:
        return np.NaN
    negatives=len(np.where(np.isnan(mask_nii.get_fdata())!=1)[0])
    gtindss = set(np.ravel_multi_index(np.array(gtinds).T, mask_nii.get_fdata().shape))
    negatives=negatives-len(gtindss)
    indss = set(np.ravel_multi_index(np.array(inds).T, mask_nii.get_fdata().shape))
    falsepositives=len(indss.difference(gtindss))
    #if(falsepositives/negatives==1 or falsepositives/negatives==0 ):
    #    return np.NaN
    return(falsepositives/negatives)

def trueNegativeRate(mask_nii,gtinds,inds):
    return (1-falsePostiveRate(mask_nii,gtinds,inds))


def sensSpec(mask_nii,tt,lines,inds,stopval):

    tang_sens=[]
    tang_spec=[]
    rad_sens=[]
    rad_spec=[]
    tang_lines=[]
    rad_lines=[]
    residue=[]
    for l in range(0,len(lines)):
        #for p in range(0,int(len(lines[l]))):
            #print(lines[l][p])
        tang_trueinds,templine=tt.connectedInds(mask_nii,stopval,lines[l][0],const_coord='u')
        if len(tang_trueinds) ==0:
            break
        # #for p in range(0, int(len(lines[l]))):
        # rad_trueinds,templine=tt.connectedInds(mask_nii,1,lines[l][0],const_coord='v')
        # if len(rad_trueinds) ==0:
        #     break
        # #if len(tang_trueinds)>3: # and
        # #if (np.isnan(len(tang_trueinds))==0):
        # tang_tpr = truePositiveRate(mask_nii, tang_trueinds, inds[l])
        # tang_tnr = trueNegativeRate(mask_nii, tang_trueinds, inds[l])
        #     #    tang_sens.append(tang_tpr)
        #     #    tang_spec.append(tang_tnr)
        #     #if (len(rad_trueinds) > 3 and
        #     #if (np.isnan(len(rad_trueinds)) == 0):
        # rad_tpr = truePositiveRate(mask_nii, rad_trueinds, inds[l])
        # rad_tnr = trueNegativeRate(mask_nii, rad_trueinds, inds[l])
        # #rad_sens.append(rad_tpr)
        # #rad_spec.append(rad_tnr)
        # if (tang_tpr >= rad_tpr):
        #     tang_sens.append(tang_tpr)
        #     tang_spec.append(tang_tnr)
        #     tang_lines.append(lines[l])
        # elif (tang_tpr < rad_tpr):
        #     rad_sens.append(rad_tpr)
        #     rad_spec.append(rad_tnr)
        #     rad_lines.append(lines[l])
        # else:
        #     residue.append(lines[l])
        tang_tpr = truePositiveRate(mask_nii, tang_trueinds, inds[l])
        tang_tnr = trueNegativeRate(mask_nii, tang_trueinds, inds[l])
        if np.isnan(tang_tpr)==0:
            tang_sens.append(tang_tpr)
        if np.isnan(tang_tnr) == 0:
            tang_spec.append(tang_tnr)

    tang_sens = np.asarray(tang_sens)
    tang_spec = np.asarray(tang_spec)
    rad_sens = np.asarray(rad_sens)
    rad_spec = np.asarray(rad_spec)


    #return tang_sens[tang_sens>0.3], tang_spec[tang_sens>0.3], rad_sens[rad_sens > 0.3], rad_spec[rad_sens > 0.3]
    return tang_sens, tang_spec, rad_sens, rad_spec, tang_lines, rad_lines, residue

def nonbuggytarget(lines,mask,affine):
    lines_out=[]
    for ii in range(0,len(lines)):
        line=lines[ii]
        try:
            temp=Streamlines(target(line,affine,mask))
        except ValueError:
            pass
            print('Buggy ValueError at line number ', ii)
        else:
            if(len(temp)>0):
                lines_out.append(temp)
    return lines_out

def lesspoints(lines):
    #line_out= [approx_polygon_track(line,0.06) for line in lines]
    if len(lines) > 1000:
        return select_random_set_of_streamlines(lines,1000)
    else:
        return lines 

def loadmasks(npath):
    mask_nii = load(npath+'nodif_brain_mask.nii.gz')
    radtang_nii = load(npath + 'radtang.nii.gz')
    halfdrt_nii = load(npath + 'halfdrt.nii.gz')
    angles_nii = load(npath + 'angles.nii.gz')
    updown_nii=load(npath + 'updown.nii.gz')
    return mask_nii, radtang_nii, halfdrt_nii, angles_nii, updown_nii

def nicelines(npath,lines_in):
    mask_nii = load(npath + 'nodif_brain_mask.nii.gz')
    radtang_nii = load(npath + 'radtang.nii.gz')
    # halfdrt_nii = load(npath + 'halfdrt.nii.gz')
    # angles_nii = load(npath + 'angles.nii.gz')
    # updown_nii = load(npath + 'updown.nii.gz')
    length=list(utils.length(lines_in))
    length=np.asarray(length)
    lines=[]
    for ll in range(0,len(length)):
        if length[ll]>5*mask_nii.affine[0,0]:
            lines.append(lines_in[ll])

    testmask=copy.deepcopy(mask_nii.get_fdata())

    testmask[:]=0
    testmask[radtang_nii.get_fdata() == 2]=1
    lines_tang = nonbuggytarget(lines, testmask,mask_nii.affine)
    return lines_tang


scale=50
res=np.linspace(1.7,1.00,8)
drt=np.linspace(0.1,0.3,5)
ang_thr=np.linspace(20,90,8)
w=np.linspace(0,0.99,4)

i_io=0#int(sys.argv[1]) #res
j_io=0#int(sys.argv[2]) #drt
k_io=0#int(sys.argv[3]) #ang_thr



for l_io in range(0,1):#len(w)):
    print(l_io)

    #setup functions for ground truth
    def change_w_scale(w,scale):
        def wrap(func):
            def inner(X,Y,Z,w=w,scale=scale):
                return func(X,Y,Z,w=w,scale=scale)
            return inner
        return wrap


    @change_w_scale(w=w[l_io], scale=scale)
    def phi(X, Y, Z, w=None, scale=None):
        if scale is None: scale = 5
        if w is None: w = 1
        C = X + Y * 1j
        A = C / scale + w + 1
        Cout = np.log(0.5 * (A + np.sqrt(A * A - 4 * w)))
        return np.real(Cout), np.imag(Cout), Z

    @change_w_scale(w=w[l_io], scale=scale)
    def phiInv(U,V,W,w=None,scale=None):
        if scale is None: scale = 5
        if w is None: w=1
        C = U + V * 1j
        Cout = np.exp(C)
        result = scale*(Cout-1 + w*(1/Cout-1))
        return np.real(result), np.imag(result), W


    #base = "/home/u2hussai/Unfolding/data/diffusionSimulations_res-"
    #npath=base + str(int(res[i] * 1000)) + "um_drt-" + str(int(drt[j] * 100)) + "_angthr-" + str(int(ang_thr[
    # k])) + "/"

    base = "/home/u2hussai/scratch/unfoldingSimulations/diffusionSimulations_scale_50_res-"
    npath = base + str(int(res[i_io] * 1000)) + "um_drt-" + str(int(drt[j_io] * 100)) + "_w-" + \
            str(int(round(w[l_io], 2) * 100)) + "_angthr-" + str(int(ang_thr[k_io])) + "/"

    mask_nii, radtang_nii, halfdrt_nii, angles_nii, updown_nii=loadmasks(npath)
    U_nii = load(npath + 'U.nii.gz')
    V_nii = load(npath + 'V.nii.gz')

    print('Loading from' + npath)
    #load native and unfolded strealines
    nlines=lesspoints(load_vtk_streamlines(npath+'native_streamlines.vtk'))
    ulines=lesspoints(load_vtk_streamlines(npath+'from_unfold_streamlines.vtk'))

    print('Filtering...')
    nlines_tang=nicelines(npath,nlines)
    ulines_tang=nicelines(npath,ulines)
    print(len(nlines))

    print('Getting native voxels')
    # #get the voxel inds
    Nrand=200
    if len(nlines_tang)>Nrand:
        nlines= select_random_set_of_streamlines(nlines_tang,Nrand)
        ulines=select_random_set_of_streamlines(ulines_tang,Nrand)
    ninds=[]
    nginds=[]
    for line in nlines:
        ninds.append(unfoldTracking.connectedInds(line,mask_nii))

    print('Getting unfold voxels')
    uinds=[]
    uginds=[]
    for line in ulines:
        uinds.append(unfoldTracking.connectedInds(line,mask_nii))

    print('Getting true tact instance')
    # #truetracts class instance
    tt=trueTracts(radtang_nii,U_nii,V_nii,2,phi,phiInv)
    #
    # #get sensitivity and specificity
    print('Native sensitivity')
    ntang_sens, ntang_spec, nrad_sens, nrad_spec, ntang_lines, nrad_lines, nresidue= sensSpec(radtang_nii,tt,nlines,
                                                                                                ninds,2)
    print('Unfold sensitivity')
    utang_sens, utang_spec, urad_sens, urad_spec, utang_lines, urad_lines, uresidue= sensSpec(radtang_nii,tt,ulines,
                                                                                                uinds,2)


    np.save(npath+'ntang_sens.npy',ntang_sens)
    np.save(npath+'ntang_spec.npy',ntang_spec)
    np.save(npath+'nrad_sens.npy',nrad_sens)
    np.save(npath+'nrad_spec.npy',nrad_spec)

    np.save(npath+'utang_sens.npy',utang_sens)
    np.save(npath+'utang_spec.npy',utang_spec)
    np.save(npath+'urad_sens.npy',urad_sens)
    np.save(npath+'urad_spec.npy',urad_spec)

    np.save(npath+'ntang_lines.npy',ntang_lines)
    np.save(npath+'nrad_lines.npy',nrad_lines)
    np.save(npath+'nresidue.npy',nresidue)
    np.save(npath+'utang_lines.npy',utang_lines)
    np.save(npath+'urad_lines.npy',urad_lines)
    np.save(npath+'uresidue.npy',uresidue)


    plt.hist(ntang_sens,50,density=True,alpha=0.5,color='blue')
    plt.hist(utang_sens,50,density=True,alpha=0.5,color='orange')
    plt.savefig(npath+'tang_sens_hist_alt.png')
    plt.close()

    plt.figure(2)
    plt.hist(ntang_spec,50,density=True,alpha=0.5,color='blue')
    plt.hist(utang_spec,50,density=True,alpha=0.5,color='orange')
    plt.savefig(npath+'tang_spec_hist_alt.png')
    plt.close()

    plt.figure(3)
    plt.hist(nrad_sens,50,density=True,alpha=0.5,color='blue')
    plt.hist(urad_sens,50,density=True,alpha=0.5,color='orange')
    plt.savefig(npath+'rad_sens_hist_alt.png')
    plt.close()

    plt.figure(4)
    plt.hist(nrad_spec,50,density=True,alpha=0.5,color='blue')
    plt.hist(urad_spec,50,density=True,alpha=0.5,color='orange')
    plt.savefig(npath+'rad_spec_hist_alt.png')
    plt.close()

    print(npath)








