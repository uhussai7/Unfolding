##---- Here we want to compute threshold sensitivity and specificity for *all* simulations------#
#we will use both the ground truth mask and the voxel mask to do so.

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
from unfoldTracking import connectedInds
import copy
from coordinates import toInds
import sys


def phiInv(U, V, W, w=None, scale=None):
    if scale is None: scale = 5
    if w is None: w = 1
    C = U + V * 1j
    Cout = np.exp(C)
    result = scale * (Cout - 1 + w * (1 / Cout - 1))
    return np.real(result), np.imag(result), W


def phi(X, Y, Z, w=None, scale=None, beta=None):
    if scale is None: scale = 5
    if w is None: w = 1
    C = X + Y * 1j
    A = C / scale + w + 1
    Cout = np.log(0.5 * (A + np.sqrt(A * A - 4 * w)))
    return np.real(Cout), np.imag(Cout), Z

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

        self.mask[np.isnan(self.mask)] = 0
        self.radtang[np.isnan(self.radtang)] = 0
        self.halfdrt[np.isnan(self.halfdrt)] =  0
        self.angles[np.isnan(self.angles)] = 0
        self.updown[np.isnan(self.updown)] = 0

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
        fn = len(np.where(((test < threshold) & (self.radtang != 2)) == True)[0])
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

Nthres=20
scale=50
res=np.linspace(1.7,1.00,8)
drt=np.linspace(0.1,0.3,5)
ang_thr=np.linspace(20,90,8)
w=np.linspace(0,0.99,4)
thres=np.linspace(0,150,Nthres)

uu=0.3
u=0
vv=np.pi / 6
v=-np.pi / 6

i_io=int(sys.argv[1])
j_io=int(sys.argv[2])
l_io=int(sys.argv[3])

spec_roc=np.zeros([Nthres,8,2])
sens_roc=np.zeros([Nthres,8,2])
for ttt in range(0,len(thres)):
    for k_io in range(0, len(ang_thr)):
        print(ttt,k_io)
        base = "/home/u2hussai/scratch/unfoldingSimulations/diffusionSimulations_scale_50_res-"
        npath = base + str(int(res[i_io] * 1000)) + "um_drt-" + str(int(drt[j_io] * 100)) + "_w-" + \
                str(int(round(w[l_io], 2) * 100)) + "_angthr-" + str(int(ang_thr[k_io])) + "/"

        nlines = lines(npath + 'native_streamlines.vtk')
        ulines = lines(npath + 'from_unfold_streamlines.vtk')
        simlines = linesFromSims(npath, nlines, ulines)

        simlines.filter(simlines.nlines)
        simlines.filter(simlines.ulines)

        sens_roc[ttt,k_io,0],spec_roc[ttt,k_io,0]= simlines.thresholdSenSpec(simlines.nlines, thres[ttt])
        sens_roc[ttt, k_io, 1], spec_roc[ttt, k_io, 1] = simlines.thresholdSenSpec(simlines.ulines,thres[ttt])

        np.save("/home/u2hussai/scratch/unfoldingSimulations/rocs/"+"sens_"+str(int(res[i_io] * 1000)) + "um_drt-" + str(int(drt[j_io] * 100)) + "_w-" + \
                str(int(round(w[l_io], 2) * 100)) + "_angthr-" + str(int(ang_thr[k_io]))+".npy",sens_roc)

        np.save("/home/u2hussai/scratch/unfoldingSimulations/rocs/"+"spec_"+str(int(res[i_io] * 1000)) + "um_drt-" + str(int(drt[j_io] * 100)) + "_w-" + \
                str(int(round(w[l_io], 2) * 100)) + "_angthr-" + str(int(ang_thr[k_io]))+".npy",spec_roc)