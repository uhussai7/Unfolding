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
#from dipy.tracking.metrics import set_number_of_points
from dipy.tracking.streamline import select_random_set_of_streamlines
from unfoldTracking import connectedInds
import copy
from coordinates import toInds
import sys
from PIL import Image
from coordinates import toWorld

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

    def filter(self, lines):
        # lines that go though seed region
        testmask = copy.deepcopy(self.mask)

        # these are the raw lines from the seed
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

        # lines that fail to cross over
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


scale = 50
res = np.linspace(1.7, 1.00, 16)
drt = np.linspace(0.1, 0.3, 16)
ang_thr = np.linspace(20, 90, 16)
w = np.linspace(0, 0.99, 4)


#low class
i_io=int(sys.argv[1])
j_io=int(sys.argv[2])
#k_io=int(sys.argv[3])
l_io=-1

for k_io in range(0,15):

    base0= "/home/u2hussai/scratch/unfoldingSimulations/"
    base = base0+"diffusionSimulations-Alignment-LambdaStep-k-60_Fine_scale_50_res-"
    npath = base + str(int(res[i_io] * 1000)) + "um_drt-" + str(int(drt[j_io] * 100)) + "_w-" + \
            str(int(round(w[l_io], 2) * 100)) + "_angthr-" + str(int(ang_thr[k_io])) + "_beta-0/"

    print('loading '+ npath)
    nlines = lines(npath + 'native_streamlines.vtk')
    ulines = lines(npath + 'from_unfold_streamlines.vtk')

    simlines = linesFromSims(npath, nlines, ulines)

    simlines.filter(simlines.nlines)
    simlines.filter(simlines.ulines)

    #pplot the mask in world coordinates
    max_x=[simlines.mask.shape[0],simlines.mask.shape[1],0]
    max_y=[0,0,0]
    delta=simlines.mask_nii.affine[0,0]/2
    world=toWorld(simlines.mask_nii, [max_x,max_y])

    fig1,ax1=plt.subplots()
    ax1.imshow((simlines.radtang==2)[:,:,0].T,extent=[world[1][0]-delta,world[0][0]+delta,world[1][1]-delta,
                                                    world[0][1]+delta],origin='lower')
    ax1.axis('equal')
    for line in simlines.nlines.seedlines:
        ax1.plot(line[:,0]+delta,line[:,1]+delta,color='blue')
    fig1.savefig('/home/u2hussai/scratch/unfoldingSimulationsSnapsFine/'+str(int(res[i_io] * 1000)) + "um_drt-" + str(int(drt[j_io] * 100)) + "_w-" + \
            str(int(round(w[l_io], 2) * 100)) + "_angthr-" + str(int(ang_thr[k_io])) + "_beta-0_native.png")
    plt.close()
    
    fig2,ax2=plt.subplots()
    ax2.imshow((simlines.radtang==2)[:,:,0].T,extent=[world[1][0]-delta,world[0][0]+delta,world[1][1]-delta,
                                                    world[0][1]+delta],origin='lower')
    ax2.axis('equal')
    for line in simlines.ulines.seedlines:
        ax2.plot(line[:,0]+delta,line[:,1]+delta,color='blue')
    fig2.savefig('/home/u2hussai/scratch/unfoldingSimulationsSnapsFine/'+str(int(res[i_io] * 1000)) + "um_drt-" + str(int(drt[j_io] * 100)) + "_w-" + \
            str(int(round(w[l_io], 2) * 100)) + "_angthr-" + str(int(ang_thr[k_io])) + "_beta-0_unfold.png")
    plt.close()