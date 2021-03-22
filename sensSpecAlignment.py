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
from math import sqrt
from math import acos
from numpy import dot
import cmath


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
    def __init__(self, npath, nlines, ulines, phi, phiInv,drt,maskpath):

        self.omask_nii = load(npath + 'nodif_brain_mask.nii.gz') 
        self.mask_nii = load(maskpath + 'nodif_brain_mask.nii.gz')
        self.radtang_nii = load(maskpath + 'radtang.nii.gz')
        self.halfdrt_nii = load(maskpath + 'halfdrt.nii.gz')
        self.angles_nii = load(maskpath + 'angles.nii.gz')
        self.updown_nii = load(maskpath + 'updown.nii.gz')

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

        self.phi=phi
        self.phiInv=phiInv
        self.dphi=dphi
        self.drt=drt

        self.ntanglines, self.nradlines=self.target_parametric(nlines.lines)
        self.utanglines, self.uradlines=self.target_parametric(ulines.lines)

        

    def target_parametric(self,liness):
        tanglines=[]
        radlines=[]
        uu = 1/1.75
        u = 0.02
        #x_h= 0.5*
        x_h= 0.5*(self.phiInv(uu,0,0)[0]+self.phiInv(u,0,0)[0])
        drt= self.phi(x_h,0,0)[0]
        for line in liness:
            rad_int=0
            for p in line:
                u,v,w=self.phi(p[0],p[1],p[2])
                if u > drt:
                    rad_int=1
                    break
            if rad_int==1:
                radlines.append(line)
            else:
                tanglines.append(line)

        return tanglines, radlines
        
                    
                    



    def filter(self,lines,dphi):
        # lines that go though seed region
        #testmask=copy.deepcopy(self.mask)

        #these are the raw lines from the seed
        #testmask[:] = 0
        #testmask[(self.radtang == 2) & (self.angles == 1)] = 1
        lines.seedlines = lines.lines#Streamlines(target(lines.lines, self.mask_nii.affine, testmask))
        
        #lines.seedlines should only be the tangential part. Have to measure angle with tangential corrdinate curve.
        # seedlines_rad_tang=Streamlines(target(lines.lines, self.mask_nii.affine, testmask))
        lines.seedlines=[]
        #for line in seedlines_rad_tang:
        for line in lines.lines:
            if(len(line)>2):
                p1=np.asarray(line[0][:])
                p2=np.asarray(line[1][:])
                v=p2-p1
                v1_coord,v2_coord,v3_coord=self.dphi(line[0][0],line[0][1],line[0][2])
                lengths=sqrt(dot(v,v))*sqrt(dot(v1_coord,v1_coord))
                angle=acos(dot(v,v1_coord)/lengths)
                if((angle <= np.pi/4) & (angle >= -np.pi/4)):
                    lines.seedlines.append(line)

        # seeds
        #testmask[:] = 0
        #testmask[(self.radtang == 2) & (self.angles == 1)] = 1
        #lines.seedlines = Streamlines(target(lines.lines, self.mask_nii.affine, testmask))


        # lines that cross over successfully
        #testmask[:] = 0
        #testmask[(self.radtang == 2) & (self.updown == 1)] = 1
        #lines.lines_crsovr = Streamlines(target(lines.seedlines, self.mask_nii.affine, testmask))
        #lines.lines_crsovr = Streamlines(target(lines.lines_crsovr, self.mask_nii.affine, (self.radtang == 1),
        #                                   include=False))

        #lines that fail to cross over
        #lines.lines_crsovr_fail = Streamlines(target(lines.seedlines, self.mask_nii.affine, testmask, include=False))\

        



    def lineCount(self,lines):
        sz=self.mask.shape
        tp_inds_linear=np.zeros(sz[0]*sz[1]*sz[2])
        for line in lines:
            inds=np.asarray(toInds(self.mask_nii,line).round().astype(int))
            oinds=np.asarray(toInds(self.omask_nii,line).round().astype(int))
                
            #print(inds,sz)
            #print(oinds[0:10,:])
            #lin_inds=np.ravel_multi_index([inds[:, 0], inds[:, 1], inds[:, 2]], (sz[0], sz[1], sz[2]))
            lin_inds=np.ravel_multi_index([inds[:, 0], inds[:, 1], oinds[:, 2]], (sz[0], sz[1], sz[2]),mode='clip')
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
        #fn = len(np.where(((test < threshold) & (self.radtang != 2) & (self.mask ==0) ) == True)[0])
        p = len(np.where(self.radtang==2)[0])

        #fp = len(np.where(((test>=threshold) & (self.radtang==1) & (self.mask ==0) )==True)[0])
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
        
        tangCount= sum(inds[self.radtang==2])
        radCount= sum(inds[self.radtang==1])

        return sens, spec,tangCount, radCount




Nthres=1
scale=1
res=np.linspace(0.03,0.13,16)
drt=[0.5*(1/1.75 - 0.07)]#np.linspace(0.1,0.3,16)
ang_thr=np.linspace(20,90,16)
w=np.linspace(1.0,1.99,16)
beta=np.linspace(0,np.pi/4,5)
#thres=np.linspace(0,10,Nthres)
thres=[1]

def L1L2L3_drt_w_scale(drt,w,scale,beta):
    def wrap(func):
        def inner(X,Y,Z,drt=drt,w=w,scale=scale,beta=beta):
            return func(X,Y,Z,drt=drt,w=w,scale=scale,beta=beta)
        return inner
    return wrap

def change_w_scale(w,scale,beta):
    def wrap(func):
        def inner(X,Y,Z,w=w,scale=scale,beta=beta):
            return func(X,Y,Z,w=w,scale=scale,beta=beta)
        return inner
    return wrap


uu = 1/1.75
u = 0.02
vv = np.pi / 4
v = -np.pi / 4

i_io=int(sys.argv[1]) 
#j_io=int(sys.argv[2]) #w
#k_io=int(sys.argv[3])

#spec_roc=np.zeros([Nthres,len(beta),2])
#sens_roc=np.zeros([Nthres,len(beta),2])

spec_roc=np.zeros([Nthres,2])
sens_roc=np.zeros([Nthres,2])


tangCount=np.zeros([2])
radCount=np.zeros([2])
tangCountPara=np.zeros([2])
radCountPara=np.zeros([2])

#for i_io in range(0,len(res)):
for j_io in range(0,len(w)):
    for k_io in range(0,len(ang_thr)):
        for l_io in range(0,1):#len(w)):
            for b in range(0,1):#len(beta)):    
                for ttt in range(0,len(thres)):
                    print(i_io,j_io,k_io,l_io,b,ttt)

                    @change_w_scale(w=w[j_io],scale=scale,beta=None)
                    def phi(X,Y,Z,w=None,scale=None,beta=None):
                        if scale is None: scale=5
                        if w is None: w=1
                        C=X+Y*1j
                        #A=C/scale + w+1
                        #Cout=np.log(0.5*(A+np.sqrt(A*A-4*w)))
                        #A=C/scale
                        #Cout=np.log(0.5*(A+np.sqrt(A*A-4*w)))
                        Cout=np.power(C,1./w)
                        return np.real(Cout), np.imag(Cout), Z

                    @change_w_scale(w=w[j_io], scale=scale,beta=None)
                    def phiInv(U,V,W,w=None,scale=None,beta=None):
                        if scale is None: scale = 5
                        if w is None: w=1
                        C = U + V * 1j
                        #Cout = np.exp(C)
                        #result = scale*(Cout-1 + w*(1/Cout-1))
                        #result = scale*(Cout+ w*(1/Cout))
                        result = np.power(C,w)
                        return np.real(result), np.imag(result), W

                    @change_w_scale(w=w[j_io], scale=scale,beta=beta)
                    def dphi(X,Y,Z,w=None, scale=None, beta=None):
                        if scale is None: scale = 5
                        if w is None: w=1
                        if beta is None: beta=0
                        #rot=cmath.rect(1,beta)
                        C=X+Y*1j
                        #C=C*rot
                        # A = C / scale + w +1
                        # dCout= 1/np.sqrt(-4 * w + A*A)*(1/scale)
                        dCout = (1/w)*np.power(C,1/w-1) 
                        norm = np.sqrt(np.real(dCout)*np.real(dCout) +np.imag(dCout)*np.imag(dCout))
                        zeros = np.zeros(X.shape)
                        ones = np.ones(X.shape)
                        v1 = [np.imag(dCout) / norm, np.real(dCout) / norm, zeros]
                        v2 = [v1[1], -v1[0], zeros]
                        v3 = [zeros, zeros, ones]
                        return v1, v2, v3

                    
                    @L1L2L3_drt_w_scale(drt=drt,w=w[j_io],scale=scale,beta=None)
                    @np.vectorize
                    def L1L2L3(X,Y,Z,w=None,scale=None,drt=None,beta=None):
                        if scale is None: scale = 5
                        #if drt is None: drt=1.5 #drt is the radial coordinate where we transition from tang to rad
                        if w is None: w=1
                        if beta is None: beta=0

                        l1 = 0.01
                        l2 = l1/100
                        l3 = 0.00
                        rot = cmath.rect(1, beta)
                        C = X + Y * 1j
                        C = C * rot
                        A = C / scale + w+1
                        #Cout = np.log(0.5*(A+np.sqrt(A*A-4*w)))
                        Cout = np.log(0.5*(A+np.sqrt(A*A-4*w)))
                        U= np.real(Cout)

                        #if U<drt:
                        L1=l1
                        L2=l2
                        L3=l3
                        #else:
                        #    L1 = l1
                        #    L2 = l2
                        #    L3 = l3
                        return L1,L2,L3

                    @L1L2L3_drt_w_scale(drt=drt,w=w[j_io],scale=scale,beta=None)
                    @np.vectorize
                    def windowFunction(X,Y,Z,w=None,scale=None,drt=None,beta=None):
                        if scale is None: scale = 5
                        #if drt is None: drt=1.5 #drt is the radial coordinate where we transition from tang to rad
                        if w is None: w=1
                        if beta is None: beta=0

                        uu = 1/1.75
                        u = 0.02
                        #x_h= 0.5*
                        x_h= 0.5*(phiInv(uu,0,0,w=w)[0]+phiInv(u,0,0,w=w)[0])
                
                        drt= phi(x_h,0,0,beta=beta)[0]

                        rot = cmath.rect(1, beta)
                        C = X + Y * 1j
                        C = C * rot
                        A = C / scale + w+1
                        #Cout = np.log(0.5*(A+np.sqrt(A*A-4*w)))
                        Cout=np.power(C,1/w)
                        U= np.real(Cout)

                        # if U<drt:
                        #     out=1
                        # else:
                        #     out=0

                        lk=60
                        out=1/(1+np.exp(-lk*(U-drt)))

                        return out

                    
                    base = "/home/u2hussai/scratch/simulations/diffusionSimulations_LambdaStep-k-60_scale-50_res-"
                    npath=base + str(int(res[i_io] * 1000)) + "_um-drt-" +str(int(drt[l_io] * 100)) +"_w-"+\
                        str(int(round(w[j_io],5)*1000))+"_angthr-" + str(int(ang_thr[k_io])) + "_beta-"+str(int(beta[b]*100))+"/"
                    maskpath=base + str(int(res[0] * 1000)) + "_um-drt-" +str(int(drt[l_io] * 100)) +"_w-"+\
                        str(int(round(w[j_io],5)*1000))+"_angthr-" + str(int(ang_thr[k_io])) + "_beta-"+str(int(beta[b]*100))+"/"

                    nlines = lines(npath + 'native_streamlines.vtk')
                    ulines = lines(npath + 'from_unfold_streamlines.vtk')
                    simlines = linesFromSims(npath, nlines, ulines,phi, phiInv,drt[l_io],maskpath)

                    simlines.filter(simlines.nlines,dphi)
                    simlines.filter(simlines.ulines,dphi)
                    save_vtk_streamlines(simlines.nlines.seedlines, npath+'seedlines_tang_native.vtk')
                    save_vtk_streamlines(simlines.ulines.seedlines, npath+'seedlines_tang_unfold.vtk')


                    sens_roc[ttt,0],spec_roc[ttt,0],tangCount[0],radCount[0]= simlines.thresholdSenSpec(simlines.nlines, thres[ttt])
                    sens_roc[ttt,1], spec_roc[ttt, 1],tangCount[1],radCount[1] = simlines.thresholdSenSpec(simlines.ulines,thres[ttt])
                    #print(sens_roc)
                    tangCountPara[0]=len(simlines.ntanglines)
                    tangCountPara[1]=len(simlines.utanglines)
                    radCountPara[0]=len(simlines.nradlines)
                    radCountPara[1]=len(simlines.uradlines)

            print(npath)

            np.save("/home/u2hussai/scratch/simulations_sens_spec_curv_tang/"+"sens_"+str(int(res[i_io] * 1000)) + "um_drt-" + str(int(drt[l_io] * 100)) + "_w-" + \
                str(int(round(w[j_io], 2) * 1000)) + "_angthr-" + str(int(ang_thr[k_io]))+".npy",sens_roc)

            np.save("/home/u2hussai/scratch/simulations_sens_spec_curv_tang/"+"spec_"+str(int(res[i_io] * 1000)) + "um_drt-" + str(int(drt[l_io] * 100)) + "_w-" + \
                str(int(round(w[j_io], 2) * 1000)) + "_angthr-" + str(int(ang_thr[k_io]))+".npy",spec_roc)

            np.save("/home/u2hussai/scratch/simulations_sens_spec_curv_tang/"+"tang_"+str(int(res[i_io] * 1000)) + "um_drt-" + str(int(drt[l_io] * 100)) + "_w-" + \
                str(int(round(w[j_io], 2) * 1000)) + "_angthr-" + str(int(ang_thr[k_io]))+".npy",tangCount)

            np.save("/home/u2hussai/scratch/simulations_sens_spec_curv_tang/"+"rad_"+str(int(res[i_io] * 1000)) + "um_drt-" + str(int(drt[l_io] * 100)) + "_w-" + \
                str(int(round(w[j_io], 2) * 1000)) + "_angthr-" + str(int(ang_thr[k_io]))+".npy",radCount)

            np.save("/home/u2hussai/scratch/simulations_sens_spec_curv_tang/"+"tangPara_"+str(int(res[i_io] * 1000)) + "um_drt-" + str(int(drt[l_io] * 100)) + "_w-" + \
                str(int(round(w[j_io], 2) * 1000)) + "_angthr-" + str(int(ang_thr[k_io]))+".npy",tangCountPara)

            np.save("/home/u2hussai/scratch/simulations_sens_spec_curv_tang/"+"radPara_"+str(int(res[i_io] * 1000)) + "um_drt-" + str(int(drt[l_io] * 100)) + "_w-" + \
                str(int(round(w[j_io], 2) * 1000)) + "_angthr-" + str(int(ang_thr[k_io]))+".npy",radCountPara)

            