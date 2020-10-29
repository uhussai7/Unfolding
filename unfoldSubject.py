import diffusion
import coordinates
import numpy as np
import nibabel as nib
from scipy.interpolate import griddata
import copy
from scipy.spatial import KDTree


class unfoldSubject:
    def __init__(self):
        self.coords= []
        self.T1= []
        self.diff = diffusion.diffVolume()
        self.diffUnfold=[]
        self.diff_nograd=[]
        # -------put other volumes functions----------------#

        # -----------------------------------------------#

    def loadCoordinates(self,path=None,prefix=None):
        self.coords=coordinates.coordinates(path,prefix)

    def loadDiffusion(self,path=None):
        self.diff.getVolume(folder=path)
        self.diff.shells()

    #-------put other load functions----------------#

    #-----------------------------------------------#

    def pushToUnfold(self,volume_nii,type=None):

        Ndims=volume_nii.header['dim']
        size = tuple(self.coords.X_uvw_nii.header['dim'][1:4])
        if Ndims[0]>3:
            size=size+ (Ndims[4],)
            volume_out_nii=np.zeros(size) #not yet nifti
            volume_out_nii[volume_out_nii==0]=np.NaN
            for i in range(0,Ndims[4]):
                print("vol: %d" % (i))
                points,S = coordinates.getPointsData(volume_nii,i)
                volume_out_nii[:,:,:,i]=griddata(points,S,
                                                (self.coords.X_uvw_nii.get_data(),
                                                 self.coords.Y_uvw_nii.get_data(),
                                                 self.coords.Z_uvw_nii.get_data()))
            volume_out_nii=nib.Nifti1Image(volume_out_nii,self.coords.X_uvw_nii.affine)
        else:
            volume_out_nii = np.zeros(size)  # not yet nifti
            volume_out_nii[volume_out_nii==0]=np.NaN
            points, S = coordinates.getPointsData(volume_nii)
            volume_out_nii= griddata(points, S,(self.coords.X_uvw_nii.get_data(),
                                                self.coords.Y_uvw_nii.get_data(),
                                                self.coords.Z_uvw_nii.get_data()))
            volume_out_nii = nib.Nifti1Image(volume_out_nii, self.coords.X_uvw_nii.affine)

        if type==None:
            return volume_out_nii

        if type=='diffusion':
            diff_out=copy.deepcopy(self.diff)
            diff_out.vol=volume_out_nii
            diff_out.grad_dev_nii=self.coords.gradDevUVW_nii
            diff_out.mask=self.makeMask(diff_out.vol)
            return diff_out



    def makeMask(self,volume_nii):
        Ndims = volume_nii.header['dim']
        size = tuple(self.coords.X_uvw_nii.header['dim'][1:4])
        volume_out_nii = np.zeros(size)  # not yet nifti
        if Ndims[0] > 3:
            i=0
        else:
            i=None

        points, S = coordinates.getPointsData(volume_nii, i)
        S[np.isnan(S)==0]=1
        volume_out_nii[:, :, :] = griddata(points, S,
                                           (self.coords.X_uvw_nii.get_data(),
                                            self.coords.Y_uvw_nii.get_data(),
                                            self.coords.Z_uvw_nii.get_data()))
        volume_out_nii = nib.Nifti1Image(volume_out_nii, self.coords.X_uvw_nii.affine)
        return volume_out_nii



    def diffNoGradDev(self):

        def invDistInterp(tree, S, vec):
            if S.shape[0] < 3:
                NN=1
            else:
                NN=3
            print(NN)
            dis, inds = tree.query(vec, NN)
            p = 1
            w = 1 / dis ** p
            S = S[inds]
            return np.sum(S * w) / np.sum(w)

        diff_no_grad_nii=copy.deepcopy(self.diffUnfold.vol.get_data())
        diff_no_grad_nii[:]=np.NaN
        graddevsum=np.sum(self.coords.gradDevUVW_nii.get_data(),axis=3)
        inds=np.asarray(np.where(np.isnan(graddevsum)==0))
        S=self.diffUnfold.vol.get_data()[inds[0],inds[1],inds[2],:]
        gd=self.coords.gradDevUVW_nii.get_data()[inds[0],inds[1],inds[2],:]
        gd=gd.reshape([len(inds[0]),3,3],order='F')
        #we want to generate a new image with bval directions
        for shell in range(0,len(self.diffUnfold.bvecs)):
            print(shell)
            bvecs=np.array( self.diffUnfold.bvecs_hemi_cart[shell])
            S_shell = S[:,self.diffUnfold.inds[shell]]
            S_shell_out=copy.deepcopy(S_shell)
            S_shell_out[:]=np.NaN
            for p in range(0,len(inds[0])):
                J=np.linalg.inv(gd[p,:,:])
                Jbvecs=np.einsum('ij,bj->bi',J,bvecs)
                Jbvecs_tree=KDTree(Jbvecs,4)
                b=0
                for bvec in bvecs:
                    S_shell_out[p,b]=invDistInterp(Jbvecs_tree,S_shell[p,:],bvec)
                    diff_no_grad_nii[inds[0][p],inds[1][p],inds[2][p],self.diffUnfold.inds[shell][b]]=S_shell_out[p,b]
                    b=b+1
        self.diff_nograd=diff_no_grad_nii









