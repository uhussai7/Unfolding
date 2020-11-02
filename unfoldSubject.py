import diffusion
import coordinates
import numpy as np
import nibabel as nib
from scipy.interpolate import griddata
import copy
from scipy.spatial import KDTree
from pyshtools.shtools import SHExpandLSQ, MakeGridPoint

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
        volume_out_nii=copy.deepcopy(self.coords.X_uvw_nii.get_fdata())
        volume_out_nii[np.isnan(volume_out_nii)==0]=1
        return nib.Nifti1Image(volume_out_nii,self.coords.X_uvw_nii.affine)



    def diffNoGradDev(self):

        def invDistInterp(tree, S, vec):
            if S.shape[0] < 3:
                NN=1
            else:
                NN=5
            #print(NN)
            dis, inds = tree.query(vec, NN)
            p = 4
            w = 1 / dis ** p
            S = S[inds]
            return np.sum(S * w) / np.sum(w)

        def cart2latlon(x, y, z):
            R = x * x + y * y + z * z
            lat = np.arcsin(z / R)
            lon = np.arctan2(y, x)
            return lat, lon

        #diff_no_grad_nii=copy.deepcopy(self.diffUnfold.vol.get_data())
        diff_no_grad_nii = np.zeros(self.diffUnfold.vol.get_data().shape)
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

            #too pull vector from unfolded to native
            #lat, lon = cart2latlon(bvecs[:, 0], bvecs[:, 1], bvecs[:, 2])

            for p in range(0,len(inds[0])):
                J=(gd[p,:,:])+np.eye(3)
                Jbvecsp=np.einsum('ij,bj->bi',J,bvecs)
                Jbvecs=copy.deepcopy(Jbvecsp)
                for b in range(0,Jbvecs.shape[0]):
                    Jbvecs[b,:]= np.asarray( Jbvecsp[b,:])/np.asarray( np.linalg.norm(Jbvecsp[b,:]))
                #Jbvecs=Jbvecs.T
                #Jbvecs_tree=KDTree(Jbvecs,30)
                b=0
                lat,lon = cart2latlon(Jbvecs[:,0],Jbvecs[:,1],Jbvecs[:,2])
                cilm=SHExpandLSQ(S_shell[p,:],lat,lon,140)
                for bvec in bvecs:
                    #S_shell_out[p,b]=invDistInterp(Jbvecs_tree,S_shell[p,:],bvec)
                    #J=np.linalg.inv(J)
                    #diff_no_grad_nii[inds[0][p],inds[1][p],inds[2][p],self.diffUnfold.inds[shell][b]]=invDistInterp(Jbvecs_tree,S_shell[p,:],bvec)#S_shell_out[p,b]
                    if shell > 0:
                        #vec = np.matmul(J, bvec)
                        #vec=vec/np.linalg.norm(vec)
                        latp,lonp=cart2latlon(bvec[0],bvec[1],bvec[2])
                        #diff_no_grad_nii[inds[0][p],inds[1][p],inds[2][p],self.diffUnfold.inds[shell][b]]=griddata(Jbvecs,S_shell[p,:],bvec)
                        diff_no_grad_nii[
                             inds[0][p],
                             inds[1][p],
                             inds[2][p],
                             self.diffUnfold.inds[shell][b]]=MakeGridPoint(cilm[0],latp,lonp)
                    else:
                         diff_no_grad_nii[inds[0][p], inds[1][p], inds[2][p], self.diffUnfold.inds[shell][b]]=S_shell[p,0]
                    #vec=np.matmul(J,bvec)
                    #vec=vec/np.linalg.norm(vec)
                    #diff_no_grad_nii[inds[0][p], inds[1][p], inds[2][p],
                    #                 self.diffUnfold.inds[shell][b]] = invDistInterp(self.diffUnfold.bvecs_hemi_cart_kdtree[shell],
                    #                                                                 S_shell[p,:],
                    #                                                                 vec)  # S_shell_out[p,b]
                    b=b+1
        self.diff_nograd=diff_no_grad_nii









