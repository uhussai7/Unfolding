import numpy as np
from coordinates import domainParams, applyMask
import nibabel as nib
from dipy.core.gradients import gradient_table
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
import matplotlib.pyplot as plt


def diffusionTensor(L1,L2,L3,v1,v2,v3):


    P = np.moveaxis( [v1, v2, v3], 0, -1)
    P = np.moveaxis(P, 0, -1)
    Pinv = np.linalg.inv(P)
    L = np.asarray([L1,L2,L3])
    D = np.einsum('l...,lj->...lj',L,np.eye(3))

    diffD = np.matmul(np.matmul(P,D),Pinv)
    diffD = np.moveaxis(diffD,-1,0)

    return np.moveaxis(diffD, -1, 0)

class simulateDiffusion:
    def __init__(self, phi,dphi,phiInv, Uparams, L1L2L3,bvals, bvecs,N0=20):
        self.L1= []
        self.L2 = []
        self.L3 = []
        self.L1L2L3=L1L2L3
        self.gtab = gradient_table(bvals,bvecs)
        self.bvals = []
        self.bvecs = [] #this can be taken from hcp file and then use diffusion class to split shells
        self.bvalsSingle= []
        self.bvecsSingle = []
        self.Nparams=[]
        self.Uparams=Uparams
        self.diff_nii=[]
        self.mask_nii=[]
        self.U_nii=[]
        self.V_nii = []
        self.W_nii = []
        self.shells()
        self.v1 =[]
        self.v2 = []
        self.v3 = []
        self.dTensor=[]
        self.phi=phi
        self.dphi=dphi
        self.phiInv=phiInv
        self.N0=N0

    def simulate(self,path=None):

        #these are XYZ in terms of UVW
        X,Y,Z = self.phiInv(self.Uparams.A,self.Uparams.B,self.Uparams.C )


        #these are native space parameters
        Nx = self.N0
        dx = (np.nanmax(X) - np.nanmin(X)) / (Nx - 1)

        # self.Nparams = domainParams(np.nanmin(X), np.nanmax(X),
        #                             np.nanmin(Y), np.nanmax(Y),
        #                             self.Uparams.min_c, self.Uparams.max_c,
        #                             deltas=[dx, dx, dx])
        self.Nparams = domainParams(np.nanmin(X), np.nanmax(X),
                                    np.nanmin(Y), np.nanmax(Y),
                                    np.nanmin(Z), np.nanmax(Z),
                                    deltas=[dx, dx, dx])

        #make the native space coordinates
        print('Making native space coordinates...')
        self.U_nii, self.V_nii, self.W_nii= self.phi(self.Nparams.A,self.Nparams.B,self.Nparams.C)
        self.U_nii, self.V_nii, self.W_nii = applyMask(self.U_nii, self.V_nii, self.W_nii,self.Uparams)
        self.v1, self.v2, self.v3 = self.dphi(self.Nparams.A,self.Nparams.B,self.Nparams.C)

        # make the diffusion tensor
        print('Calculating diffusion tensor...')
        self.L1,self.L2,self.L3=self.L1L2L3(self.Nparams.A,self.Nparams.B,self.Nparams.C)
        self.dTensor = diffusionTensor(self.L1,self.L2,self.L3,self.v1,self.v2,self.v3)

        #generate the signal
        print('Generating diffusion signal')
        self.diff_nii = self.diffusionSignal()
        self.diff_nii[np.isnan( self.U_nii)==1,:]=np.NaN


        #make a mask
        self.mask_nii = np.copy(self.U_nii)
        self.mask_nii[np.isnan(self.mask_nii)==0]=1


        print('Coverting to nifti...')
        #make save the nifti coordinate files
        self.U_nii = nib.Nifti1Image(self.U_nii, self.Nparams.affine)
        self.V_nii = nib.Nifti1Image(self.V_nii, self.Nparams.affine)
        self.W_nii = nib.Nifti1Image(self.W_nii, self.Nparams.affine)
        self.diff_nii=nib.Nifti1Image(self.diff_nii,self.Nparams.affine)
        self.mask_nii=nib.Nifti1Image(self.mask_nii,self.Nparams.affine)

        print('Saving files...')
        self.saveAll(path)


    def saveAll(self,path):
        if path == None: path='./'
        nib.save(self.U_nii, path+ 'U.nii.gz')
        nib.save(self.V_nii, path+ 'V.nii.gz')
        nib.save(self.W_nii, path+ 'W.nii.gz')
        nib.save(self.diff_nii,path+'data.nii.gz')
        nib.save(self.mask_nii,path+'nodif_brain_mask.nii.gz')

        #save the bvalsSings and bvecsSingle files
        fbvals = open(path+'bvals','w')
        fbvecs = open(path + 'bvecs', 'w')

        for bval in self.bvalsSingle:
            fbvals.write("%f "  % (bval))
        fbvals.close()

        for i in range(0,3):
            for bvec in self.bvecsSingle:
                fbvecs.write("%f "% (bvec[i]))
            fbvecs.write("\n")
        fbvecs.close()

    def diffusionSignal(self):
        #def fxn(X,Y,Z,b,diffD,bvecs):

        #lets just make a single shell image, this can be changed later
        self.bvalsSingle= np.zeros(np.asarray(self.bvals[1].shape)+1)
        self.bvalsSingle[0]=0
        self.bvalsSingle[1:]=1000
        self.bvecsSingle= np.zeros(np.asarray(self.bvecs[1].shape)+(1,0))
        self.bvecsSingle[0]=[0,0,0]
        self.bvecsSingle[1:]=self.bvecs[1]

        #make meshgrid for diffusion
        b = np.arange(0,len(self.bvals[1])+1)
        x = np.linspace(self.Nparams.max_a,self.Nparams.max_a,self.Nparams.Na)
        y = np.linspace(self.Nparams.max_b, self.Nparams.max_b, self.Nparams.Nb)
        z = np.linspace(self.Nparams.max_c, self.Nparams.max_c, self.Nparams.Nc)
        X,Y,Z,B = np.meshgrid(x,y,z,b,indexing='ij')

        #compute the signal
        gDg = np.einsum('ijabc,abcdj->abcdi',self.dTensor,self.bvecsSingle[B])
        gDg= np.einsum('...k,...k->...',self.bvecsSingle[B],gDg)
        S_0 = 1000
        S=S_0*np.exp(-gDg*self.bvalsSingle[B])

        return S

    def shells(self):
        """
        Changes the format of the bvals and bvecs to make it a bit user-friendly
        :return: Updates bvals bvecs bvecs_hemi_..., etc in class
        """
        tempbvals = np.round(self.gtab.bvals, -2)
        inds_sort = np.argsort(tempbvals)
        bvals_sorted = self.gtab.bvals[inds_sort]
        bvecs_sorted = self.gtab.bvecs[inds_sort]
        tempbvals = np.sort(tempbvals)
        gradbvals = np.gradient(tempbvals)
        inds_shell_cuts = np.where(gradbvals != 0)
        shell_cuts = []
        for i in range(int(len(inds_shell_cuts[0]) / 2)):
            shell_cuts.append(inds_shell_cuts[0][i * 2])
        shell_cuts.insert(0, -1)
        shell_cuts.append(len(bvals_sorted))
        temp_bvals = []
        temp_bvecs = []
        temp_inds = []
        for t in range(int(len(shell_cuts) - 1)):
            print(shell_cuts[t] + 1, shell_cuts[t + 1])
            temp_bvals.append(bvals_sorted[shell_cuts[t] + 1:1 + shell_cuts[t + 1]])
            temp_bvecs.append(bvecs_sorted[shell_cuts[t] + 1:1 + shell_cuts[t + 1]])
            temp_inds.append(inds_sort[shell_cuts[t] + 1:1 + shell_cuts[t + 1]])
        self.bvals = temp_bvals
        self.bvecs = temp_bvecs
        self.inds = temp_inds
        self.inds = np.asarray(self.inds)


    def plotTangVecs(self):
        fig = plt.figure()
        ax = fig.gca(projection='3d')

        z_mid=int(round(self.Nparams.A.shape[2]/2))

        skip = (slice(None, None, 2), slice(None, None, 2), slice(z_mid, self.Nparams.A.shape[2], 1000))
        #skip1= (slice(None, None, 2), slice(None, None, 2), slice(None, None, 2))
        ax.quiver(self.Nparams.A[skip], self.Nparams.B[skip], self.Nparams.C[skip],
                  self.v1[0][skip], self.v1[1][skip], self.v1[2][skip],
                  length=0.1,normalize=True
                  )
        ax.quiver(self.Nparams.A[skip], self.Nparams.B[skip], self.Nparams.C[skip],
                   self.v2[0][skip], self.v2[1][skip], self.v2[2][skip],
                   length=0.1,normalize=True,color='red'
                   )
        # ax.quiver(self.Nparams.A[skip], self.Nparams.B[skip], self.Nparams.C[skip],
        #            self.v3[0][skip], self.v3[1][skip], self.v3[2][skip],
        #            length=0.1,normalize=True
        #            )
        ax.set_zlim(np.nanmin(self.Nparams.C[:]), np.nanmax(self.Nparams.C[:]))
        plt.show()