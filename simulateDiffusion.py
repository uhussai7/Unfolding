import numpy as np
from coordinates import domainParams
import nibabel as nib
from dipy.core.gradients import gradient_table
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
import matplotlib.pyplot as plt

def maskCondition(Cout,dParams):
    return np.invert(((np.real(Cout) >= dParams.min_a) &
               (np.real(Cout) <= dParams.max_a) &
               (np.imag(Cout) >= dParams.min_b) &
               (np.imag(Cout) <= dParams.max_b)))

def conformal(X,Y,Z,dParams):

    def fxn(C):
        w=1
        return np.log((1 / 2) * (C + np.sqrt(C*C - 4 * w)))

    def dfxn(C):
        w=1
        return 1/np.sqrt(-4 * w + C*C)

    #the unfolding coordiantes
    Cout = fxn(X + Y*1j)

    #the tangent vectors
    dCout = dfxn(X + Y*1j)

    #apply domain mask
    condition = maskCondition(Cout,dParams)
    Cout[ condition]= np.NaN + np.NaN*1j
    Z[ condition] = np.NaN
    dCout[condition] = np.NaN + np.NaN*1j

    #for v3
    zeros = np.zeros(X.shape)
    ones = np.ones(X.shape)
    zeros[condition]= np.NaN
    ones[condition]= np.NaN

    #tangent vectors for output
    norm = np.sqrt(np.real(dCout)*np.real(dCout) +np.imag(dCout)*np.imag(dCout))
    v1 = [np.imag(dCout)/norm, np.real(dCout)/norm,zeros]
    v2 = [v1[1], -v1[0], zeros]
    v3 = [zeros, zeros, ones]

    return np.real(Cout), np.imag(Cout), Z, v1, v2, v3

def conformalInverse(z):
    z = np.exp(z)
    return  z + 1/z

def diffusionTensor(L1,L2,L3,v1,v2,v3):
    P = np.moveaxis( [v1, v2, v3], 0, -1)
    P = np.moveaxis(P, 0, -1)
    Pinv = np.linalg.inv(P)
    D = np.diag([L1,L2,L3])

    diffD = np.matmul(np.matmul(P,D),Pinv)
    diffD = np.moveaxis(diffD,-1,0)

    return np.moveaxis(diffD, -1, 0)

class simulateDiffusion:
    def __init__(self, L1,L2,L3,bvals, bvecs):
        self.L1= L1
        self.L2 = L2
        self.L3 = L3
        self.gtab = gradient_table(bvals,bvecs)
        self.bvals = []
        self.bvecs = [] #this can be taken from hcp file and then use diffusion class to split shells
        self.bvalsSingle= []
        self.bvecsSingle = []
        self.Nparams=[]
        self.Uparams=[]
        self.diff_nii=[]
        self.mask_nii=[]
        phiInverse=[]
        self.U_nii=[]
        self.V_nii = []
        self.W_nii = []
        self.shells()
        self.v1 =[]
        self.v2 = []
        self.v3 = []
        self.dTensor=[]

    def simulate(self,path=None):

        #these are the unfolded space domain parameters
        self.Uparams=domainParams(0.3,1.2,-np.pi/2,np.pi/2,0,0,50,50,0)
        a = np.linspace(self.Uparams.max_a,self.Uparams.min_a,self.Uparams.Na)
        b = np.linspace(self.Uparams.max_b,self.Uparams.min_b,self.Uparams.Nb)
        A, B = np.meshgrid(a,b,indexing='ij')
        C=A+B*1j
        C=conformalInverse(C) #this brings us back to native

        #these are native space parameters
        Na = 32
        da = (np.nanmax(np.real(C[:])-np.nanmin(np.real(C[:]))))/(Na-1)
        Nb = (np.nanmax(np.imag(C[:]))-np.nanmin(np.imag(C[:])))/da +1
        Nc = 4
        self.Nparams = domainParams(np.nanmin(np.real(C[:])), np.nanmax(np.real(C[:])),
                               np.nanmin(np.imag(C[:])), np.nanmax(np.imag(C[:])),
                               0,(Nc-1)*da,
                               Na,
                               Nb,
                               Nc)

        #make the native space coordinates
        print('Making native space coordinates...')
        self.U_nii, self.V_nii, self.W_nii, self.v1, self.v2, self.v3 = conformal(self.Nparams.A,
                                                                                  self.Nparams.B,
                                                                                  self.Nparams.C,
                                                                                  self.Uparams)

        # make the diffusion tensor
        print('Calculating diffusion tensor...')
        L1 = 99.9E-4
        L2 = 0.1E-4
        L3 = 0.00
        self.dTensor = diffusionTensor(L1,L2,L3,self.v1,self.v2,self.v3)

        #generate the signal
        print('generate diffusion signal')
        self.diff_nii = self.diffusionSignal()

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

        print('saving files...')
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