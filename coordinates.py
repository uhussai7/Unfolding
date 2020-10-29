import ioFunctions
import numpy as np
from scipy.interpolate import griddata
import nibabel as nib

class domainParams:
    def __init__(self,min_a,max_a,min_b,max_b,min_c,max_c,dims=[None],deltas=[None]):
        self.min_a = min_a
        self.max_a = max_a
        self.min_b = min_b
        self.max_b = max_b
        self.min_c = min_c
        self.max_c = max_c
        if dims[0]==None and deltas[0]==None:
            raise ValueError('Please provide either number of voxels dims=[Na,Nb,Nc] or provide'
                             'size of voxels deltas=[da,db,dc] if both provided, dims is taken'
                             ' and deltas are discarded')
        if dims[0] !=None:
            self.Na = dims[0]
            self.Nb = dims[1]
            self.Nc = dims[2]
            self.da = (max_a - min_a) / (self.Na - 1)
            self.db = (max_b - min_b) / (self.Nb - 1)
            self.dc = (max_c - min_c) / (self.Nc - 1)
        else:
            self.da = deltas[0]
            self.db = deltas[1]
            self.dc = deltas[2]
            self.Na = round((max_a - min_a) / self.da + 1)
            self.Nb = round( (max_b - min_b) / self.db + 1)
            self.Nc = round((max_c - min_c) / self.dc + 1)

        self.affine = np.asarray([[ self.da,       0,       0,    self.min_a],
                                 [        0, self.db,       0,    self.min_b],
                                 [        0,       0, self.dc,    self.min_c],
                                 [        0,       0,       0,             1]])
        self.A, self.B,self.C = self.makeMeshGrid()

        
    def makeMeshGrid(self):
        A = np.linspace(self.max_a, self.min_a, self.Na)
        B = np.linspace(self.max_b, self.min_b, self.Nb)
        C = np.linspace(self.max_c, self.min_c, self.Nc)
        return np.meshgrid(A, B, C, indexing='ij')

    def griddata3(self,points,S):
        Sout1 = griddata(points, S[0], (self.A, self.B, self.C))
        Sout2 = griddata(points, S[1], (self.A, self.B, self.C))
        Sout3 = griddata(points, S[2], (self.A, self.B, self.C))
        return [Sout1, Sout2, Sout3]

def getPointsData(img_nii, i=None):
    if i == None:
        S = img_nii.get_data()
        inds = np.transpose(np.asarray(np.where(np.isnan(S) == 0)))
        S = S[np.isnan(S) == 0]
    else:
        S = img_nii.get_data()[:, :, :, i]
        inds = np.transpose(np.asarray(np.where(np.isnan(S) == 0)))
        S = S[np.isnan(S) == 0]

    worlds = toWorld(img_nii, inds)
    X = worlds[:, 0]
    Y = worlds[:, 1]
    Z = worlds[:, 2]

    points = np.asarray([X, Y, Z]).transpose()

    return points, S

def applyMask(U, V, W, dParams):
    """
    Makes a mask based on min max of curvilinear coordinates
    :param U,V,W: Coordinates
    :param dParams: domain parameter class dParams
    :return:
    """
    condition = np.invert(((U >= dParams.min_a) &
                           (U <= dParams.max_a) &
                           (V >= dParams.min_b) &
                           (V <= dParams.max_b) &
                           (W >= dParams.min_c) &
                           (W <= dParams.max_c)))

    U[condition] = np.NaN
    V[condition] = np.NaN
    W[condition] = np.NaN

    return U, V, W


def toWorld(nii,inds):
    world = []
    for ind in inds:
        world.append(np.matmul(nii.affine,np.append(ind,1) )[0:3])
    return np.asarray(world)

def toInds(nii,worlds):
    affine=np.linalg.inv(nii.affine)
    inds=[]
    for world in worlds:
        inds.append((np.matmul(affine,np.append(world,1)))[0:3])
    return np.asarray(inds)

class coordinates:
    """
    this is a class to handle various operations
    related to 3d coordinates on top of usual
    cartesian coordinates
    """
    def __init__(self,path=None,prefix=None):
        self.U_xyz_nii=[]
        self.V_xyz_nii=[]
        self.W_xyz_nii=[]
        self.X_uvw_nii=[] #this will be in world coordinates
        self.Y_uvw_nii=[] #this will be in world coordinates
        self.Z_uvw_nii=[] #this will be in world coordinates
        self.X_uvw_a = []  # these are to calculate arc length
        self.Y_uvw_a = []  # these are to calculate arc length
        self.Z_uvw_a = []  # these are to calculate arc length
        # the following five may not need to be stored
        self.mean_u = []
        self.mean_v = []
        self.mean_w = []
        self.grads = []
        self.cumsum=[]
        self.U = []  # these are as lists, XYZ is in world coordinates, size of lists inferred from inputs
        self.V = []
        self.W = []
        self.X = []
        self.Y = []
        self.Z = []
        self.Uparams=[]
        self.Nparams=[]
        self.gradDevXYZ_nii=[]
        self.gradDevUVW_nii=[]
        self.gradDevPolarDecompUVW_nii=[]
        self.loadCoordinates(path,prefix)
        self.initialize()


    def loadCoordinates(self,path=None,prefix=None):
        if path is None:
            raise ValueError("Please provide path (with trailing slash) for nifti coordinate files, path=...")
        if prefix is None:
            raise ValueError("Please provide provide prefix for coordinates. prefix=... "
                             "Coordinates should end with ...U.nii.gz, ...V.nii.gz, ...W.nii.gz")


        self.U_xyz_nii = ioFunctions.loadVol(path+prefix+"U.nii.gz")
        self.V_xyz_nii = ioFunctions.loadVol(path+prefix+"V.nii.gz")
        self.W_xyz_nii = ioFunctions.loadVol(path+prefix+"W.nii.gz")

    def initialize(self):

        self.meanArcLength()
        self.nativeUnfold()

    def meanArcLength(self):

        print("Inverting coordinates...")
        self.U = self.U_xyz_nii.get_data()[np.isnan(self.U_xyz_nii.get_data()) == 0]
        self.V = self.V_xyz_nii.get_data()[np.isnan(self.V_xyz_nii.get_data()) == 0]
        self.W = self.W_xyz_nii.get_data()[np.isnan(self.W_xyz_nii.get_data()) == 0]

        inds = np.transpose(np.asarray(np.where(np.isnan(self.U_xyz_nii.get_data()) == 0)))
        worlds = toWorld(self.U_xyz_nii, inds)
        
        self.X = worlds[:, 0]
        self.Y = worlds[:, 1]
        self.Z = worlds[:, 2]

        # the following Nparams is wrong because you are using X,Y,Z only where isnan(U...) == 0
        # self.Nparams=domainParams(min(self.X),max(self.X),
        #                           min(self.Y),max(self.Y),
        #                           min(self.Z),max(self.Z),
        #                           dims=np.asarray(self.U_xyz_nii.get_data().shape))

        # compute mean arclength
        N = 16  # make NxNxN grid and calculate arclength
        UALParams = domainParams(min(self.U), max(self.U), 
                                 min(self.V), max(self.V), 
                                 min(self.W), max(self.W), dims=[N, N, N])

        # create X_uvw, Y_uvw, Z_uvw for mean arclength
        points = np.asarray([self.U, self.V, self.W]).transpose()

        print("Computing mean arc lengths...")
        [self.X_uvw_a, self.Y_uvw_a, self.Z_uvw_a]=UALParams.griddata3(points,[self.X,self.Y,self.Z])

        self.grads=np.zeros((3,3)+self.X_uvw_a.shape)
        self.cumsum=np.zeros((3,)+self.X_uvw_a.shape)
        self.grads[:]=np.NaN
        self.cumsum[:] = np.NaN

        Xi_uvw_a=[self.X_uvw_a, self.Y_uvw_a, self.Z_uvw_a]

        #compute the grads
        for i in range(0,3):
            self.grads[i][0],self.grads[i][1],self.grads[i][2]= np.gradient(Xi_uvw_a[i],edge_order=1)

        #calculate the distances
        for i in range(0,3):
            self.cumsum[i]=np.sqrt(self.grads[0][i]*self.grads[0][i]+ \
                      self.grads[1][i] * self.grads[1][i] + \
                      self.grads[2][i] * self.grads[2][i])

        #calculate cumalative sum
        for i in range(0,3):
            self.cumsum[i] = np.nancumsum(self.cumsum[i], axis=i)

        #now calculate the mean arclenth
        self.mean_u = abs(self.cumsum[0][0, :, :] - self.cumsum[0][-1, :, :])
        self.mean_v = abs(self.cumsum[1][:, 0, :] - self.cumsum[1][:, -1, :])
        self.mean_w = abs(self.cumsum[2][:, :, 0] - self.cumsum[2][:,  :,-1])
        self.mean_u[self.mean_u == 0] = np.NaN
        self.mean_v[self.mean_v == 0] = np.NaN
        self.mean_w[self.mean_w == 0] = np.NaN
        self.mean_u = np.nanmean(self.mean_u.flatten())
        self.mean_v = np.nanmean(self.mean_v.flatten())
        self.mean_w = np.nanmean(self.mean_w.flatten())

        self.U = self.mean_u * (self.U - np.nanmin(self.U)) / np.nanmax(self.U)
        self.V = self.mean_v * (self.V - np.nanmin(self.V)) / np.nanmax(self.V)
        self.W = self.mean_w * (self.W - np.nanmin(self.W)) / np.nanmax(self.W)

    def nativeUnfold(self):
        """
        Native world coordinates in terms of unfold
        :return:
        """
        print("Creating cartesian coordinates in terms of mean arc length unfolded space...")
        res = self.U_xyz_nii.header['pixdim'][1]
        #
        self.Uparams = domainParams(np.nanmin(self.U), np.nanmax(self.U),  # these are arclength corrected
                               np.nanmin(self.V), np.nanmax(self.V),
                               np.nanmin(self.W), np.nanmax(self.W),
                               deltas=[res, res, res])
        points = np.asarray([self.U, self.V, self.W]).transpose()
        self.X_uvw_nii, self.Y_uvw_nii, self.Z_uvw_nii = self.Uparams.griddata3(points,[self.X,self.Y,self.Z])

        affine=self.Uparams.affine
        self.X_uvw_nii = nib.Nifti1Image(self.X_uvw_nii, affine)
        self.Y_uvw_nii = nib.Nifti1Image(self.Y_uvw_nii, affine)
        self.Z_uvw_nii = nib.Nifti1Image(self.Z_uvw_nii, affine)



    def computeGradDev(self):
        """
        Computes all the different graddevs
        :return: grad_dev_nii
        """
        C = [self.U_xyz_nii.get_data(),self.V_xyz_nii.get_data(),self.W_xyz_nii.get_data()]

        grads = np.zeros((C[0].shape)+(3,3))

        for i in range(0,3):
            grads[:,:,:,i,0], grads[:,:,:,i,1],grads[:,:,:,i,2]=self.gradientNaN(C[i])

        grads=grads-np.identity(3)
        self.gradDevXYZ_nii=grads.reshape(grads.shape[0:3]+(9,),order='F') #not yet nifti
        self.gradDevXYZ_nii=nib.Nifti1Image(self.gradDevXYZ_nii,self.U_xyz_nii.affine)

        #move this data to unfolded space
        self.gradDevUVW_nii=np.zeros(self.Uparams.A.shape+(9,))
        self.gradDevUVW_nii[self.gradDevUVW_nii==0]=np.NaN


        for i in range(0,9):
            points, S = getPointsData(self.gradDevXYZ_nii,i)
            self.gradDevUVW_nii[:,:,:,i]=griddata(points,S,
                                                  (self.X_uvw_nii.get_data(),
                                                   self.Y_uvw_nii.get_data(),
                                                   self.Z_uvw_nii.get_data())
                                                  )

        self.gradDevUVW_nii=nib.Nifti1Image(self.gradDevUVW_nii,self.X_uvw_nii.affine)


    def gradientNaN(self,E):

        def nanChecker(A,B,E=None):
            if (np.isnan(A) == True and np.isnan(B) == True):
                C = np.NaN
            if(np.isnan(A) == True and np.isnan(B) == False):
                C = B
            if(np.isnan(A) == False and np.isnan(B) == True):
                C = A
            if(np.isnan(A) == False and np.isnan(B) == False):
                C = 0.5 * (A + B)
            return C

        dxEL=np.copy(E)
        dxEL[:]=np.NaN
        dxER = np.copy(dxEL)
        dxE = np.copy(dxEL)
        dyEL = np.copy(dxEL)
        dyER = np.copy(dxEL)
        dyE = np.copy(dxEL)
        dzEL = np.copy(dxEL)
        dzER = np.copy(dxEL)
        dzE = np.copy(dxEL)

        for i in range(0,E.shape[0]):
            for j in range(0,E.shape[1]):
                for k in range(0,E.shape[2]):

                    if i+1<E.shape[0]:
                        dxEL[i, j, k] = E[i + 1, j, k] - E[i, j, k]
                    if i>0:
                        dxER[i, j, k] = E[i, j, k] - E[i - 1, j, k]
                    dxE[i,j,k]=nanChecker(dxEL[i, j, k],dxER[i, j, k],E[i,j,k])

                    if j+1<E.shape[1]:
                        dyEL[i, j, k] = E[i , j + 1, k] - E[i, j, k]
                    if j>0:
                        dyER[i, j, k] = E[i, j, k] - E[i, j-1, k]
                    dyE[i, j, k] = nanChecker(dyEL[i, j, k], dyER[i, j, k],E[i,j,k])

                    if k+1< E.shape[2]:
                        dzEL[i, j, k] = E[i, j, k+1] - E[i, j, k]
                    if k> 0:
                        dzER[i, j, k] = E[i, j, k] - E[i, j, k-1]
                    dzE[i,j,k]=nanChecker(dzEL[i, j, k],dzER[i, j, k],E[i,j,k])

        return dxE,dyE,dzE






    def matrix2gradev(self):
        pass

    def graddev2matrix(self,graddev):
        """
        :param graddev: ...,3,3 matrix to be converted to [...,9]
        :return: graddev with size [...,9]
        """








