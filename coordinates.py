import ioFunctions
import numpy as np
from scipy.interpolate import griddata
import nibabel as nib

class domainParams:
    def __init__(self,min_a,max_a,min_b,max_b,min_c,max_c,Na,Nb,Nc):
        self.min_a = min_a
        self.max_a = max_a
        self.min_b = min_b
        self.max_b = max_b
        self.min_c = min_c
        self.max_c = max_c
        self.Na = Na
        self.Nb = Nb
        self.Nc = Nc
        self.da=  (max_a - min_a) / (Na - 1)
        self.db = (max_a - min_a) / (Na - 1)
        self.dc = (max_a - min_a) / (Na - 1)
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
    def __init__(self):
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

        print("Inverting coordinates...")
        U = self.U_xyz_nii.get_data()
        V = self.V_xyz_nii.get_data()
        W = self.W_xyz_nii.get_data()

        U=U[np.isnan(self.U_xyz_nii.get_data()) == 0]
        V=V[np.isnan(self.U_xyz_nii.get_data()) == 0]
        W=W[np.isnan(self.U_xyz_nii.get_data()) == 0]

        inds = np.transpose(np.asarray(np.where(np.isnan(self.U_xyz_nii.get_data())==0)))
        worlds = toWorld(self.U_xyz_nii,inds)

        X = worlds[:, 0]
        Y = worlds[:, 1]
        Z = worlds[:, 2]

        #compute mean arclength
        N=16 #make NxNxN grid and calculate arclength

        u = np.linspace(min(U), max(U), N)
        v = np.linspace(min(V), max(V), N)
        w = np.linspace(min(W), max(W), N)

        uu, vv, ww = np.meshgrid(u,v,w, indexing='ij')

        #create X_uvw, Y_uvw, Z_uvw for mean arclength
        points=np.asarray([U,V,W]).transpose()

        print("Computing mean arc lengths...")
        self.X_uvw_a = griddata(points,X, (uu,vv,ww))
        self.Y_uvw_a = griddata(points,Y, (uu,vv,ww))
        self.Z_uvw_a = griddata(points, Z, (uu, vv, ww))

        #call method for mean arclength
        self.meanArcLength()

        print("Creating domain with mean arc lengths...")
        #recompute unfolded space with mean arclength
        U=  U - np.nanmin(U)
        V = V - np.nanmin(V)
        W = W - np.nanmin(W)


        U=  self.mean_u* U / np.nanmax(U)
        V = self.mean_v * V / np.nanmax(V)
        W = self.mean_w * W / np.nanmax(W)

        res=self.U_xyz_nii.header['pixdim'][1]
        Nu = np.round(self.mean_u / res + 1)
        Nv = np.round(self.mean_v / res + 1)
        Nw = np.round(self.mean_w / res + 1)


        #choose each N to match incoming resolution
        u = np.linspace(np.nanmin(U), np.nanmax(U), Nu)
        v = np.linspace(np.nanmin(V), np.nanmax(V), Nv)
        w = np.linspace(np.nanmin(W), np.nanmax(W), Nw)

        uu, vv, ww = np.meshgrid(u, v, w, indexing='ij')

        # create X_uvw, Y_uvw, Z_uvw for mean arclength
        points = np.asarray([U, V, W]).transpose()

        self.X_uvw_nii = griddata(points, X, (uu, vv, ww))
        self.Y_uvw_nii = griddata(points, Y, (uu, vv, ww))
        self.Z_uvw_nii = griddata(points, Z, (uu, vv, ww))

        #turn in to nifti image
        affine=[[res,   0,   0, u[0]],
                [  0, res,   0, v[0]],
                [  0,   0, res, w[0]],
                [0,0,0,1]
                ]
        self.X_uvw_nii = nib.Nifti1Image(self.X_uvw_nii,affine)
        self.Y_uvw_nii = nib.Nifti1Image(self.Y_uvw_nii, affine)
        self.Z_uvw_nii = nib.Nifti1Image(self.Z_uvw_nii, affine)




    def meanArcLength(self):

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

