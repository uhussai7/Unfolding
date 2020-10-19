import ioFunctions
import numpy as np
from scipy.interpolate import griddata

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

        uu, vv, ww = np.meshgrid(u,v,w)

        #create X_uvw, Y_uvw, Z_uvw for mean arclength
        points=np.asarray([U,V,W]).transpose()

        self.X_uvw_a = griddata(points,X, (uu,vv,ww))
        self.Y_uvw_a = griddata(points,Y, (uu,vv,ww))
        self.Z_uvw_a = griddata(points, Z, (uu, vv, ww))

    def meanArcLength(self):

        self.grads=np.zeros((3,3)+self.X_uvw_a.shape)
        cumsum=np.zeros((3,)+self.X_uvw_a.shape)

        Xi_uvw_a=[self.X_uvw_a, self.Y_uvw_a, self.Z_uvw_a]

        #compute the grads
        for i in range(0,3):
            self.grads[i][0],self.grads[i][1],self.grads[i][2]= np.gradient(Xi_uvw_a[i],edge_order=1)

        #calculate the distances
        for i in range(0,3):
            cumsum[i]=np.sqrt(self.grads[0][i]*self.grads[0][i]+ \
                      self.grads[1][i] * self.grads[1][i] + \
                      self.grads[2][i] * self.grads[2][i])

        #calculate cumalative sum
        # for i in range(0,3):
        #     cumsum[i] = np.nancumsum(cumsum[i],axis=i)
        #     cumsum[i] = np.nancumsum(cumsum[i], axis=i)
        #     cumsum[i] = np.nancumsum(cumsum[i], axis=i)


        self.cumsum=cumsum

        #now calculate the mean arclenth

        self.mean_u = abs(cumsum[0][0, :, :] - cumsum[0][-1, :, :])
        self.mean_v = abs(cumsum[1][:, 0, :] - cumsum[1][:, -1, :])
        self.mean_w = abs(cumsum[2][:, :, 0] - cumsum[2][:, :, -1])



        # self.mean_u = np.nanmean(mean_u.flatten())
        # self.mean_v = np.nanmean(mean_v.flatten())
        # self.mean_w = np.nanmean(mean_w.flatten())

        #return mean_u, mean_v, mean_w


        #
        #         #compute the distance:
        # for i in range(self.X_uvw_a.shape(0)):
        #     for j in range(self.X_uvw_a.shape(1)):
        #         for k in range(self.X_uvw_a.shape(2)):
        #             u = grad[0][0]
        #




