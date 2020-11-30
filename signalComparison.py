import dipy
from dipy.io.image import load_nifti
import coordinates
import nibabel as nib
from scipy.interpolate import griddata, LinearNDInterpolator
import numpy as np

difff1=nib.load('/home/uzair/PycharmProjects/Unfolding/data/diffusionSimulations_res-6mm_drt-17+w-99'
                         '/Unfolded/data.nii.gz')
difff2=nib.load('/home/uzair/PycharmProjects/Unfolding/data/diffusionSimulations_res-6250mm_drt-17+w-99'
                         '/Unfolded/data.nii.gz')


diff1=difff1.get_fdata()
diff2=difff2.get_fdata()

S1=[]
S2=[]
for b in range(0, diff2.shape[3]):
    print(b)
    points,S=coordinates.getPointsData(difff1,b)
    interp=LinearNDInterpolator(points, S)
    for i in range(0,diff2.shape[0]):
        for j in range(0, diff2.shape[1]):
            for k in range(0, diff2.shape[2]):
                S2.append(diff2[i,j,k,b])
                ind=np.asarray([i,j,k])
                point=coordinates.toWorld(difff1,[ind])
                S1.append(interp(point[0]))

