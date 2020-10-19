import coordinates
import matplotlib.pyplot as plt
import numpy as np
import nibabel as nib

coords=coordinates.coordinates()

coords.loadCoordinates(path="K:\\Datasets\\sampleNiftiCoordinates\\",prefix="")

coords.initialize()
coords.meanArcLength()

nib.save(coords.X_uvw_nii,"X_uvw.nii.gz")


plt.imshow(coords.X_uvw_nii.get_data()[:,:,3])