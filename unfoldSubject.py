import scipy
import dipy

class unfoldSubject:
    def __init__(self):
        self.U_nii = []
        self.V_nii = []
        self.W_nii = []
        self.T1= []
        self.diffusion = []