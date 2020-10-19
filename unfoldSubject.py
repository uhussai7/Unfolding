import diffusion
import coordinates

class unfoldSubject:
    def __init__(self):
        self.coords= coordinates.coordinates()
        self.T1= []
        self.diffusion = diffusion.diffVolume()

    def loadDiffusion(self,path=None):
        self.diffusion.getVolume(folder=path)
        self.diffusion.shells()


