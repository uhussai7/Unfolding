from dipy.data import default_sphere
from dipy.tracking.local_tracking import LocalTracking
import numpy as np
from scipy.interpolate import griddata
from dipy.viz import window, actor, has_fury, colormap
from dipy.tracking.streamline import Streamlines


def pointsPerLine(streamlines):
    N_lines=len(streamlines)
    ppline=np.zeros(N_lines)
    for l in range(0,N_lines):
        ppline[l]=len(streamlines[l])
    return ppline.astype(int)

def allLines2Lines(allLines,pointsPerLine):
    streamlines=[]
    first=0
    for i in range(0,len(pointsPerLine)):
        templine=[]
        for p in range(0,pointsPerLine[i]):
            point=allLines[first+i]
            if( np.isnan(np.sum(point))==0):
                templine.append(point)
        if(len(templine)>1):
            streamlines.append(templine)
        first=pointsPerLine[i]
    return streamlines

class tracking:
    def __init__(self,peaks,stopping_criterion,seeds,affine,
                 graddev=None,sphere=default_sphere.subdivide(),seed_density=1):
        self.peaks=peaks
        self.stopping_criterion=stopping_criterion
        self.seeds=seeds
        self.affine=affine
        self.graddev=graddev
        self.sphere=sphere
        self.streamlines=[]
        self.NpointsPerLine=[]
        self.seed_density=seed_density

        #adding some key direction to the sphere
        self.sphere.vertices = np.append(self.sphere.vertices, [[1, 0, 0]])
        self.sphere.vertices = np.append(self.sphere.vertices, [[-1, 0, 0]])
        self.sphere.vertices = np.append(self.sphere.vertices, [[0, 1, 0]])
        self.sphere.vertices = np.append(self.sphere.vertices, [[0, -1, 0]])
        self.sphere.vertices = self.sphere.vertices.reshape([-1, 3])

    def localTracking(self):
        if self.graddev is None:
            streamlines_generator=LocalTracking(self.peaks,
                                           self.stopping_criterion,
                                           self.seeds,
                                           self.affine,
                                           step_size=self.affine[0,0]/10)
            self.streamlines=Streamlines(streamlines_generator)


        else:
            shape=self.graddev.shape
            self.graddev=self.graddev.reshape(shape[0:3]+ (3, 3), order='F')
            self.graddev=self.graddev.reshape([-1,3,3])+np.eye(3)
            self.graddev=self.graddev.reshape(shape[0:3]+(3,3))

            #multiply by the jacobian
            new_peak_dirs = np.einsum('ijkab,ijkvb->ijkva',
                                      self.graddev, self.peaks.peak_dirs)
            new_peak_dirs = new_peak_dirs.reshape([-1, self.peaks.peak_indices.shape[-1], 3])
            #update self.peaks.peak_indices
            peak_indices=np.zeros(self.peaks.peak_indices.shape)
            peak_indices=peak_indices.reshape([-1,self.peaks.peak_indices.shape[-1]])

            for i in range(0, peak_indices.shape[0]):
                for k in range(0, self.peaks.peak_indices.shape[-1]):
                    peak_indices[i, k] = self.sphere.find_closest(new_peak_dirs[i, k, :])

            self.peaks.peak_indices = peak_indices.reshape(self.peaks.peak_indices.shape)

            streamlines_generator= LocalTracking(self.peaks,
                                             self.stopping_criterion,
                                             self.seeds,
                                             self.affine,
                                             step_size=self.affine[0, 0] / 4)

            self.streamlines=Streamlines(streamlines_generator)
            self.NpointsPerLine=pointsPerLine(self.streamlines)


    def plot(self):
        if has_fury:
            # Prepare the display objects.
            color = colormap.line_colors(self.streamlines)

            streamlines_actor = actor.line(self.streamlines,
                                           colormap.line_colors(self.streamlines))

            # Create the 3D display.
            scene = window.Scene()
            scene.add(streamlines_actor)

            # Save still images for this static example. Or for interactivity use
            window.show(scene)


class unfoldStreamlines:
    def __init__(self,nativeStreamlines,unfoldStreamlines,nppl,uppl,coords):
        self.nativeStreamlines=nativeStreamlines
        self.unfoldStreamlines=unfoldStreamlines
        self.coords=coords
        self.streamlinesFromUnfold=[]
        self.streamlinesFromNative=[]
        self.nppl=nppl
        self.uppl=uppl

    def moveStreamlines2Native(self):
        #assuming that the coordinates are in world coordinates.

        points=np.asarray([self.coords.Ua,
                           self.coords.Va,
                           self.coords.Wa]).transpose()

        allLines=self.unfoldStreamlines.get_data()
        x = griddata(points, self.coords.X, allLines)
        y = griddata(points, self.coords.Y, allLines)
        z = griddata(points, self.coords.Z, allLines)

        allLines=np.asarray([x,y,z]).T

        self.streamlinesFromUnfold=allLines2Lines(allLines,
                                                  self.uppl)


    def moveStreamlines2Unfold(self):

        points = np.asarray([self.coords.X,
                             self.coords.Y,
                             self.coords.Z]).transpose()

        allLines = self.nativeStreamlines.get_data()
        ua = griddata(points, self.coords.Ua, allLines)
        va = griddata(points, self.coords.Va, allLines)
        wa = griddata(points, self.coords.Wa, allLines)

        allLines = np.asarray([ua, va, wa]).T

        self.streamlinesFromNative = allLines2Lines(allLines,
                                                    self.nppl)

