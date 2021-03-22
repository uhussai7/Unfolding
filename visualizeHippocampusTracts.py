import nibabel as nib
from dipy.io.streamline import load_vtk_streamlines
from dipy.viz import colormap
from dipy.viz import window, actor, has_fury
import numpy as np
import open3d as o3d
import pyvista as pv
from coordinates import toWorld



#subs_strings = ['sub', 'CA1', 'CA2', 'CA3', 'CA4']
subs_strings = ['CA1CA3e']
s=0
i=1

native_streamlines="native_streamline_%s_bin-%d.vtk" % (subs_strings[s],i)
unfold_streamlines= "unfold_streamlines_%s_bin-%d.vtk" % (subs_strings[s], i)
from_unfold_streamlines = "from_unfold_streamlines_%s_bin-%d.vtk" % (subs_strings[s], i)
roi_nii=nib.load('data/oldUnfold/U.nii.gz')
roi = roi_nii.get_fdata()

nlines=load_vtk_streamlines('data/oldUnfold/'+native_streamlines)
ulines=load_vtk_streamlines('data/oldUnfold/'+from_unfold_streamlines)
#unfoldlines=load_vtk_streamlines('data/oldUnfold/Unfolded/'+unfold_streamlines)

surface_opacity = 0.05
surface_color = [0, 1, 1]

roi_actor = actor.contour_from_roi(roi, roi_nii.affine,
                                       surface_color, surface_opacity)


def plot_streamlines(streamlines,roi_actor):
    if has_fury:
        # Prepare the display objects.
        color = colormap.line_colors(streamlines)

        streamlines_actor = actor.line(streamlines,
                                       colormap.line_colors(streamlines))

        # Create the 3D display.
        scene = window.Scene()
        scene.add(roi_actor)
        scene.add(streamlines_actor)


        # Save still images for this static example. Or for interactivity use
        window.show(scene)
#
