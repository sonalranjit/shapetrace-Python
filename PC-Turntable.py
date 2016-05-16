""" Demonstrates use of visual.Markers to create a point cloud with a
standard turntable camera to fly around with and a centered 3D Axis.
"""
import numpy as np
import vispy.plot
import vispy.scene
from vispy.scene import visuals

#
# Make a canvas and add simple view
#
canvas = vispy.scene.SceneCanvas(keys='interactive', show=True)
view = canvas.central_widget.add_view()


# generate data
centers = np.random.normal(size=(50, 3))
pos = np.random.normal(size=(100000, 3), scale=0.2)
indexes = np.random.normal(size=100000, loc=centers.shape[0]/2.,
                           scale=centers.shape[0]/3.)
indexes = np.clip(indexes, 0, centers.shape[0]-1).astype(int)
scales = 10**(np.linspace(-2, 0.5, centers.shape[0]))[indexes][:, np.newaxis]
pos *= scales
pos += centers[indexes]

# create scatter object and fill in the data
scatter = visuals.Markers()
scatter.set_data(pos, edge_color=None, face_color='white', size=5)
view.add(scatter)

view.camera = 'turntable'

axis = visuals.XYZAxis(parent=view.scene)

if __name__ == '__main__':
    import sys
    if sys.flags.interactive != 1:
        vispy.app.run()