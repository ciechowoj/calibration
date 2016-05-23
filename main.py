from numpy import *
from numpy.linalg import *
import scipy
import scipy.linalg
from plot import *
from test import *

points = make_test_points()
cameras = camera_test_set1()

def camera_matrix(cameras):
    return vstack([camera[0] for camera in cameras])

P = camera_matrix(cameras)

MM = dot(P, points)


def recover_cameras_and_points(MM, dist = NaN):
    pass










# points = dot(camera, points.T).T
# print(points)


plot_scene(points, cameras)
plot_view(points, cameras[10], 800, 600)

# plt.show()