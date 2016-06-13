from numpy import *
from numpy.linalg import *
import scipy
import scipy.linalg
from plot import *
from test import *

# Setup test data.

X = make_test_points2()

cameras = camera_test_set1()[0:10]
resolutions = [camera[1] for camera in cameras]

def camera_matrix(cameras):
    return vstack([camera[0] for camera in cameras])

def separate_cameras(P, resolutions):
    return list(zip(
        [P[i * 3 + 0:i * 3 + 3, :] for i in range(P.shape[0] // 3)],
        resolutions))

P = camera_matrix(cameras)
W = dot(P, X)
W = remove_projective_depths(W)

# Recover structure and motion.

W = recover_projective_depths(W)

balance_measurement_matrix(W)

# print_matrix(W)


hP, hX = factor_measurement_matrix(W)

F = find_frame_of_reference(X[:,:5], hX[:,:5])
hP = dot(hP, linalg.pinv(F))
hX = dot(F, hX)

hP, hX = resolve_camera_ambiguity(W, hP, hX)

hCameras = separate_cameras(hP, resolutions)

plot_scene(hX, hCameras)
plot_scene(X, cameras, "scene2.png")
id = 3
plot_view(hX, hCameras[id])
plot_view(X, cameras[id], "view2.png")
