from numpy import *
from numpy.linalg import *
from numpy.random import *
import scipy
import scipy.linalg
from plot import *
from test import *
import sys


# Setup test data.

X = make_test_points2()[:, :200]

print(X.shape)

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




# W = remove_point_locations0(W)
W = remove_point_locations(W, 0.0)







# W = W[:15, :]

# Recover structure and motion.

# W = load_measurements("data/dataset1.csv")
# X = load_truth("data/truth.csv", 1.0)
# resolutions = [(640, 480), (640, 480), (1920, 1088), (1280, 720), (1280, 720)]

W = remove_invalid_locations(W)

print_matrix(W)

W = reconstruct_missing_data(W)

print_matrix(W)

W = recover_all_projective_depths(W)
balance_measurement_matrix(W)

hP, hX = factor_measurement_matrix(W)

F = find_frame_of_reference(X, hX)
hP = dot(hP, linalg.pinv(F))
hX = dot(F, hX)

hP, hX = resolve_camera_ambiguity(W, hP, hX)

hCameras = separate_cameras(hP, resolutions)

plot_scene(hX, hCameras)

id = 3
plot_view(hX, hCameras[id])
# plot_view(X, cameras[id], "view2.png")


