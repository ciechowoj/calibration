from numpy import *
from numpy.linalg import *
from numpy.random import *
import scipy
import scipy.linalg
from plot import *
from test import *
import sys


# Setup test data.

X = make_test_points2()[:, :64]

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

def load_measurements(path):
    file = open(path)
    lines = [line for line in file.readlines() if len(line) > 10 and not line.startswith('#')]

    num_points = len(lines)
    num_cameras = (len(lines[0].split(";")) - 2) // 2

    W = empty((num_cameras * 3, num_points))

    for i, line in enumerate(lines):
        fields = list(map(lambda x: float(x.strip()), line.split(";")))[2:]

        for j in range(num_cameras):
            x, y = fields[j * 2 + 0], fields[j * 2 + 1]

            if isnan(x) or isnan(y):
                W[j * 3 + 0, i] = NaN
                W[j * 3 + 1, i] = NaN
                W[j * 3 + 2, i] = NaN
            else:
                W[j * 3 + 0, i] = x
                W[j * 3 + 1, i] = y
                W[j * 3 + 2, i] = 1.0

    return W

def load_truth(path, scale = 1.0):
    file = open(path)
    lines = [line for line in file.readlines() if len(line) > 10 and not line.startswith('#')]

    num_points = len(lines)

    X = ones((4, num_points))

    for i, line in enumerate(lines):
        fields = list(map(lambda x: float(x.strip()), line.split(";")))

        X[0, i] = fields[0] * scale
        X[1, i] = fields[1] * scale
        X[2, i] = fields[2] * scale

    return X

W = load_measurements("data/dataset3.csv")
X = load_truth("data/truth.csv", 4.0)
resolutions = [(640, 480), (640, 480), (1920, 1088), (1280, 720), (1280, 720)]

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


