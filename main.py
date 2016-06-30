#!/usr/bin/python3

from numpy import *
from numpy.linalg import *
from numpy.random import *
from plot import *
from test import *
import sys
import os.path

def separate_cameras(P, resolutions):
    return list(zip(
        [P[i * 3 + 0:i * 3 + 3, :] for i in range(P.shape[0] // 3)],
        resolutions))

def print_help():
    help = """\
Final project for DIGWOR. Generates files scene.png and view.png which presents recovered scene. Prints recovered intrinsic end extrinsic parameters of cameras.

Usage:
    {0} <input_data.csv> <ground_truth.csv> [<camera_id>]
"""

    print(help.format(os.path.basename(sys.argv[0])))


create_datasets()

# Recover structure and motion.

if len(sys.argv) < 3:
    print_help()
    exit(1)

try:
    if len(sys.argv) > 3:
        int(sys.argv[3])
except:
    print_help()
    exit(1)

W, resolutions = load_measurements(sys.argv[1])
X = load_truth(sys.argv[2], 1.0)

W = remove_invalid_locations(W)

print("Initial measurement matrix")
print_matrix(W)

W = reconstruct_missing_data(W)

print("Final measurement matrix")
print_matrix(W)

W = recover_all_projective_depths(W)

balance_measurement_matrix(W)

hP, hX = factor_measurement_matrix(W)

F = find_frame_of_reference(X, hX)
hP = dot(hP, linalg.pinv(F))
hX = dot(F, hX)

hP, hX = resolve_camera_ambiguity(W, hP, hX)

hCameras = separate_cameras(hP, resolutions)

plot_scene(hX[:, :-89], hCameras)

if len(sys.argv) > 3:
    plot_view(hX[:, :-140], hCameras[int(sys.argv[3])])

