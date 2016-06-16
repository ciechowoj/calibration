from numpy import *
from numpy.linalg import *
from numpy.random import *
import scipy
import scipy.linalg
from plot import *
from test import *
import sys
import itertools

# Setup test data.

X = make_test_points2()[:, :32]

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
W = remove_point_locations0(W)

# Recover structure and motion.

def rank_four_column_space(W):
    import random

    indices = arange(W.shape[1])

    N = []
    num = 0

    for i in range(W.shape[1] ** 2):
        At = W[:, choice(indices, 4, True)]

        # print_matrix(At)

        masks = [logical_not(isnan(At[:, i])) for i in range(4)]
        mask = logical_and(
            logical_and(masks[0], masks[1]),
            logical_and(masks[2], masks[3]))

        u, s, v = svd(At[mask,:])

        if (s.shape[0] < 4 or s[3] < 10e-10):
            continue

        Bt = []

        standard = set()

        for j in range(At.shape[1]):
            for k in range(At.shape[0] // 3):
                if isnan(At[k * 3 + 0, j]):
                    standard.add(k)
                    At[k * 3 + 0, j] = 0.0
                    At[k * 3 + 1, j] = 0.0
                    At[k * 3 + 2, j] = 0.0
                elif isnan(At[k * 3 + 2, j]):
                    a = zeros(At.shape[0])
                    a[k * 3 + 0] = At[k * 3 + 0, j]
                    a[k * 3 + 1] = At[k * 3 + 1, j]
                    a[k * 3 + 2] = 1.0
                    At[k * 3 + 0, j] = 0.0
                    At[k * 3 + 1, j] = 0.0
                    At[k * 3 + 2, j] = 0.0

            Bt.append(At[:, j])

        for k in sorted(list(standard)):
            a = zeros(At.shape[0])
            b = zeros(At.shape[0])
            c = zeros(At.shape[0])
            a[k * 3 + 0] = 1
            b[k * 3 + 1] = 1
            c[k * 3 + 2] = 1
            Bt.append(a)
            Bt.append(b)
            Bt.append(c)

        Bt = vstack(Bt).T
        u, s, v = svd(Bt)

        N.append(u[:, 4:])

        num += 1

        if num == W.shape[1]:
            break

    N = hstack(N)

    u, s, v = svd(N)

    return u[:, u.shape[1] - 4:u.shape[1]]

def reconstruct_columns(W, B):
    W = W.copy()

    for i in range(W.shape[1]):
        C = W[:, i]






def reconstruct_missing_data(W, threshold = 8):
    m, n = W.shape
    m = m // 3
    rM, cM, rI, cI = [], [], [], []
    W = W.copy()

    while m != len(rI) and n != len(cI):
        # Find a row with the greatest number of filled elements.
        best = (0, -1)

        for j in range(m):
            mask = logical_not(isnan(W[j * 3]))
            num = sum(mask)

            if best[0] < num:
                best = (num, j)
                cM = mask

        cI = arange(n)[cM]
        best = best[1]

        # Collect rows that have at least 8 common elements with the best row.
        rI = [(-1, best)]
        M = isnan(W[best * 3])

        for j in (i for i in range(m) if i != best):
            num = sum(logical_not(logical_or(isnan(W[j * 3]), M)))

            if num >= threshold:
                rI.append((num, j))

        rI = array([x[1] for x in sorted(rI, key = lambda x: x[1])])

        # Recover projective depths in collected rows with respect to the best row.
        W[best * 3 + 2, cI] = ones(cI.shape[0])

        for i in (i for i in rI if i != best):
            mask = logical_and(logical_not(isnan(W[i * 3])), cM)
            R = W[i * 3: (i + 1) * 3, mask]
            B = W[best * 3: (best + 1) * 3, mask]
            W[i * 3: (i + 1) * 3, mask] = recover_projective_depths(B, R)

        # Find column space.
        rI = array(sorted(hstack([rI * 3 + 0, rI * 3 + 1, rI * 3 + 2])))
        B = rank_four_column_space(W[rI[:, newaxis], cI])

        # Fit columns.
        W[rI[:, newaxis], cI] = reconstruct_columns(W[rI[:, newaxis], cI], B)


        break



    return W

W = reconstruct_missing_data(W)

print_matrix(W)

exit(0)

# W = recover_projective_depths(W)
# print_matrix(W)


# print_matrix(W)


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
