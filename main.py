from numpy import *
from numpy.linalg import *
import scipy
import scipy.linalg
from plot import *
from test import *

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

#print_matrix(W)
W = remove_projective_depths(W)

F = fundamental_matrix(W[0:3], W[3:6])

def right_epipole(F):
    U, S, V = svd(F)
    return V[-1,:]

e = right_epipole(F)

def recover_projective_depths(W):

    for i in range(1, W.shape[0] // 3):
        j = 0
        F = fundamental_matrix(W[i * 3:i * 3 + 3], W[j * 3:j * 3 + 3])
        e = right_epipole(F.T)

        for k in range(W.shape[1]):
            qi = W[i * 3:i * 3 + 3, k]
            qj = W[j * 3:j * 3 + 3, k]

            eq = cross(e, qi)

            W[i * 3 + 2, k] = (dot(eq, dot(F, qj)) / (norm(eq) ** 2)) * W[j * 3 + 2, k]

        W[i * 3: i * 3 + 2] *= W[i * 3 + 2]

    return W


W = recover_projective_depths(W)
#print_matrix(W)


hP, hX = factor_measurement_matrix(W)

F = find_frame_of_reference(X[:,:5], hX[:,:5])
hP = dot(hP, linalg.pinv(F))
hX = dot(F, hX)

hP, hX = resolve_camera_ambiguity(W, hP, hX)

hCameras = separate_cameras(hP, resolutions)

plot_scene(hX, hCameras)
plot_scene(X, cameras, "scene2.png")
id = 9
#plot_view(hX, hCameras[id])
#plot_view(X, cameras[id], "view2.png")
