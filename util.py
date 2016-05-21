from numpy import *
from numpy.linalg import *
import scipy
import scipy.linalg

def normalize(v):
    return v / norm(v)

def decompose(P):
    M = P[:3, :3]
    T = P[:3, 3]

    K, R = scipy.linalg.rq(M)

    for i in range(2):
        if K[i,i] < 0:
            K[:, i] *= -1
            R[i, :] *= -1

    if K[2,2] > 0:
        K[:, 2] *= -1
        R[2, :] *= -1

    if det(R) < 0:
        R *= -1

    return K, R, -linalg.inv(dot(K, R)).dot(T.reshape((3, 1)))

def project(points, camera):
    points2D = dot(camera, points)
    points2D = points2D[:,points2D[2] > 0]
    points2D /= points2D[2]

    return points2D

def look_at(eye, center, up):
    Z = normalize(eye - array(center))
    X = normalize(cross(up, Z))
    Y = normalize(cross(Z, X))

    R = array([
        concatenate((X, [0])),
        concatenate((Y, [0])),
        concatenate((Z, [0])),
        [0, 0, 0, 1]])

    T = array([
        [1, 0, 0, -eye[0]],
        [0, 1, 0, -eye[1]],
        [0, 0, 1, -eye[2]],
        [0, 0, 0, 1]])

    return dot(R, T)[:3,:]

def make_camera(eye, center, up, fovy, width, height):
    M = look_at(eye, center, up)

    f = tan(fovy * 0.5) ** -1

    alpha_x = f * width *  0.5
    alpha_y = f * height * 0.5

    K = array([
        [alpha_x, 0, 0],
        [0, alpha_y, 0],
        [0, 0, -1]])

    return dot(K, M)


