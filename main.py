from numpy import *
from numpy.linalg import *
import scipy
import scipy.linalg
from plot import *
from test import *

pointsX = sample_cube(200, center = [0,2,0], size=[10, 1, 1]) # make_test_points()[:,:20]
pointsY = sample_cube(400, center = [0,2,0], size=[1, 20, 1]) # make_test_points()[:,:20]
pointsZ = sample_cube(200, center = [0,2,0], size=[1, 1, 10]) # make_test_points()[:,:20]

points = hstack((pointsX, pointsY, pointsZ))

points = sample_sphere(500, [-5, 0, -5], 5)

# points = make_test_points()

cameras = camera_test_set1()[:10]
resolutions = [camera[1] for camera in cameras]

def camera_matrix(cameras):
    return vstack([camera[0] for camera in cameras])

def separate_cameras(P, resolutions):
    return list(zip(
        [P[i * 3 + 0:i * 3 + 3, :] for i in range(P.shape[0] // 3)],
        resolutions))

P = camera_matrix(cameras)
MM = dot(P, points)

def recover_cameras_and_points(MM):
    m, n = MM.shape
    m = m // 3

    P, S, X = svd(MM)
    S = diag(sqrt(S)[:4])

    P = dot(P[:, :4], S)
    X = dot(S, X[:4, :])

    Ts = sum(MM, axis = 1)

    C = empty((2 * m, 4))

    for i in range(m):
        Tx = Ts[i * 3 + 0]
        Ty = Ts[i * 3 + 1]
        Tz = Ts[i * 3 + 2]

        C[i * 2 + 0][0] = P[i * 3 + 2][0] * Tx - P[i * 3 + 0][0] * Tz
        C[i * 2 + 0][1] = P[i * 3 + 2][1] * Tx - P[i * 3 + 0][1] * Tz
        C[i * 2 + 0][2] = P[i * 3 + 2][2] * Tx - P[i * 3 + 0][2] * Tz
        C[i * 2 + 0][3] = P[i * 3 + 2][3] * Tx - P[i * 3 + 0][3] * Tz
        C[i * 2 + 1][0] = P[i * 3 + 2][0] * Ty - P[i * 3 + 1][0] * Tz
        C[i * 2 + 1][1] = P[i * 3 + 2][1] * Ty - P[i * 3 + 1][1] * Tz
        C[i * 2 + 1][2] = P[i * 3 + 2][2] * Ty - P[i * 3 + 1][2] * Tz
        C[i * 2 + 1][3] = P[i * 3 + 2][3] * Ty - P[i * 3 + 1][3] * Tz

    b = svd(C)[2][-1,:].reshape(4, 1)

    C = empty((5 * m, 10))
    D = zeros((5 * m,))

    x = 0
    y = 1
    z = 2

    for i in range(m):
        i3 = i * 3

        C[i * 5 + 0][0] =      P[i3 + x][0] * P[i3 + x][0] - P[i3 + y][0] * P[i3 + y][0]
        C[i * 5 + 0][1] = 2 * (P[i3 + x][0] * P[i3 + x][1] - P[i3 + y][0] * P[i3 + y][1])
        C[i * 5 + 0][2] = 2 * (P[i3 + x][0] * P[i3 + x][2] - P[i3 + y][0] * P[i3 + y][2])
        C[i * 5 + 0][3] = 2 * (P[i3 + x][0] * P[i3 + x][3] - P[i3 + y][0] * P[i3 + y][3])
        C[i * 5 + 0][4] =      P[i3 + x][1] * P[i3 + x][1] - P[i3 + y][1] * P[i3 + y][1]
        C[i * 5 + 0][5] = 2 * (P[i3 + x][1] * P[i3 + x][2] - P[i3 + y][1] * P[i3 + y][2])
        C[i * 5 + 0][6] = 2 * (P[i3 + x][1] * P[i3 + x][3] - P[i3 + y][1] * P[i3 + y][3])
        C[i * 5 + 0][7] =      P[i3 + x][2] * P[i3 + x][2] - P[i3 + y][2] * P[i3 + y][2]
        C[i * 5 + 0][8] = 2 * (P[i3 + x][2] * P[i3 + x][3] - P[i3 + y][2] * P[i3 + y][3])
        C[i * 5 + 0][9] =      P[i3 + x][3] * P[i3 + x][3] - P[i3 + y][3] * P[i3 + y][3]

        C[i * 5 + 1][0] = P[i3 + x][0] * P[i3 + y][0]
        C[i * 5 + 1][1] = P[i3 + x][0] * P[i3 + y][1] + P[i3 + x][1] * P[i3 + y][0]
        C[i * 5 + 1][2] = P[i3 + x][0] * P[i3 + y][2] + P[i3 + x][2] * P[i3 + y][0]
        C[i * 5 + 1][3] = P[i3 + x][0] * P[i3 + y][3] + P[i3 + x][3] * P[i3 + y][0]
        C[i * 5 + 1][4] = P[i3 + x][1] * P[i3 + y][1]
        C[i * 5 + 1][5] = P[i3 + x][1] * P[i3 + y][2] + P[i3 + x][2] * P[i3 + y][1]
        C[i * 5 + 1][6] = P[i3 + x][1] * P[i3 + y][3] + P[i3 + x][3] * P[i3 + y][1]
        C[i * 5 + 1][7] = P[i3 + x][2] * P[i3 + y][2]
        C[i * 5 + 1][8] = P[i3 + x][2] * P[i3 + y][3] + P[i3 + x][3] * P[i3 + y][2]
        C[i * 5 + 1][9] = P[i3 + x][3] * P[i3 + y][3]

        C[i * 5 + 2][0] = P[i3 + y][0] * P[i3 + z][0]
        C[i * 5 + 2][1] = P[i3 + y][0] * P[i3 + z][1] + P[i3 + y][1] * P[i3 + z][0]
        C[i * 5 + 2][2] = P[i3 + y][0] * P[i3 + z][2] + P[i3 + y][2] * P[i3 + z][0]
        C[i * 5 + 2][3] = P[i3 + y][0] * P[i3 + z][3] + P[i3 + y][3] * P[i3 + z][0]
        C[i * 5 + 2][4] = P[i3 + y][1] * P[i3 + z][1]
        C[i * 5 + 2][5] = P[i3 + y][1] * P[i3 + z][2] + P[i3 + y][2] * P[i3 + z][1]
        C[i * 5 + 2][6] = P[i3 + y][1] * P[i3 + z][3] + P[i3 + y][3] * P[i3 + z][1]
        C[i * 5 + 2][7] = P[i3 + y][2] * P[i3 + z][2]
        C[i * 5 + 2][8] = P[i3 + y][2] * P[i3 + z][3] + P[i3 + y][3] * P[i3 + z][2]
        C[i * 5 + 2][9] = P[i3 + y][3] * P[i3 + z][3]

        C[i * 5 + 3][0] = P[i3 + z][0] * P[i3 + x][0]
        C[i * 5 + 3][1] = P[i3 + z][0] * P[i3 + x][1] + P[i3 + z][1] * P[i3 + x][0]
        C[i * 5 + 3][2] = P[i3 + z][0] * P[i3 + x][2] + P[i3 + z][2] * P[i3 + x][0]
        C[i * 5 + 3][3] = P[i3 + z][0] * P[i3 + x][3] + P[i3 + z][3] * P[i3 + x][0]
        C[i * 5 + 3][4] = P[i3 + z][1] * P[i3 + x][1]
        C[i * 5 + 3][5] = P[i3 + z][1] * P[i3 + x][2] + P[i3 + z][2] * P[i3 + x][1]
        C[i * 5 + 3][6] = P[i3 + z][1] * P[i3 + x][3] + P[i3 + z][3] * P[i3 + x][1]
        C[i * 5 + 3][7] = P[i3 + z][2] * P[i3 + x][2]
        C[i * 5 + 3][8] = P[i3 + z][2] * P[i3 + x][3] + P[i3 + z][3] * P[i3 + x][2]
        C[i * 5 + 3][9] = P[i3 + z][3] * P[i3 + x][3]

        C[i * 5 + 4][0] =     (P[i3 + z][0] * P[i3 + z][0]);
        C[i * 5 + 4][1] = 2 * (P[i3 + z][0] * P[i3 + z][1]);
        C[i * 5 + 4][2] = 2 * (P[i3 + z][0] * P[i3 + z][2]);
        C[i * 5 + 4][3] = 2 * (P[i3 + z][0] * P[i3 + z][3]);
        C[i * 5 + 4][4] =     (P[i3 + z][1] * P[i3 + z][1]);
        C[i * 5 + 4][5] = 2 * (P[i3 + z][1] * P[i3 + z][2]);
        C[i * 5 + 4][6] = 2 * (P[i3 + z][1] * P[i3 + z][3]);
        C[i * 5 + 4][7] =     (P[i3 + z][2] * P[i3 + z][2]);
        C[i * 5 + 4][8] = 2 * (P[i3 + z][2] * P[i3 + z][3]);
        C[i * 5 + 4][9] =     (P[i3 + z][3] * P[i3 + z][3]);

        D[i * 5 + 4] = 1

    tQ = dot(linalg.pinv(C), D.reshape((C.shape[0], 1))) # svd(C)[2][-1,:]

    Q = empty((4, 4))

    Q[0,0] = tQ[0]; Q[0,1] = tQ[1]; Q[0,2] = tQ[2]; Q[0,3] = tQ[3];
    Q[1,0] = tQ[1]; Q[1,1] = tQ[4]; Q[1,2] = tQ[5]; Q[1,3] = tQ[6];
    Q[2,0] = tQ[2]; Q[2,1] = tQ[5]; Q[2,2] = tQ[7]; Q[2,3] = tQ[8];
    Q[3,0] = tQ[3]; Q[3,1] = tQ[6]; Q[3,2] = tQ[8]; Q[3,3] = tQ[9];

    A = rank3decomp(Q)[0]

    H = hstack((A, b))

    hP = dot(P, H)
    hX = dot(inv(H), X)

    return hP, hX

def find_frame_of_reference(target, source):
    M = dot(linalg.pinv(source.T), target.T)
    return M


hP, hX = recover_cameras_and_points(MM)


F = find_frame_of_reference(points[:, :4], hX[:,:4])


hCameras = separate_cameras(dot(hP, linalg.pinv(F.T)), resolutions)


# plot_scene(dot(F.T, hX), hCameras)
plot_scene(points, cameras)


# points = dot(camera, points.T).T
# print(points)

id = 9
plot_view(points, cameras[id])
plot_view(hX, hCameras[id]) # , "view2.png")

# plt.show()