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

    T = linalg.inv(dot(K, -R)).dot(T.reshape((3, 1)))
    K /= -K[2,2]

    return K, R, T

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

    alpha_x = f * height *  0.5
    alpha_y = f * height * 0.5

    K = array([
        [alpha_x, 0, 0],
        [0, alpha_y, 0],
        [0, 0, -1]])

    return (dot(K, M), (width, height))

def rank3decomp(Q):
    U, S, V = svd(Q)

    S[3] = 0.0
    S = diag(sqrt(S))

    return dot(U, S)[:,:3], dot(S, V)[:3,:]

def factor_measurement_matrix(MM):
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
    target = copy(target)
    source = copy(source)

    target /= target[3,:]
    source /= source[3,:]

    return dot(linalg.pinv(source.T), target.T).T

def resolve_camera_ambiguity(W, P, X):
    sign = 0
    X /= X[3,:]

    for i in range(W.shape[0] // 3):
        if not isnan(W[i * 3 + 0, 0]):
            p = dot(P[i * 3 + 0:i * 3 + 3,:], X[:,0])
            if p[2] < 0:
                sign -= 1
            else:
                sign += 1

    sign /= abs(sign)

    return P * sign, X
