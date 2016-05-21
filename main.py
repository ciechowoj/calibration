import matplotlib.pyplot as pyplot
import mpl_toolkits.mplot3d
from numpy import *
from numpy.linalg import *
import scipy
import scipy.linalg

def plot_points(axes3d, points):
    for point in points:
        axes3d.scatter(
            point[0],
            point[2],
            point[1])

def plot_camera(axes3d, camera):
    def quiver(axes3d, p, d, **kwargs):
        axes3d.quiver(
            p[0], p[2], p[1],
            d[0], d[2], d[1],
            pivot = 'tail',
            length = norm(d),
            **kwargs)

    K, R, T = decompose(camera)

    quiver(axes3d, T, R[0] * 2, colors=[(1, 0, 0, 1)])
    quiver(axes3d, T, R[1] * 2, colors=[(0, 1, 0, 1)])
    quiver(axes3d, T, R[2] * 2, colors=[(0, 0, 1, 1)])

def make_axes3d():
    figure = pyplot.figure(figsize=(10, 6.5))
    axes3d = figure.add_subplot(111, projection="3d")
    axes3d.autoscale(False)
    axes3d.set_xlabel('X')
    axes3d.set_ylabel('Z')
    axes3d.set_zlabel('Y')
    axes3d.set_xlim3d(-12, 12)
    axes3d.set_ylim3d(12, -12)
    axes3d.set_zlim3d(-10, 10)
    return axes3d

def normalize(v):
    return v / norm(v)

def decompose(P):
    M = P[:3, :3]
    T = P[:3, 3]

    K, R = scipy.linalg.rq(M)

    for i in range(3):
        if K[i,i] < 0:
            K[:, i] *= -1
            R[i, :] *= -1

    return K, R, -linalg.inv(K * R).dot(T.reshape((3, 1)))

def make_test_points(num, center = array([0, 0, 0]), size = 1):
    result = (random.rand(3, num).T - array([0.5] * 3)) * size + center
    result = hstack((result, ones((num, 1))))

    return result

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

    alpha_x = f * (width / height) * width *  0.5
    alpha_y = f * height * 0.5

    K = array([
        [alpha_x, 0, 0],
        [0, alpha_y, 0],
        [0, 0, 1]])

    return dot(K, M)

points = make_test_points(1, center = [0, 0, 0], size = [0, 0, 0])
# print(points)

camera = make_camera([0, 0, 5], [0, -2, 0], [0, 1, 0], pi / 2, 800, 600)


# points = dot(camera, points.T).T
# print(points)


axes3d = make_axes3d()
plot_points(axes3d, points)
plot_camera(axes3d, camera)
pyplot.savefig("myplot.png")

# plt.show()