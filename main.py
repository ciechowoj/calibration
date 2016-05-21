from numpy import *
from numpy.linalg import *
import scipy
import scipy.linalg
from plot import *

def make_cube(num, center = array([0, 0, 0]), size = 5):
    num = num // 6

    size = array(size)

    points =                 make_test_points(num, center = size * [0, -0.5, 0], size = size * [1, 0, 1])
    points = hstack((points, make_test_points(num, center = size * [0, 0.5, 0], size = size * [1, 0, 1])))
    points = hstack((points, make_test_points(num, center = size * [-0.5, 0.0, 0], size = size * [0, 1, 1])))
    points = hstack((points, make_test_points(num, center = size * [0.5, 0.0, 0], size = size * [0, 1, 1])))
    points = hstack((points, make_test_points(num, center = size * [0, 0.0, -0.5], size = size * [1, 1, 0])))
    points = hstack((points, make_test_points(num, center = size * [0, 0.0, 0.5], size = size * [1, 1, 0])))

    return (points.T + append(center, 0)).T

def make_test_points(num, center = array([0, 0, 0]), size = 1):
    result = (random.rand(3, num).T - array([0.5] * 3)) * size + center
    result = hstack((result, ones((num, 1))))

    return result.T

points = make_cube(200, [10, 10, 0])

# print(points.T)

camera = make_camera([0, 0, 20], [0, 0, 0], [0, 1, 0], pi / 2, 800, 600)


# points = dot(camera, points.T).T
# print(points)


plot_scene(points, [camera])
plot_view(points, camera, 800, 600)

# plt.show()