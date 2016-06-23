import numpy
import cv2
import time

capture = cv2.VideoCapture(1)

square_size = 238
pattern_size = (9, 6)
pattern_points = numpy.zeros((numpy.prod(pattern_size), 3), numpy.float32)
pattern_points[:, :2] = numpy.indices(pattern_size).T.reshape(-1, 2)
pattern_points *= square_size

obj_points = []
img_points = []

timeout = time.time()

camera_matrix, dist_coefs, new_camera_matrix = None, None, None

while True:
    ret, img = capture.read()


    # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY);
    # retval, img = cv2.threshold(img, 240, 255, cv2.THRESH_BINARY)


    cv2.imshow("distortion", img)

    if cv2.waitKey(1) >= 0:
        break

exit(0)

while True:
    ret, img = capture.read()

    height, width = img.shape[:2]

    if timeout < time.time() and len(img_points) < 20:
        found, corners = cv2.findChessboardCorners(img, pattern_size)

        print(found, flush = True)

        if found:
            term = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 30, 0.1)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY);
            cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1), term)
            cv2.drawChessboardCorners(img, pattern_size, corners, found)
            img_points.append(corners.reshape(-1, 2))
            obj_points.append(pattern_points)

        if len(img_points) > 10:
            rms, camera_matrix, dist_coefs, rvecs, tvecs = cv2.calibrateCamera(
                obj_points,
                img_points,
                (width, height),
                None,
                None)

            print(rms, flush = True)

            new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(
                camera_matrix,
                dist_coefs,
                (width, height),
                1,
                (width, height))

        timeout = time.time() + 0.1


    if dist_coefs != None:
        img = cv2.undistort(img, camera_matrix, dist_coefs, None, new_camera_matrix)

    cv2.imshow("distortion", img)

    if cv2.waitKey(1) >= 0:
        break
