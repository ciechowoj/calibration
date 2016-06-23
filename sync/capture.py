 #!/usr/bin/python3

import cv2
import sys
import time
import numpy
from threading import Thread, Lock

capture = cv2.VideoCapture(int(sys.argv[2]))

fourcc = cv2.VideoWriter_fourcc(*'XVID')

FPS = 12.5
SPF = 1.0 / FPS

width, height = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

global_quit = False
global_frame = None
global_error = False
grabber_lock = Lock()

def grabber():
    global global_quit, global_frame, global_error, grabber_lock

    local_quit = False

    while not local_quit:
        retval, new_frame = capture.read()

        grabber_lock.acquire()

        if retval:
            global_frame = new_frame
        else:
            global_error = True
            return

        local_quit = global_quit

        grabber_lock.release()


grabber_thread = Thread(target = grabber)
grabber_thread.start()

local_frame = numpy.zeros((height, width, 3), "uint8")
frame_time = 0.0
output = None

while(capture.isOpened()):
    grabber_lock.acquire()

    if global_error:
        break

    if not isinstance(global_frame, type(None)):
        local_frame = global_frame

    grabber_lock.release()

    if output == None:
        output = cv2.VideoWriter(
            sys.argv[1],
            fourcc,
            FPS,
            (width, height))

        frame_time = time.time()

    new_time = time.time()

    if new_time < frame_time + SPF:
        time.sleep(0.001)
    else:
        if new_time > frame_time + SPF * 1.1:
            print("WARNING! Too slow recording...", flush = True)

        output.write(local_frame)
        frame_time = new_time
        cv2.imshow('frame', local_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            global_quit = True
            break

grabber_thread.join()

# Release everything if job is finished
capture.release()
output.release()
cv2.destroyAllWindows()