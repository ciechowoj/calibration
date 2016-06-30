 #!/usr/bin/python3

import cv2
import time
import numpy

print("Sync")

def open_capture(name, frame):
    capture = cv2.VideoCapture(name)
    width = capture.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
    fps = capture.get(cv2.CAP_PROP_FPS)

    capture.set(cv2.CAP_PROP_POS_FRAMES, frame)

    print("Opened ", name, ", resolution ", width, "x", height, ", fps ", fps, flush = True)

    return capture

def shift_stream(stream, offset, master_fps = 30.0):
    name, frame, fps, brightness, env_mask = stream
    return (name, frame + offset / master_fps * fps, fps, brightness, env_mask)

def environ_mask(size):
    w, h = size

    result = numpy.empty((h, w), 'float32')

    H = 0.33 * h

    for y in range(result.shape[0]):
        for x in range(result.shape[1]):
            result[y, x] = max(1.0 - max((H - y) / H, 0.0), 0.0)

    return result

MASK = environ_mask((320, 240))

def show_frames(frames, threshold, env_mask, cell_size, nframe, ftime):
    global MASK

    array_size = int(numpy.ceil(numpy.sqrt(len(frames))))

    width = array_size * cell_size[0]
    height = array_size * cell_size[1]
    aspect = cell_size[0] / cell_size[1]

    output = numpy.zeros((height, width), 'uint8')

    result = [nframe, ftime] + [numpy.NaN, numpy.NaN] * len(frames)

    for i, frame in enumerate(frames):
        if isinstance(frame, type(None)):
            continue

        row = i // array_size
        col = i % array_size
        h, w = frame.shape[:2]

        resized = frame

        if w != width or h != height:
            a = w / h

            if a < aspect:
                dsize = (cell_size[0], cell_size[1])
                resized = cv2.resize(frame, dsize, None, 0, 0, cv2.INTER_CUBIC)
            else:
                dsize = (cell_size[0], cell_size[1])
                resized = cv2.resize(frame, dsize, None, 0, 0, cv2.INTER_CUBIC)

        x0 = col * cell_size[0]
        y0 = row * cell_size[1]
        x1 = x0 + resized.shape[1]
        y1 = y0 + resized.shape[0]

        resized = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY);
        resized = cv2.medianBlur(resized, 3)
        resized = (resized * (1.0 - (1.0 - MASK) * env_mask[i]))

        (minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(resized)

        if maxVal > threshold[i]:
            resized2 = resized.copy()
            cv2.circle(resized2, maxLoc, 40, (0, 0, 0), -1)
            (minVal2, maxVal2, minLoc2, maxLoc2) = cv2.minMaxLoc(resized2)

            if maxVal - maxVal2 > 4:
                cv2.circle(resized, maxLoc, 7, (0, 0, 0), 2)

                x, y = maxLoc
                x = (x / cell_size[0] - 0.5) * w
                y = (0.5 - y / cell_size[1]) * h

                result[2 + i * 2 + 1] = int(y)
                result[2 + i * 2 + 0] = int(x)

        output[y0:y1, x0:x1] = resized.clip(0, 255)

    # cv2.createTrackbar('trackbar', 'frame', value, 100, dummy)

    cv2.putText(
        output,
        "frame: {}, time: {:.2f}".format(nframe, ftime),
        (width - 350, height - 30),
        cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, 255)

    cv2.imshow('frame', output)

    return result

# name, offset, fps

offset = 2300

streams = [
    ("data/Z200.mp4",      13946, 24.815,       250, 0),
    ("data/HD.mp4",        14283, 24.83,        170, 0),
    ("data/notebook.mp4",  14521, 24.834,       200, 0),
    ("data/P8 Lite.mp4",   9513,  29.739420,    235, 1.0),
    ("data/620.mp4",       11998, 30.000,       255, 0.25),
    ("data/820.mp4",       8210,  30.00036,     253, 0),
    ("data/MB860.m4v",     5529,  22.86227973,  250, 0.0),
]

streams = [shift_stream(stream, offset) for stream in streams]

captures = [open_capture(stream[0], stream[1]) for stream in streams]

master_name = "data/P8 Lite.mp4"
master_index = [i for i, stream in enumerate(streams) if stream[0] == master_name][0]
master = captures[master_index]

frame_numbers = [0] * len(streams) # [int(stream[1]) for stream in streams]
frame_times = [1.0 / stream[2] for stream in streams]
frames = [None] * len(frame_numbers)

ESCAPE = 27
SPACE = ord(' ')
RETURN = ord('\n')

pause = False

positions = []
position = None

output_path = "positions.csv"
output_file = open(output_path, "w+")

while True:
    wk = cv2.waitKey(1)

    if wk & 0xff == SPACE:
        pause = not pause

    if not pause:
        retval, frames[master_index] = master.read()
        frame_numbers[master_index] += 1

        master_time = frame_numbers[master_index] * frame_times[master_index]

        valid = 0

        for i, frame_time in enumerate(frame_times):
            if i != master_index:
                time = frame_time * frame_numbers[i]

                while time <= master_time:
                    # resync last stream
                    if streams[i][0].endswith("MB860.m4v") and frame_numbers[master_index] % 128 == 0:
                        captures[i].set(cv2.CAP_PROP_POS_FRAMES, streams[i][1] + master_time * streams[i][2])

                    retval, new_frame = captures[i].read()

                    if retval:
                        valid += 1

                    frames[i] = new_frame
                    frame_numbers[i] += 1
                    time = frame_time * frame_numbers[i]

        position = show_frames(
            frames,
            [stream[3] for stream in streams],
            [stream[4] for stream in streams],
            (320, 240),
            frame_numbers[master_index] - 1 + offset,
            master_time)

    if  frame_numbers[master_index] % 4 == 0 or wk & 0xFF == RETURN:
        positions.append(position)
        print("len(positions) = ", len(positions), flush = True)

    if wk & 0xFF == ESCAPE:
        break

for position in positions:
    output_file.write("; ".join(map(str, position)) + "\n")

# Release everything if job is finished
for capture in captures:
    capture.release()

cv2.destroyAllWindows()







