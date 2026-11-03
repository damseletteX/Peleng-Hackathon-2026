import numpy as np
import cv2
import time


# creating the videocapture object
# and reading from the input file
# Change it to 0 if reading from webcam

cap = cv2.VideoCapture('../configs/cam1mp4')

# used to record the time when we processed last frame
prev_frame_time = 0
fps = 0
# used to record the time at which we processed current frame
new_frame_time = 0

# Reading the video file until finished
while(cap.isOpened()):

    # Capture frame-by-frame

    ret, frame = cap.read()

    # if video finished or no Video Input
    if not ret:
        break

    # Our operations on the frame come here
    gray = frame
    new_frame_time = time.time()

    # resizing the frame size according to our need
    gray = cv2.resize(gray, (500, 300))
    # Calculating the fps

    # fps will be number of frame processed in given time frame
    # since their will be most of time error of 0.001 second
    # we will be subtracting it to get more accurate result
    fps = 1/(new_frame_time-prev_frame_time)
    prev_frame_time = new_frame_time

    # converting the fps into integer
    fps = int(fps)
    print("FPS: ", fps)
# When everything done, release the capture
cap.release()
# Destroy the all windows now
cv2.destroyAllWindows()