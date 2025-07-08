#!/usr/bin/env python3.8

import cv2
import numpy as np
import matplotlib.pyplot as plt
import time

def capture_image(frame):
    filename = f'root/YOLOv11/src/yolov11/scripts/raw_img/raw_{time.time()}.jpg'
    cv2.imwrite(filename, frame)

# Open the default camera
cam = cv2.VideoCapture(1)

# Get the default frame width and height
frame_width = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))

count_colors = [0, 0, 0, 0]
label = ['red', 'green', 'blue', 'yellow']
label_idx = 0


while True:
    ret, frame = cam.read()
    if not ret:
        break

    # Display the captured frame with label
    cv2.imshow('Camera', frame)

    # Only call waitKey once per loop
    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break

    capture_image(frame)
    time.sleep(0.1)

# Release the capture and writer objects
cam.release()
cv2.destroyAllWindows()
