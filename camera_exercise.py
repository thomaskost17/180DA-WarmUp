# camera_exercise.py

# This script makes use of https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_colorspaces/py_colorspaces.html#converting-colorspaces
# This script makes use of https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_contours/py_contour_features/py_contour_features.html#contour-features
# This script makes use of https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_gui/py_video_display/py_video_display.html
# In this script the above code was adapted to mask and track a water bottle and identify it with a bounding box

import numpy as np
import cv2
from sklearn.cluster import KMeans
#import matplotlib.pyplot as plt


cap = cv2.VideoCapture(0)

# error opening
if not (cap.isOpened()):
    print("Could not open video device")

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Our operations on the frame come here
    lower_blue = np.array([85,50,50])
    upper_blue = np.array([115,255,255])

    # Threshold the HSV image to get only blue colors
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_blue, upper_blue)
    res = cv2.bitwise_and(frame,frame, mask= mask)

    #ret,thresh = cv2.threshold(mask,127,255,cv2.THRESH_BINARY)
    contours,hierarchy = cv2.findContours(mask, 1, 2)
    size = []
    for i in contours:
        size.append(len(i))
    print(size)
    cnt = contours[np.argmax(size)]
    x,y,w,h = cv2.boundingRect(cnt)
    img = cv2.rectangle(res,(x,y),(x+w,y+h),(0,255,0),2)

    cframe = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # implementing k means
    # region of intrest

    # selected = cframe[640:1280, 360:720]
    # print(selected.shape[0])
    # selected = selected.reshape((selected.shape[0] * selected.shape[1],3)) #represent as row*column,channel number
    # clt = KMeans(n_clusters=3) #cluster number
    # clt.fit(selected)

    # Display the resulting frame
    cv2.imshow('frame',img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()