import numpy as np
import cv2

cap = cv2.VideoCapture(0)
while(cap.isOpened()):
    ret, frame = cap.read()
    if ret == False:
        break
    cv2.imwrite('camImg.jpg',frame)
    break
# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()