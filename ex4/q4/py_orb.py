import sys # args
import numpy as np # numpy<2
import cv2 as cv
from matplotlib import pyplot as plt


if len(sys.argv) < 2:
    print("Usage: python py_orb.py <image_file>")
    sys.exit(1)

img = cv.imread(sys.argv[1], cv.IMREAD_GRAYSCALE)
# Initiate ORB detector
orb = cv.ORB_create()
# find the keypoints with ORB
kp = orb.detect(img,None)
# compute the descriptors with ORB
kp, des = orb.compute(img, kp)
# draw only keypoints location,not size and orientation
img2 = cv.drawKeypoints(img, kp, None, color=(0,255,0), flags=0)
plt.imshow(img2), plt.show()