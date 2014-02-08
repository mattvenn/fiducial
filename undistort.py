import numpy as np
import cv2
import glob

# Load previously saved data
calib = np.load('B.npz')
mtx = calib['mtx']
dist = calib['dist']

#load in one of our distorted images
img = cv2.imread('canon_training/left03.jpg')
h, w = img.shape[:2]
newcameramtx, roi=cv2.getOptimalNewCameraMatrix(mtx,dist,(w,h),1,(w,h))
#undistort!
dst = cv2.undistort(img, mtx, dist, None, newcameramtx)

newx,newy = dst.shape[1]/4,dst.shape[0]/4 #new size (w,h)
dst = cv2.resize(dst,(newx,newy))
cv2.imshow('img',dst)
cv2.waitKey()
