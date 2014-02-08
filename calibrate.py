import numpy as np
import cv2
import glob

# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

chess_r = 7
chess_c = 7
square_size = 10 #mm
# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((chess_r*chess_c,3), np.float32)
objp[:,:2] = np.mgrid[0:chess_c*square_size:square_size,0:chess_r*square_size:square_size].T.reshape(-1,2)

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

images = glob.glob('left*.jpg')
#images = ['left01.jpg']

for fname in images:
    print(fname)
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(gray, (chess_c,chess_r),None)

    # If found, add object points, image points (after refining them)
    if ret == True:
        objpoints.append(objp)

        cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
        imgpoints.append(corners)

        # Draw and display the corners
        print("found corners",len(corners))
        cv2.drawChessboardCorners(img, (chess_c,chess_r), corners,ret)
        cv2.imshow('img',img)
        cv2.waitKey()

#get calibration of camera
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)
np.savez('B',mtx=mtx,dist=dist)

cv2.destroyAllWindows()
