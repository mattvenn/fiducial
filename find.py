import numpy as np
import cv2
from matplotlib import pyplot as plt

MIN_MATCH_COUNT = 10

tag1 = cv2.imread('artag1.jpg',0)          # queryImage
tag2 = cv2.imread('artag2.jpg',0)          # queryImage
img2 = cv2.imread('artag_rotate.jpg',0) # trainImage

# Initiate SIFT detector
sift = cv2.SIFT()

# find the keypoints and descriptors with SIFT
tag1_kp1, tag1_des1 = sift.detectAndCompute(tag1,None)
tag2_kp1, tag2_des1 = sift.detectAndCompute(tag2,None)

kp2, des2 = sift.detectAndCompute(img2,None)


def find(tag1_des1,tag1_kp1):
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks = 50)

    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(tag1_des1,des2,k=2)

    # store all the good matches as per Lowe's ratio test.
    good = []
    for m,n in matches:
        if m.distance < 0.7*n.distance:
            good.append(m)


    if len(good)>MIN_MATCH_COUNT:
        src_pts = np.float32([ tag1_kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
        dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)

        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
        matchesMask = mask.ravel().tolist()

        h,w = tag1.shape
        pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
        dst = cv2.perspectiveTransform(pts,M)

        print(dst)
        cv2.polylines(img2,[np.int32(dst)],True,(0,0,0),3, cv2.CV_AA)
        cv2.circle(img2,(100,200),3,(255,0,0),-1)

        M = cv2.moments(dst)
        centroid_x = int(M['m10']/M['m00'])
        centroid_y = int(M['m01']/M['m00'])
        return(centroid_x,centroid_y)

    else:
        print "Not enough matches are found - %d/%d" % (len(good),MIN_MATCH_COUNT)
        matchesMask = None

pt1=find(tag1_des1,tag1_kp1)
pt2=find(tag2_des1,tag2_kp1)
cv2.line(img2,pt1,pt2,(0,0,0),3)
cv2.imshow('found',img2)
cv2.waitKey()
