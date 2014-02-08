import numpy as np
import cv2

MIN_MATCH_COUNT = 80
SHADOW_V = 120 #the (hs)v threshold for up or down.

#load fiducial
tag = cv2.imread('artags/60mmpair.jpg',0)
real_width = 95.0 #mm
# h and w of fiducial
h,w = tag.shape

#braille pin spacing and positioning wrt top left corner of fiducial
x_pitch = 2.5
y_pitch = 2.5
pin_x = 50.6
pin_y = 14.8
shadow_size = 10

#useful function
def mm2px(mm):
    return mm * (w/real_width)

#load in one of our distorted images
img = cv2.imread('photo.jpg') # trainImage

# Load previously saved data about the camera - generated with the chessboard photos
calib = np.load('B.npz')
mtx = calib['mtx']
dist = calib['dist']

#undistort the image!
ph, pw = img.shape[:2]
newcameramtx, roi=cv2.getOptimalNewCameraMatrix(mtx,dist,(pw,ph),1,(pw,ph))
img = cv2.undistort(img, mtx, dist, None, newcameramtx)

# Initiate ORB detector
orb = cv2.ORB()

#find the keypoints and descriptors with ORB
#fiducial
tag_kp = orb.detect(tag,None)
tag_kp, tag_des = orb.compute(tag,tag_kp)
#photo
img_kp = orb.detect(img,None)
img_kp, img_des = orb.compute(img,img_kp)

#does the match, if it's good returns the homography transform
def find(des,kp):
    # create BFMatcher object
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    # Match descriptors.
    matches = bf.match(des,img_des)

    # Sort them in the order of their distance.
    matches = sorted(matches, key = lambda x:x.distance)

    print "matches are found - %d/%d" % (len(matches),MIN_MATCH_COUNT)
    
    if len(matches)>MIN_MATCH_COUNT:
        src_pts = np.float32([ kp[m.queryIdx].pt for m in matches[:MIN_MATCH_COUNT] ]).reshape(-1,1,2)
        dst_pts = np.float32([ img_kp[m.trainIdx].pt for m in matches[:MIN_MATCH_COUNT] ]).reshape(-1,1,2)

        #get the transformation between the flat fiducial and the found fiducial in the photo
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)

        #return the transform
        return M
    else:
        print "Not enough matches are found - %d/%d" % (len(matches),MIN_MATCH_COUNT)

#draws a box round the fiducial
def draw_outline(M):
    #array containing co-ords of the fiducial
    pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
    #transform the coords of the fiducial onto the picture
    dst = cv2.perspectiveTransform(pts,M)
    #draw a box around the fiducial
    cv2.polylines(img,[np.int32(dst)],True,(255,0,0),5, cv2.CV_AA)

#find the shadows of the pins!
def detect_shadows(M):
    pin_num = 0
    #for each shadow: pins are numbered 0 to 2 in the left column, 3 to 5 in the right
    for x in range(2):
        for y in range(3):

            pin_pt=np.float32([
                [mm2px(pin_x+x*x_pitch),mm2px(pin_y+y*y_pitch)],
                ]).reshape(-1,1,2)
            #transform the pin coords
            pin_dst = cv2.perspectiveTransform(pin_pt,M)

            #the pin x and y in the image
            pin_x_dst = pin_dst[0][0][0]
            pin_y_dst = pin_dst[0][0][1]
            print pin_x_dst , pin_y_dst

            #define a ROI for the shadow - cv2 only does square
            roi = img[pin_y_dst:pin_y_dst+shadow_size,pin_x_dst:pin_x_dst+shadow_size]

            #save it for reference
            cv2.imwrite('roi' + str(pin_num) + '.png', roi)

            #convert to HSV
            hsvroi = cv2.cvtColor(roi,cv2.COLOR_BGR2HSV)

            #mean value
            val = cv2.mean(hsvroi)[2]

            #is it up or down?
            print("pin %d v%0.2f = %s" % (pin_num, val, "down" if val > SHADOW_V else "up"))
            pin_num += 1

    #cv2.circle(img,(int(pin_x_dst),int(pin_y_dst)),10,(255,0,0),-1)
    #put a box round the pins
    #roidst = cv2.perspectiveTransform(roipts,M)
    #cv2.polylines(img,[np.int32(roidst)],True,(255,0,0),3, cv2.CV_AA)
    #cv2.line(img,tuple(roidst[0][0]),tuple(roidst[1][0]),(255,0,0),3)


#find the fiducial
print( "find fiducial")
M = find(tag_des,tag_kp)
draw_outline(M)
detect_shadows(M)

#write out full size image
cv2.imwrite('found.png', img)

#resize for display
newx,newy = img.shape[1]/4,img.shape[0]/4 #new size (w,h)
img = cv2.resize(img,(newx,newy))

#show the image
cv2.imshow('found',img)
cv2.waitKey()
