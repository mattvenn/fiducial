import cv2
import numpy as np
import glob

# Load previously saved data
calib = np.load('B.npz')
mtx = calib['mtx']
dist = calib['dist']

chess_r = 7
chess_c = 7
square_size = 10 #mm

def draw(img, corners, imgpts):
    corner = tuple(corners[0].ravel())
    cv2.line(img, corner, tuple(imgpts[0].ravel()), (255,0,0), 3)
    cv2.line(img, corner, tuple(imgpts[1].ravel()), (0,255,0), 3)
    cv2.line(img, corner, tuple(imgpts[2].ravel()), (0,0,255), 3)
    return img

def show_pins(img,corners,imgpts):
    for pin in range(3):
        point = tuple(imgpts[pin].ravel())
        shadow_offset = 5
        shadow_size = 20
        #range has to be from lower to higher number
        x1 = int(point[0]-shadow_offset)
        x2 = int(point[0]-(shadow_offset+shadow_size))
        y1 = int(point[1]+shadow_offset)
        y2 = int(point[1]+(shadow_offset+shadow_size))
        pt1 = (x1,y1)
        pt2 = (x2,y2)
        #print x2,x1,y1,y2
        roi = img[y1:y2,x2:x1] #,y1:y2]
        print roi.shape
        cv2.imwrite('roi' + str(pin) + '.png', roi)
    
        hsvroi = cv2.cvtColor(roi,cv2.COLOR_BGR2HSV)
        val = cv2.mean(hsvroi)[2]
        print("pin %d %f %s" % (pin, val, "down" if val > 120 else "up"))

        cv2.rectangle(img,pt1,pt2,(255,0,0),3)
        cv2.circle(img,tuple(imgpts[pin].ravel()), 10, (255,0,0),-1)

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
objp = np.zeros((chess_r*chess_c,3), np.float32)
objp[:,:2] = np.mgrid[0:chess_c*square_size:square_size,0:chess_r*square_size:square_size].T.reshape(-1,2)


axis = np.float32([
    [9*square_size,1*square_size,0],
    [9*square_size,2*square_size,0],
    [9*square_size,3*square_size,0],
    ]).reshape(-1,3)

images = glob.glob('pinboard_pics/*.jpg')
#images = ['pinboard101.jpg'] # works
#images = ['pinboard111.jpg'] # wrong corner
#mages = ['pinboard100.jpg'] # wrong corner
#for fname in glob.glob('left*.jpg'):
for fname in images:
    print(fname)
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, (chess_c,chess_r),None)

    if ret == True:
        cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)

        # Find the rotation and translation vectors.
        rvecs, tvecs, inliers = cv2.solvePnPRansac(objp, corners, mtx, dist)

        # project 3D points to image plane
        imgpts, jac = cv2.projectPoints(axis, rvecs, tvecs, mtx, dist)
        #print imgpts

        show_pins(img,corners,imgpts)
        draw(img,corners,imgpts)
        #resize
        newx,newy = img.shape[1]/4,img.shape[0]/4 #new size (w,h)
        img = cv2.resize(img,(newx,newy))
        cv2.imshow('img',img)
        k = cv2.waitKey(0) & 0xff

cv2.destroyAllWindows()

