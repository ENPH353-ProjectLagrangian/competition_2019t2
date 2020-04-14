# import the necessary packages
import numpy as np
import cv2

params_1 = cv2.SimpleBlobDetector_Params()
params_1.filterByArea = True
params_1.filterByCircularity = False
params_1.filterByConvexity = False
params_1.filterByInertia = False
params_1.filterByColor = False

params_1.minArea = 0
params_1.maxArea = 10000

params_1.minThreshold = 120
params_1.maxThreshold = 255

params_1.minConvexity = 0
params_1.maxConvexity = 1

params_1.minRepeatability = 1

params_1.minDistBetweenBlobs = 1

params_ref = cv2.SimpleBlobDetector_Params()
params_ref.filterByArea = True
params_ref.filterByCircularity = True
params_ref.filterByConvexity = True
params_ref.filterByInertia = False
params_ref.filterByColor = False

params_ref.minArea = 10
params_ref.maxArea = 50000

params_ref.minThreshold = 150
params_ref.maxThreshold = 255
params_ref.minConvexity = 0.9
params_ref.maxConvexity = 1
#
params_ref.minCircularity = 0.8
params_ref.maxCircularity = 1

detector1 = cv2.SimpleBlobDetector_create(params_1)
detector_ref = cv2.SimpleBlobDetector_create(params_ref)

nrows = 4
ncolumns = 11

objp = []
grid_size = 0.03  # 3cm, or whatever

for i in range(ncolumns):
    for j in range(nrows):
        objp.append((i * grid_size, (2 * j + i % 2) * grid_size, 0))

objp = np.array(objp).astype('float32')

# Arrays to store object points and image points from all the images.
ref_image = cv2.imread("new_ref.jpg")

img_1 = cv2.imread("sim-calibration.png")

gray_img = cv2.cvtColor(img_1, cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(gray_img, 200, 255, cv2.THRESH_BINARY)

img = cv2.imread("sim-calibration.png")
print(img.shape)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
# cv2.imshow("thresh", thresh)
# cv2.waitKey(0)
keypoints = detector1.detect(thresh)

circles = np.array([[kp.pt] for kp in keypoints], dtype=np.float32)

ret_1, ref_circles = cv2.findCirclesGrid(ref_image, (nrows, ncolumns), flags=cv2.CALIB_CB_ASYMMETRIC_GRID,
                                         blobDetector=detector_ref)
print(ret_1)

objpoints = []  # 3d point in real world space
imgpoints = []  # 2d points in image plane.

objpoints.append(objp)

imgpoints.append(circles)

# Draw and display the corners


h, mask = cv2.findHomography(circles, ref_circles)
np.save("homography-sim.npy", h)
np.save("shape-sim.npy", ref_image.shape)
print(ref_image.shape)
cal_out = cv2.warpPerspective(img_1, h, (ref_image.shape[1], ref_image.shape[0]))
drawn_image = cv2.drawChessboardCorners(img_1, (nrows, ncolumns), circles, ret_1)
drawn_reference = cv2.drawChessboardCorners(ref_image, (nrows, ncolumns), ref_circles, ret_1)

cv2.imshow('img', drawn_reference)
cv2.imshow('img', drawn_image)
cv2.imshow("cal_out.png", cal_out)
cv2.waitKey(0)
