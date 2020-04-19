import numpy as np
import cv2

"""
Script for generating the top down homography used to find the top down view of image seen by robot.
"""
# For explanation of parameters see https://www.learnopencv.com/blob-detection-using-opencv-python-c/

# Parameters for image seen by robot.
observed_img_detector_param = cv2.SimpleBlobDetector_Params()
observed_img_detector_param.filterByArea = True
observed_img_detector_param.filterByCircularity = False
observed_img_detector_param.filterByConvexity = False
observed_img_detector_param.filterByInertia = False
observed_img_detector_param.filterByColor = False

observed_img_detector_param.minArea = 0
observed_img_detector_param.maxArea = 10000

observed_img_detector_param.minThreshold = 120
observed_img_detector_param.maxThreshold = 255

observed_img_detector_param.minConvexity = 0
observed_img_detector_param.maxConvexity = 1

observed_img_detector_param.minRepeatability = 1

observed_img_detector_param.minDistBetweenBlobs = 1

# Parameters for reference image
reference_img_detector_params = cv2.SimpleBlobDetector_Params()
reference_img_detector_params.filterByArea = True
reference_img_detector_params.filterByCircularity = True
reference_img_detector_params.filterByConvexity = True
reference_img_detector_params.filterByInertia = False
reference_img_detector_params.filterByColor = False

reference_img_detector_params.minArea = 10
reference_img_detector_params.maxArea = 50000

reference_img_detector_params.minThreshold = 150
reference_img_detector_params.maxThreshold = 255
reference_img_detector_params.minConvexity = 0.9
reference_img_detector_params.maxConvexity = 1

reference_img_detector_params.minCircularity = 0.8
reference_img_detector_params.maxCircularity = 1

# Create detector for observed image blobs
detector_obs = cv2.SimpleBlobDetector_create(observed_img_detector_param)
# Create detector for reference image blobs.
detector_ref = cv2.SimpleBlobDetector_create(reference_img_detector_params)

# Size of the assmetric circle grid
nrows = 4
ncolumns = 11

# 3cm between circles
grid_size = 0.03

# Arrays to store object points and image points from all the images.
objp = []
for i in range(ncolumns):
    for j in range(nrows):
        objp.append((i * grid_size, (2 * j + i % 2) * grid_size, 0))

objp = np.array(objp).astype('float32')

"""
Reference image new_ref.jpg generated from make_homography_reference.py
"""
ref_image = cv2.imread("new_ref.jpg")

# Image read from robot. Image found by placing assymetrical circle grid flat in front of robot in gazebo simulation.

obs_img = cv2.imread("sim-calibration.png")
print(obs_img.shape)
gray = cv2.cvtColor(obs_img, cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)

"""
Find circles in the observed image. 
FindCirclesGrid didn't work here for whatever reason but using the detector directly does.
"""
keypoints = detector_obs.detect(thresh)
circles = np.array([[kp.pt] for kp in keypoints], dtype=np.float32)

"""
Find circles in the reference image.
"""
ret_1, ref_circles = cv2.findCirclesGrid(ref_image, (nrows, ncolumns), flags=cv2.CALIB_CB_ASYMMETRIC_GRID,
                                         blobDetector=detector_ref)
print(ret_1)

# Draw and display the corners

h, mask = cv2.findHomography(circles, ref_circles)
np.save("homography-sim.npy", h)
np.save("shape-sim.npy", ref_image.shape)
print(ref_image.shape)
cal_out = cv2.warpPerspective(obs_img, h, (ref_image.shape[1], ref_image.shape[0]))
drawn_image = cv2.drawChessboardCorners(obs_img, (nrows, ncolumns), circles, ret_1)
drawn_reference = cv2.drawChessboardCorners(ref_image, (nrows, ncolumns), ref_circles, ret_1)

cv2.imshow('img', drawn_reference)
cv2.imshow('img', drawn_image)
cv2.imshow("cal_out.png", cal_out)
cv2.waitKey(0)
