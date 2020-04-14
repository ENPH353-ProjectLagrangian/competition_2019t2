"""
Playing with opencvs blob detector parameters for finding our top down homography.
"""

import cv2
import numpy as np

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

params_ref.minArea = 500
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

img = cv2.imread("ref_img.png")
img = cv2.resize(img, (665, 915))

print(img.shape)

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

keypoints = detector_ref.detect(gray)

im_with_keypoints = cv2.drawKeypoints(img, keypoints, np.array([]),
                                      (0, 0, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

cv2.imwrite("blobs1.jpg", im_with_keypoints)

img = cv2.imread("sim-calibration.png")
print(img.shape)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
# cv2.imshow("thresh", thresh)
# cv2.waitKey(0)
keypoints = detector1.detect(thresh)

centers = np.array([[kp.pt] for kp in keypoints], dtype=np.float32)

print(centers.shape)

im_with_keypoints = cv2.drawKeypoints(img, keypoints, np.array([]),
                                      (0, 0, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

cv2.drawChessboardCorners(img, (4, 11), centers, True)
cv2.imshow("img", img)
cv2.waitKey(0)
cv2.imwrite("blobs2.jpg", im_with_keypoints)
