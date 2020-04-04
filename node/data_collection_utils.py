import numpy as np
import cv2
import global_variables as gv


def undistort(img):
    mtx = np.load(gv.path + "/Data_Collection/mtx.npy")
    dist = np.load(gv.path + "/Data_Collection/dist.npy")
    h, w = img.shape[:2]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(
        mtx, dist, (w, h), 1, (w, h))

    # undistort
    dst = cv2.undistort(img, mtx, dist, None, newcameramtx)

    # crop the image
    x, y, w, h = roi
    dst = dst[y:y + h, x:x + w]
    return dst
