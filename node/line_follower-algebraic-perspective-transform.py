#!/usr/bin/env python

import image_processing_utils as ipu
import numpy as np
import rospy as ros
import cv2
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from geometry_msgs.msg import Twist
import global_variables as gv

visualize = True

ros.init_node('topic_publisher')

pub = ros.Publisher('/cmd_vel', Twist, queue_size=1)
capture_period = ros.Rate(50)

capture_mode = "periodic"

bridge = CvBridge()
move = Twist()

# Dimension stuff
h = 1830
w = 1330

cx = 320.0
cy = 240.0
f = 320.0 / np.tan(1)

ty = 0.365

homography = np.load(gv.path + "/assets/homography-sim_v0.npy")
shape = np.load(gv.path + "/assets/shape-sim.npy")

threshold = 125

# K means stuff
# Define criteria = ( type, max_iter = 10 , epsilon = 1.0 )
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)

# Set flags (Just to avoid line break in the code)
flags = cv2.KMEANS_RANDOM_CENTERS

# Path following

heading = 0


def get_image(imgmsg):
    # Try to convert
    cv_image = bridge.imgmsg_to_cv2(imgmsg)

    return cv_image


def process_image(imgmsg):
    global heading
    img = get_image(imgmsg)

    gray_0 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    ret, thresh_0 = cv2.threshold(gray_0, 210, 255, cv2.THRESH_BINARY)
    edges_0 = cv2.Canny(thresh_0, 50, 150, apertureSize=3)
    lines_0 = ipu.get_hough_lines(img=edges_0, img_type="edges", threshold=100)

    prime_lines = []
    for line in lines_0:
        rho, theta = line[0]
        theta_prime = np.arctan(-1 / f * (rho / np.cos(theta) - cy * np.tan(theta) - cx))
        rho_prime = (ty * np.tan(theta) + w / 2 - h * np.tan(theta_prime)) * np.cos(theta_prime)
        prime_lines.append([[rho_prime, theta_prime]])

    prime_lines = np.array(prime_lines)
    print(np.array_str(prime_lines[:, 0, 1], precision=4, suppress_small=True))

    top_down = cv2.warpPerspective(img, homography, (shape[1], shape[0]))

    gray = cv2.cvtColor(top_down, cv2.COLOR_BGR2GRAY)

    ret, thresh = cv2.threshold(gray, 210, 255, cv2.THRESH_BINARY)
    edges = cv2.Canny(thresh, 50, 150, apertureSize=3)
    lines = ipu.get_hough_lines(img=edges, img_type="edges", threshold=threshold)

    if lines is not None:

        if visualize:
            ipu.draw_lines(top_down, lines, color=(255, 0, 0))
            ipu.draw_lines(img, lines_0, color=(0, 255, 0))

            ipu.draw_lines(top_down, prime_lines, color=(0, 255, 0))

        angles = prime_lines[:, 0, 1]
        # angles = lines[:, 0, 1]

        angles = np.array(list([[-angle, angle - np.pi][angle > np.pi / 2]] for angle in angles), dtype=np.float32)

        if angles.__len__() == 1:
            error = angles[0][0]
        else:

            # Apply KMeans
            compactness, labels, headings = cv2.kmeans(angles, 2, None, criteria, 10, flags)

            # Chooses the heading closest to the previous heading to follow

            idx = (np.abs(headings - 0)).argmin()

            switch = False
            if switch is True:
                heading = headings[not idx][0]
            else:
                heading = headings[idx][0]

            # print("The previous heading was: {0!s}. Out of these headings {1!s} the current heading is {2!s}.".format(
            #        heading, headings, headings[idx][0]))

            heading = headings[idx][0]

            error = heading
        p_gain = 1.1

        move.angular.z = p_gain * error
        move.linear.x = 0.2
    else:
        move.angular.x = -0.3
        move.angular.z = 0.5

    if visualize:
        cv2.imshow('top down', top_down)
        cv2.imshow("image", img)
        cv2.waitKey(1)

    pub.publish(move)


if __name__ == '__main__':
    ros.Subscriber('/rrbot/camera1/image_raw', Image, process_image)
    ros.spin()
