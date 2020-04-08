#!/usr/bin/env python

import image_processing_utils as ipu
import numpy as np
import rospy as ros
import cv2
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from geometry_msgs.msg import Twist
# import copy
import global_variables as gv

from plate_reader import PlateReader

visualize = True

ros.init_node('topic_publisher')

pub = ros.Publisher('/cmd_vel', Twist, queue_size=1)
capture_period = ros.Rate(50)

capture_mode = "periodic"

bridge = CvBridge()
move = Twist()

h = 1830
w = 1330

homography = np.load(gv.path + "/assets/homography-sim.npy")
shape = np.load(gv.path + "/assets/shape-sim.npy")

threshold = 125

# K means stuff
# Define criteria = ( type, max_iter = 10 , epsilon = 1.0 )
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)

# Set flags (Just to avoid line break in the code)
flags = cv2.KMEANS_RANDOM_CENTERS

# Path following

heading = 0

plate_reader = PlateReader(gv.path + '/assets/num_model.h5',
                           gv.path + '/assets/char_model.h5')


def get_image(imgmsg):
    # Try to convert
    cv_image = bridge.imgmsg_to_cv2(imgmsg)

    return cv2.cvtColor(cv_image, cv2.COLOR_RGB2BGR)


def process_image(imgmsg):
    global heading

    # --- PLATE READER ---
    # plate_reader = PlateReader(gv.path + '/assets/num_model.h5',
    #                            gv.path + '/assets/char_model.h5')
    img = get_image(imgmsg)
    parking_spot, license_text, certainty = plate_reader.process_image(img)
    if (parking_spot is not None):
        print("Parking {}: {} ({})".format(parking_spot, license_text,
                                           certainty))
    # -- PLATE READER ---

    top_down = cv2.warpPerspective(img, homography, (shape[1], shape[0]))

    gray = cv2.cvtColor(top_down, cv2.COLOR_BGR2GRAY)

    ret, thresh = cv2.threshold(gray, 210, 255, cv2.THRESH_BINARY)
    edges = cv2.Canny(thresh, 50, 150, apertureSize=3)
    lines = ipu.get_hough_lines(img=edges, img_type="edges", threshold=threshold)

    if lines is not None:

        if visualize:
            ipu.draw_lines(top_down, lines)

        if lines.__len__() == 1:
            error = -lines[:, 0, 1][0]
        else:
            angles = lines[:, 0, 1]
            angles = np.array(list([[-angle, angle - np.pi][angle > np.pi / 2]] for angle in angles), dtype=np.float32)

            # Apply KMeans
            compactness, labels, headings = cv2.kmeans(angles, 2, None, criteria, 10, flags)

            clusters = np.array([angles[labels == 0], angles[labels == 1]])

            # Chooses the heading closest to the previous heading to follow

            # idx = (np.abs(headings - heading)).argmin()
            #
            # switch = False
            # if switch is True:
            #     heading = headings[not idx][0]
            # else:
            #     heading = headings[idx][0]
            #
            # print(heading)
            # error = heading

            idx = (np.abs(headings - heading)).argmin()
            print("")

            heading = headings[idx][0]

            error = heading
        p_gain = 1

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
    plate_reader.get_all_license(print_str=True)