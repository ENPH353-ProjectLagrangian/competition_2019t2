#!/usr/bin/env python

import image_processing_utils as ipu
import numpy as np
import rospy as ros
import cv2
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from geometry_msgs.msg import Twist
import global_variables as gv
import copy


def get_image(imgmsg):
    # Try to convert
    cv_image = bridge.imgmsg_to_cv2(imgmsg)
    cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
    return cv_image


def process_image(imgmsg):
    global heading
    global turn_counter
    global switch_index
    global skip

    img = get_image(imgmsg)
    keypoints = ipu.detect_crosswalk(img)
    cv2.imshow("blue", img[:, :, 0])
    im_with_keypoints = cv2.drawKeypoints(img, keypoints, np.array([]),
                                          (0, 0, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    cv2.imshow("keypoints", im_with_keypoints)

    # Generates the top down view used to find headings.

    top_down = cv2.warpPerspective(img, homography, (shape[1], shape[0]))

    gray = cv2.cvtColor(top_down, cv2.COLOR_BGR2GRAY)

    ret, thresh = cv2.threshold(gray, 210, 255, cv2.THRESH_BINARY)
    edges = cv2.Canny(thresh, 50, 150, apertureSize=3)
    lines = ipu.get_hough_lines(img=edges, img_type="edges", threshold=threshold)

    if lines is not None:
        if visualize:
            ipu.draw_lines(top_down, lines)

        angles = lines[:, 0, 1]
        angles = np.array(list([[-angle, np.pi - angle][angle > np.pi / 2]] for angle in angles), dtype=np.float32)

        headings, cluster_indexes = ipu.find_cardinal_clusters(angles)

        clusters = [np.array([lines[i] for i in cluster_indexes[0]]),
                    np.array([lines[i] for i in cluster_indexes[1]])]

        #  There is an ambiguity in which direction is "forward" when theta is close to pi/2.
        #  -1 is right turn, 1 is left turn.
        turn_bias = -1  # Starts off at -1 could be changed by reinforcement learning
        # but probably shouldn't for this naive approach.

        # The margin of what is considered "close" to an angle
        tolerance = np.pi / 60  # 3 degrees

        head_abs = np.abs(headings[1])
        if np.abs(head_abs - 1.57) < tolerance:
            headings[1] = turn_bias * head_abs

        #  Initial state
        state_x = 3 * [0]
        state_y = 5 * [0]

        if verbose:
            print("The most recent heading was: " + str(heading))

        # Chooses the heading closest to the previous heading to follow
        aligned_index = (np.abs(headings - heading)).argmin()

        alignment_tolerance = np.pi / 60

        aligned = headings[aligned_index][0] < alignment_tolerance
        if aligned:  # Checks to see if robot is aligned

            for line in clusters[0]:
                rho = line[0][0]
                i = int(rho / w * 3)
                i = max(i, 0)
                i = min(i, 2)
                state_x[i] = 1

            for line in clusters[1]:
                rho = line[0][0]
                i = int(rho / h * 5)
                i = max(i, 0)
                i = min(i, 4)
                state_y[i] = 1

            if visualize:
                grid_lines = []
                for i in range(1, 3):
                    grid_lines.append([[int(i / 3.0 * w), 0]])
                for i in range(1, 5):
                    grid_lines.append([[int(i / 5.0 * h), np.pi / 2]])
                ipu.draw_lines(top_down, grid_lines, color=(0, 255, 0))
        else:
            print(lines)

        turn = (state_y == [0, 0, 0, 0, 1] and state_x[1] == 0)

        # The first corner defines the index count for long stretches of road.
        if turn and switch_index is None:
            switch_index = turn_counter

        # We want a cooldown before we turn again
        turn = turn and turn_counter > switch_index * 1 / 3.0

        if not turn:
            heading = headings[aligned_index][0]
            turn_counter += 1
        else:
            if turn_counter <= switch_index * 2 / 3.0:
                if skip == 2:
                    skip = -1

                    heading = headings[aligned_index][0]
                else:
                    skip += 1
                    heading = headings[int(not aligned_index)][0]
            else:
                heading = headings[int(not aligned_index)][0]

            print("turn counter: " + str(turn_counter))
            turn_counter = 0

            print("skip state: " + str(skip))

        if verbose:
            print("The current primary directions are {0!s} the new heading is {1!s}.".format(headings, heading))

        error = heading

        move.angular.z = p_gain * error
        move.linear.x = x_vel
    else:
        # If it can't find the line then
        move.angular.x = x_vel
        move.angular.z = p_gain * heading

    if visualize:
        cv2.imshow('top down', top_down)
        cv2.imshow("image", img)
        cv2.waitKey(1)

    pub.publish(move)


if __name__ == '__main__':
    # Some booleans to adjust
    visualize = True
    verbose = False
    clustering_algorithm = "custom"

    ros.init_node('topic_publisher')

    pub = ros.Publisher('/cmd_vel', Twist, queue_size=1)
    capture_period = ros.Rate(50)

    capture_mode = "periodic"

    bridge = CvBridge()
    move = Twist()

    homography = np.load(gv.path + "/assets/homography-sim_v4.npy")
    shape = np.load(gv.path + "/assets/shape-sim_v4.npy")

    h = 732
    w = shape[0]

    threshold = 100

    # Path following
    p_gain = 1
    x_vel = 0.2
    heading = 0
    skip = 0
    turn_counter = 0
    switch_index = None
    ros.Subscriber('/rrbot/camera1/image_raw', Image, process_image)
    ros.spin()
