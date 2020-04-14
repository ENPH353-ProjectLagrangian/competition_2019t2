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

from plate_reader import PlateReader


def find_cardinal_clusters(angles):
    # The K-means algorithm is sometimes unreliable.
    # But because we know the distribution of the angles already we can create a much more reliable one

    good_buckets = np.array([[0], [np.pi / 2]])
    min_include = -1

    half_bucket = np.pi * 5 / 180

    # First split the angles into buckets 90 degrees apart. Find the set of buckets that include the most points.
    # This is a course move through the state space buckets are 10 degrees wide spaced 5 degrees apart.

    for theta_0 in np.linspace(0, np.pi / 2.0, 18, endpoint=False):
        buckets = [[], [], []]
        for angle in angles:

            if theta_0 - half_bucket <= angle[0] <= theta_0 + half_bucket:
                buckets[0].append(angle)
            elif theta_0 - half_bucket - np.pi / 2.0 <= angle[0] <= theta_0 + half_bucket - np.pi / 2.0:
                buckets[1].append(angle)
            elif theta_0 - half_bucket + np.pi / 2.0 <= angle[0] <= theta_0 + half_bucket + np.pi / 2.0:
                buckets[2].append(angle)

        include = buckets[0].__len__() + buckets[1].__len__()

        if include > min_include:
            min_include = include
            good_buckets = copy.copy(buckets)

    #  Now we take a mean value to find the actual angle from the course buckets.

    theta = 0

    n = 0.0

    if good_buckets[0].__len__() != 0:
        theta += np.average(good_buckets[0])
        n += 1.0
    if good_buckets[1].__len__() != 0:
        theta += np.average(good_buckets[1]) - np.pi / 2
        n += 1.0
    if good_buckets[2].__len__() != 0:
        theta += np.average(good_buckets[2]) + np.pi / 2
        n += 1.0

    if n != 0:
        theta *= 1 / n

    #  Remake buckets at the refined center with indices.
    final_bucket = [[], []]

    for i in range(angles.__len__()):
        if theta - half_bucket <= angles[i][0] <= theta + half_bucket:
            final_bucket[0].append(i)
        elif theta - half_bucket - np.pi / 2.0 <= angles[i][0] <= theta + half_bucket - np.pi / 2.0 or \
                theta - half_bucket + np.pi / 2.0 <= angles[i][0] <= theta + half_bucket + np.pi / 2.0:
            final_bucket[1].append(i)

    headings = np.array([[theta], [theta + np.pi / 2.0]])
    clusters = np.array(final_bucket)

    # Normalize angles (pi=0, pi/2=-pi/2)
    headings = np.array(list([[heading[0], heading[0] - np.pi][heading[0] > np.pi / 2]] for heading in headings))

    if abs(headings[0]) > abs(headings[1]):
        headings = headings[::-1]
        clusters = clusters[::-1]
    return headings, clusters


def get_image(imgmsg):
    # Try to convert
    cv_image = bridge.imgmsg_to_cv2(imgmsg)

    return cv2.cvtColor(cv_image, cv2.COLOR_RGB2BGR)


def process_image(imgmsg):
    global heading
    global turn_counter
    global switch_index
    global skip
    global plate_reader

    img = get_image(imgmsg)
    parking, license_text, prob = plate_reader.process_image(img)
    if (parking is not None):
        print('Parking {}: {} ({})'.format(parking, license_text, prob))

    # Generates the top down view used to find headings.
    # (You can crop this to adjust when the dumb algorithm turns)

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

        headings, cluster_indexes = find_cardinal_clusters(angles)

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
    # image processing
    num_path = gv.path + '/assets/num_model_new.h5'
    char_path = gv.path + '/assets/char_model_new.h5'
    plate_reader = PlateReader(num_path, char_path)

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
    print(plate_reader)
