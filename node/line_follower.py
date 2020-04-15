#!/usr/bin/env python
"""
TODO:
Toon pedestrian turning
"""
import image_processing_utils as ipu
import numpy as np
import rospy as ros
import cv2
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from geometry_msgs.msg import Twist
import global_variables as gv

from plate_reader import PlateReader

bridge = CvBridge()


def get_image(imgmsg):
    # Try to convert
    cv_image = bridge.imgmsg_to_cv2(imgmsg)
    cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
    return cv_image


class ParkingBot:

    def __init__(self, verbose=True, visualize=True, hough_line_threshold=100):
        # Boolean options
        self.verbose = verbose
        self.visualize = visualize

        # image processing
        self.num_path = gv.path + '/assets/num_model_new.h5'
        self.char_path = gv.path + '/assets/char_model_new.h5'
        self.plate_reader = PlateReader(self.num_path, self.char_path)

        # Ros topic publsi

        ros.init_node('topic_publisher')
        self.period = ros.Rate(50)

        self.pub = ros.Publisher('/cmd_vel', Twist, queue_size=1)
        self.move = Twist()

        # Load the homography to convert to birds eye view.
        self.td_homography = np.load(gv.path + "/assets/homography-sim_v4.npy")
        shape = np.load(gv.path + "/assets/shape-sim_v4.npy")

        self.td_height = shape[0]
        self.td_width = shape[1]

        # Threshold for detecting lines
        self.hough_line_threshold = hough_line_threshold

        # Path following
        self.angular_gain = 1.25
        self.centering_gain = 5
        self.x_vel = 0.25
        self.heading = 0
        self.skip = 0
        self.index_length = None
        self.distance_since_last_turn = 0
        self.time_stamp = ros.get_rostime()

        # Cross walk stuff
        self.img_delta = 0.1
        self.reached_crosswalk_time = None
        self.wait_at_crosswalk = False
        self.old_img = None
        self.last_difference_check_time = None
        self.stopping_for_crosswalk = False
        self.left_cross_walk_time = None
        self.holding = None

    def process_image(self, imgmsg):
        img = get_image(imgmsg)

        parking, license_text, prob = self.plate_reader.process_image(img)

        if (parking is not None):
            print('Parking {}: {} ({})'.format(parking, license_text, prob))

        # Generates the top down view that is analyzed.

        top_down = cv2.warpPerspective(img, self.td_homography, (self.td_width, self.td_height))

        # Find crosswalks
        center = self.find_crosswalk(top_down)

        # Stops at the cross walk if the further cross walk border is the bottom center of the image.
        if center is not None:
            if not self.stopping_for_crosswalk and self.wait_at_crosswalk \
                    and self.td_width * 1 / 3 <= center[0] <= self.td_width * 2 / 3 \
                    and center[1] >= self.td_height * 4 / 5:
                if self.reached_crosswalk_time is None:
                    self.reached_crosswalk_time = ros.get_rostime()

                self.stopping_for_crosswalk = True

        else:
            self.wait_at_crosswalk = True
            if self.left_cross_walk_time is not None \
                    and ros.get_rostime() - self.left_cross_walk_time > ros.Duration(1.5):
                self.angular_gain = 1.25
                self.x_vel = 0.25
                self.left_cross_walk_time = None

        td_gray = cv2.cvtColor(top_down, cv2.COLOR_BGR2GRAY)

        ret, thresh = cv2.threshold(td_gray, 210, 255, cv2.THRESH_BINARY)
        edges = cv2.Canny(thresh, 50, 150, apertureSize=3)
        lines = ipu.get_hough_lines(img=edges, img_type="edges", threshold=self.hough_line_threshold)

        if self.stopping_for_crosswalk:
            self.move.linear.x = 0
            self.move.angular.z = 0
            time_since_stop = ros.get_rostime() - self.reached_crosswalk_time

            processed_image = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            processed_image = cv2.blur(processed_image, (15, 15))

            if self.last_difference_check_time is None or self.old_img is None:
                self.last_difference_check_time = ros.get_rostime()
                self.old_img = processed_image
            else:
                if ros.get_rostime() - self.last_difference_check_time > ros.Duration(0.05):
                    image_difference = np.average(cv2.absdiff(processed_image, self.old_img))
                    self.old_img = processed_image
                    self.last_difference_check_time = ros.get_rostime()
                    movement = image_difference >= self.img_delta
                    if self.holding is None:
                        # If the first time we measure movement we see that there is none,
                        # we want to wait for the pedestrian to cross once.
                        # If we see that the pedestrian is already moving we just need to wait for it to cross.
                        self.holding = not movement

                    if time_since_stop > ros.Duration(5) or not self.holding:
                        if not movement:
                            # If no movement is detected we move past the crosswalk.
                            self.reset_crosswalk_fields()

                            self.left_cross_walk_time = ros.get_rostime()

                            # Gun it through the crosswalk
                            self.x_vel = 0.8
                            self.angular_gain = 4

        elif lines is not None:
            """
            If no crosswalk is detected we maybe process lines to find headings and corners.
            """

            """
            Process lines into clusters and headings based on angles
            """
            angles = lines[:, 0, 1]
            angles = np.array(list([[-angle, np.pi - angle][angle > np.pi / 2]] for angle in angles), dtype=np.float32)

            headings, cluster_indexes = ipu.find_cardinal_clusters(angles)

            clusters = [np.array([lines[i] for i in cluster_indexes[0]]),
                        np.array([lines[i] for i in cluster_indexes[1]])]

            #  There is an ambiguity in which direction is "forward" when theta is close to pi/2.
            #  -1 is right turn, 1 is left turn.
            turn_bias = -1  # Starts off at -1 could be changed by advanced algorithms
            # but probably shouldn't for this naive approach.

            # The margin of what is considered "close" to an angle
            tolerance = np.pi / 60  # 3 degrees

            head_abs = np.abs(headings[1])
            if np.abs(head_abs - 1.57) < tolerance:
                headings[1] = turn_bias * head_abs

            """
            Choose the heading we wish to use as a guide
            """
            if self.verbose:
                print("The most recent heading was: " + str(self.heading))

            # Chooses the heading closest to the previous heading to follow
            aligned_index = (np.abs(headings - self.heading)).argmin()

            """
            Generates states based on positions of parallel and perpendicular lines.
            """
            alignment_tolerance = np.pi / 60

            aligned = headings[aligned_index][0] < alignment_tolerance

            if aligned:  # Checks to see if robot is aligned
                state_x, state_y = self.get_states(clusters)
            else:
                state_x = 3 * [0]
                state_y = 5 * [0]

            """
            Use states to determine if we are at a corner or intersection.
            """
            turn = (state_y == [0, 0, 0, 0, 1] and state_x[1] == 0) or state_y == [1, 0, 1, 0, 1]

            # The first corner defines the index count for long stretches of road.
            if turn and self.index_length is None:
                self.index_length = self.distance_since_last_turn

            # We want a cooldown before we turn again
            turn = turn and self.distance_since_last_turn > self.index_length * 1 / 3.0

            if not turn:
                self.heading = headings[aligned_index][0]
            else:

                if self.distance_since_last_turn <= self.index_length * 2 / 3.0:
                    if self.skip == 2:
                        self.skip = -1

                        self.heading = headings[aligned_index][0]
                    else:
                        self.skip += 1
                        self.heading = headings[int(not aligned_index)][0]
                else:
                    self.heading = headings[int(not aligned_index)][0]

                self.distance_since_last_turn = 0

            if self.verbose:
                print("The current primary directions are {0!s} the new heading is {1!s}.".format(headings,
                                                                                                  self.heading))

            error = self.heading

            self.move.angular.z = self.angular_gain * error
            self.move.linear.x = self.x_vel
        else:
            # If it can't find any lines then
            self.move.linear.x = self.x_vel
            self.move.angular.z = self.angular_gain * self.heading

        if self.visualize:
            # Draw crosswalk markers
            if center is not None:
                cv2.drawMarker(top_down, (center[0], center[1]), (255, 0, 0), markerType=cv2.MARKER_STAR,
                               markerSize=40, thickness=2, line_type=cv2.LINE_AA)

            # Draw the lines if possible
            if lines is not None:
                ipu.draw_lines(top_down, lines)

            # Draw grid lines
            grid_lines = []
            for i in range(1, 3):
                grid_lines.append([[int(i / 3.0 * self.td_width), 0]])
            for i in range(1, 5):
                grid_lines.append([[int(i / 5.0 * self.td_height), np.pi / 2]])
            ipu.draw_lines(top_down, grid_lines, color=(0, 255, 0))

            # Display the images
            cv2.imshow('top down', top_down)
            cv2.imshow("image", img)

            cv2.waitKey(1)

        time_step = ros.get_rostime() - self.time_stamp

        self.distance_since_last_turn += self.move.linear.x * time_step.nsecs * 1e-9
        self.time_stamp = ros.get_rostime()

        self.pub.publish(self.move)

    def reset_crosswalk_fields(self):
        self.stopping_for_crosswalk = False
        self.wait_at_crosswalk = False
        self.reached_crosswalk_time = None
        self.last_difference_check_time = None
        self.old_img = None
        self.holding = None

    def find_crosswalk(self, img):
        hsv_img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        lower_red = np.array([110, 150, 150])
        upper_red = np.array([130, 255, 255])
        # Threshold the HSV image to get only blue colors
        mask = cv2.inRange(hsv_img, lower_red, upper_red)

        if mask.max() == 0:
            return None
        else:
            moments = cv2.moments(mask, binaryImage=True)
            center = (int(moments["m10"] / moments["m00"]), int(moments["m01"] / moments["m00"]))
            return center

    def get_states(self, clusters):
        state_x = 3 * [0]
        state_y = 5 * [0]
        for line in clusters[0]:
            rho = line[0][0]
            i = int(rho / self.td_width * 3)
            i = max(i, 0)
            i = min(i, 2)
            state_x[i] = 1

        for line in clusters[1]:
            rho = line[0][0]
            i = int(rho / self.td_height * 5)
            i = max(i, 0)
            i = min(i, 4)
            state_y[i] = 1

        return state_x, state_y


if __name__ == '__main__':
    parking_bot = ParkingBot(verbose=False, visualize=True)

    ros.Subscriber('/rrbot/camera1/image_raw', Image, parking_bot.process_image)
    ros.spin()
