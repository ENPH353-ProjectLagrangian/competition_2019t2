#!/usr/bin/env python

import image_processing_utils as ipu
import numpy as np
import rospy as ros
import cv2
from geometry_msgs.msg import Twist
import global_variables as gv


class PathFollower:
    """
    Path follower will direct movement of ros robot by analyzing given image.
    """

    def __init__(self, homography_path=gv.path + "/assets/homography-sim_v4.npy",
                 shape_path=gv.path + "/assets/homography-sim_v4.npy",
                 hough_line_threshold=100, alignment_tolerance=np.pi / 60,
                 def_angular_gain=1.4, def_x_vel=0.25, cw_angular_gain=4, cw_x_vel=0.8,
                 verbose=True, do_visualization=True):
        """

        :param homography_path: Path to .npy array representing top down homography matrix
        :param shape_path: Path to .npy array representing shape of top down image.
        :param hough_line_threshold: Threshold for detecting lines in image
        :param alignment_tolerance: tolerance for when the robot is considered alligned with the road in radians.
        :param def_angular_gain: The default angular gain when moving on road.
                                 Angular gain acts proportionally on the alignment error
                                 to produce the angular velocity
        :param def_x_vel: The default forward velocity of the robot
        :param cw_angular_gain: The angular gain of the robot when passing through a crosswalk.
                                Angular gain acts proportionally on the alignment error
                                to produce the angular velocity
        :param cw_x_vel: The forward velocity of the robot when passing through a crosswalk
        :param verbose: If true the robot will print messages relating to its status to the console
        :param do_visualization: If true will display a visualization of what it sees.
        """
        """
        Boolean options for display and debugging.
        """
        self.verbose = verbose
        self.do_visualization = do_visualization

        """
        Ros topic publisher
        """
        ros.init_node('topic_publisher')
        self.pub = ros.Publisher('/cmd_vel', Twist, queue_size=1)
        self.move = Twist()

        """
        Load the homography for conversion to birds eye view.
        """
        self.td_homography = np.load(homography_path)
        shape = np.load(shape_path)

        self.td_height = shape[0]
        self.td_width = shape[1]
        """
        Path following parameters
        """
        self.hough_line_threshold = hough_line_threshold

        # Velocity and gain in normal behaviour
        self.def_angular_gain = def_angular_gain
        self.def_x_vel = def_x_vel

        # Velocity and gain in crosswalks
        self.cw_angular_gain = cw_angular_gain
        self.cw_x_vel = cw_x_vel

        self.angular_gain = def_angular_gain
        self.x_vel = def_x_vel
        self.alignment_tolerance = alignment_tolerance

        """
        Path following declaration and initialization
        """
        self.heading = 0
        self.skip = 0
        self.index_length = None
        self.distance_since_last_turn = 0
        self.time_stamp = ros.get_rostime()

        """
        Cross walk declaration and initialization
        """
        # Threshold for detecting movement in an image.
        self.img_delta = 0.1
        # Whether the robot should ignore a crosswalk detection.
        self.wait_at_crosswalk = False
        # Storing the previous crosswalk image for movement detecting
        self.last_crosswalk_image = None
        # When was the last time we checked for movement
        self.last_difference_check_time = None
        # Is the robot currently stopping for a crosswalk
        self.stopping_for_crosswalk = False
        # Time we left the crosswalk
        self.left_cross_walk_time = None
        # Is the bot waiting for a pedestrian to begin moving
        self.wait_for_movement_on_crosswalk = None

    def publish_move(self, linear, angular):
        """
        Publishes a move command to to robot
        :param linear: linear x-velocity
        :param angular: angular z-velocity
        """

        self.move.linear.x = linear
        self.move.angular.z = angular
        self.pub.publish(self.move)

    def process_image(self, img):
        """
        Processes image to direct movement of robot.
        :param img: image to be processed
        """
        # Generates the top down view that is analyzed.

        top_down = cv2.warpPerspective(img, self.td_homography, (self.td_width, self.td_height))

        """
        Crosswalk detection and reaction
        """

        # Find crosswalks
        crosswalk_center = ipu.find_crosswalk(top_down)

        # Stops at the cross walk if the further cross walk border is the bottom center of the image.
        self.crosswalk_reaction(crosswalk_center)

        td_gray = cv2.cvtColor(top_down, cv2.COLOR_BGR2GRAY)

        ret, thresh = cv2.threshold(td_gray, 210, 255, cv2.THRESH_BINARY)
        edges = cv2.Canny(thresh, 50, 150, apertureSize=3)

        # Use the Hough Line transform to find straight edges on the top down image. Lines are in Hesse Normal Form see:
        # https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_houghlines/py_houghlines.html

        lines = cv2.HoughLines(edges, 1, np.pi / 180, self.hough_line_threshold)

        if self.stopping_for_crosswalk:
            """
            If crosswalk is detected we stop for the crosswalk and analyze movement to see when it is safe to pass.
            """
            self.crosswalk_stop_procedure(img)

        elif lines is not None:
            """
            If no crosswalk is detected we may process lines to find headings and corners.
            """
            self.path_following_procedure(lines)
        else:
            """
            If no lines are detected then we continue on current course.
            """
            self.move.linear.x = self.x_vel
            self.move.angular.z = self.angular_gain * self.heading

        """
        Track distance travelled and publish move order
        """
        time_step = ros.get_rostime() - self.time_stamp

        self.distance_since_last_turn += self.move.linear.x * time_step.nsecs * 1e-9
        self.time_stamp = ros.get_rostime()

        self.pub.publish(self.move)

        """
        If set, visualize what the robot sees.
        """
        if self.do_visualization:
            self.visualize(crosswalk_center, lines, top_down, img)

    def crosswalk_reaction(self, crosswalk_center):
        """
        Determines the robots reaction to seeing or not seeing a crosswalk.
        :param crosswalk_center: The center of the crosswalk
        """
        if crosswalk_center is not None:
            if not self.stopping_for_crosswalk and self.wait_at_crosswalk \
                    and self.td_width * 1 / 3 <= crosswalk_center[0] <= self.td_width * 2 / 3 \
                    and crosswalk_center[1] >= self.td_height * 0.85:
                """
                If we have just seen a crosswalk close to the front of the robot and are not already stopping or passing 
                one, set the robots state so that it stops for the crosswalk.
                """
                self.stopping_for_crosswalk = True

        else:
            """
            If we have left the vicinity of the last crosswalk we want to return to normal behaviour.
            """
            self.wait_at_crosswalk = True
            if self.left_cross_walk_time is not None \
                    and ros.get_rostime() - self.left_cross_walk_time > ros.Duration(1.5):
                self.angular_gain = 1.4
                self.x_vel = 0.25
                self.left_cross_walk_time = None

    def crosswalk_stop_procedure(self, img):

        """
        Crosswalk stop procedure. Determines whether it is safe to ignore the crosswalk and continue passed it.
        :param img: Image to analyze for whether it is safe to cross the crosswalk
        """
        """
        Stop for the crosswalk
        """
        self.move.linear.x = 0
        self.move.angular.z = 0
        self.pub.publish(self.move)

        """
        Further process the image to detect movement
        """
        processed_image = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        processed_image = cv2.blur(processed_image, (15, 15))

        if self.last_difference_check_time is None:
            # Unless something has gone horribly wrong these should all be None at the same times.
            self.last_difference_check_time = ros.get_rostime()
            # We need it to slow down
            ros.sleep(1)

        elif ros.get_rostime() - self.last_difference_check_time > ros.Duration(0.05):
            if self.last_crosswalk_image is None:
                self.last_crosswalk_image = processed_image

            """
            Calculate the image difference
            """
            image_difference = np.average(cv2.absdiff(processed_image, self.last_crosswalk_image))

            # Determine if there is movement
            movement = image_difference >= self.img_delta

            self.last_crosswalk_image = processed_image
            self.last_difference_check_time = ros.get_rostime()

            # Initialize self.wait_for_movement
            if self.wait_for_movement_on_crosswalk is None:
                """
                If the first time we measure movement we see that there is none, we want to wait for the pedestrian 
                to begin moving again.

                If we see that the pedestrian is already moving we just need to wait for it to cross.
                """
                self.wait_for_movement_on_crosswalk = not movement

            """
            We can stop waiting for movement once we see movement again
            """
            if self.verbose and self.wait_for_movement_on_crosswalk:
                print("Waiting for movement")

            if self.wait_for_movement_on_crosswalk and movement:
                self.wait_for_movement_on_crosswalk = False
            elif not self.wait_for_movement_on_crosswalk and not movement:
                # If no movement is detected we move past the crosswalk and we aren't waiting for movement
                self.reset_crosswalk_fields()
                self.left_cross_walk_time = ros.get_rostime()

                # Gun it through the crosswalk
                self.x_vel = 0.8
                self.angular_gain = 4

                if self.verbose:
                    print("Passing crosswalk")

    def reset_crosswalk_fields(self):
        """
        Reset fields related to the crosswalk management algorithm for the next time we see a crosswalk
        """
        self.stopping_for_crosswalk = False
        self.wait_at_crosswalk = False
        self.last_difference_check_time = None
        self.last_crosswalk_image = None
        self.wait_for_movement_on_crosswalk = None

    def path_following_procedure(self, lines):
        """
        Directs the movement of the robot when we are not stopping at a crosswalk. Inlcuding allingment with road
        and when to turn.
        :param lines: List of lines that the robot sees on the top_down image.
        """

        """
        Process lines into clusters and headings based on angles
        """
        angles = lines[:, 0, 1]
        angles = np.array(list([[-angle, np.pi - angle][angle > np.pi / 2]] for angle in angles), dtype=np.float32)

        headings, cluster_indices = ipu.find_cardinal_clusters(angles)

        clusters = [np.array([lines[i] for i in cluster_indices[0]]),
                    np.array([lines[i] for i in cluster_indices[1]])]

        """
        Choose the heading we wish to use as a guide
        """
        if self.verbose:
            print("The most recent heading was: " + str(self.heading))

        # Chooses the heading closest to the previous heading as the one we wish to align with.
        alignment_index = (np.abs(headings - self.heading)).argmin()

        """
        Generates states based on positions of parallel and perpendicular lines.
        """
        aligned = headings[alignment_index][0] < self.alignment_tolerance

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
            self.heading = headings[alignment_index][0]
        else:

            if self.distance_since_last_turn <= self.index_length * 2 / 3.0:
                if self.skip == 2:
                    self.skip = -1

                    self.heading = headings[alignment_index][0]
                else:
                    self.skip += 1
                    self.heading = headings[int(not alignment_index)][0]
            else:
                self.heading = headings[int(not alignment_index)][0]

            self.distance_since_last_turn = 0

        if self.verbose:
            print("The current primary directions are {0!s} the new heading is {1!s}.".format(headings,
                                                                                              self.heading))

        error = self.heading

        self.move.angular.z = self.angular_gain * error
        self.move.linear.x = self.x_vel

    def get_states(self, clusters):
        """
        Reduce the "state" of the robots environment to two binary lists.

        :param clusters: tuple with list of vertical lines and list of horizontal lines.
        :return: state_x represents regions where vertical lines cross horizontal axis.
                         Regions are each 1/3th of the topdown image width.
                 state_y represents regions where horizontal lines cross vertical axis.
                         Regions are each 1/5th of the topdown image height.
        """
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

    def visualize(self, crosswalk_center, lines, top_down, img):
        # Draw crosswalk markers
        if crosswalk_center is not None:
            cv2.drawMarker(top_down, (crosswalk_center[0], crosswalk_center[1]), (255, 0, 0),
                           markerType=cv2.MARKER_STAR, markerSize=40, thickness=2, line_type=cv2.LINE_AA)

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
