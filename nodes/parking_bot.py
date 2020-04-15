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
import pandas

bridge = CvBridge()


def compare_with_csv(discovered_plates):
    """
    Compares plates found to plates generated.
    :param discovered_plates: a list of plates discovered by the parking bot.
    :return: True if all plates discovered are in the generated plates csv.
    """
    generated_plates = pandas.read_csv(gv.path + "/../scripts/plates.csv")
    return all(plate in generated_plates for plate in discovered_plates)


def get_image(imgmsg):
    # Try to convert
    cv_image = bridge.imgmsg_to_cv2(imgmsg)
    cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
    return cv_image


class ParkingBot:

    def __init__(self, verbose=True, do_visualization=True, hough_line_threshold=100):
        self.sub = ros.Subscriber('/rrbot/camera1/image_raw', Image, self.process_image)

        """
        Boolean options for display and debugging.
        """
        self.verbose = verbose
        self.do_visualization = do_visualization

        """
        image processing
        """
        self.num_path = gv.path + '/assets/num_model_new.h5'
        self.char_path = gv.path + '/assets/char_model_new.h5'
        self.plate_reader = PlateReader(self.num_path, self.char_path)
        self.parking_dict = dict()

        """
        Ros topic publisher
        """
        ros.init_node('topic_publisher')
        self.pub = ros.Publisher('/cmd_vel', Twist, queue_size=1)
        self.move = Twist()

        """
        Load the homography for conversion to birds eye view.
        """
        self.td_homography = np.load(gv.path + "/assets/homography-sim_v4.npy")
        shape = np.load(gv.path + "/assets/shape-sim_v4.npy")

        self.td_height = shape[0]
        self.td_width = shape[1]
        """
        Path following parameters
        """
        self.hough_line_threshold = hough_line_threshold
        self.angular_gain = 1.4
        self.x_vel = 0.25
        self.alignment_tolerance = np.pi / 60

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

    def process_image(self, imgmsg):
        img = get_image(imgmsg)
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
        lines = ipu.get_hough_lines(img=edges, img_type="edges", threshold=self.hough_line_threshold)

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
        If set, undertake visualization
        """
        if self.do_visualization:
            self.visualize(crosswalk_center, lines, top_down, img)

        """
        Build up a dictionary of stall numbers -> license plate pairs saving with probability
        """
        parking, license_text, prob = self.plate_reader.process_image(img)

        if parking is not None:
            if parking not in self.parking_dict:
                self.parking_dict[parking] = (license_text, prob[2])
            elif self.parking_dict[parking][0] != license_text:
                if self.verbose:
                    print("Text reader conflict, taking the highest probability")
                    print('Former {}: {} ({})'.format(parking, self.parking_dict[parking][0],
                                                      self.parking_dict[parking][1]))
                    print('New {}: {} ({})'.format(parking, license_text, prob[2]))
                if self.parking_dict[parking][1] < prob[2]:
                    self.parking_dict[parking] = (license_text, prob[2])

        probabilities = list(self.parking_dict[parking][1] for parking in self.parking_dict)
        """
        If we have read 6 license plates and are confident in their readings we finish the run.
        """
        if self.parking_dict.__len__() == 6 and min(probabilities) > 3.99:
            self.move.linear.x = 0
            self.move.angular.z = 0
            self.pub.publish(self.move)

            self.sub.unregister()
            if compare_with_csv(list(self.parking_dict[parking_number][0] for parking_number in self.parking_dict)):
                print("Probable Success.")
            else:
                print("Failed.")

            print("Completed in {} minutes and {} seconds".format(int(self.time_stamp.secs / 60),
                                                                  self.time_stamp.secs % 60))
            print(self.parking_dict)
            print("Done")

    def crosswalk_reaction(self, crosswalk_center):
        if crosswalk_center is not None:
            if not self.stopping_for_crosswalk and self.wait_at_crosswalk \
                    and self.td_width * 1 / 3 <= crosswalk_center[0] <= self.td_width * 2 / 3 \
                    and crosswalk_center[1] >= self.td_height * 0.85:
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
        stop for the crosswalk
        """
        self.move.linear.x = 0
        self.move.angular.z = 0
        self.pub.publish(self.move)
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
            print("movement: " + str(movement) + ", " + str(image_difference))

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
            print("waiting for movement: " + str(self.wait_for_movement_on_crosswalk))

            if self.wait_for_movement_on_crosswalk and movement:
                self.wait_for_movement_on_crosswalk = False
            elif not self.wait_for_movement_on_crosswalk and not movement:
                # If no movement is detected we move past the crosswalk and we aren't waiting for movement
                self.reset_crosswalk_fields()
                self.left_cross_walk_time = ros.get_rostime()

                # Gun it through the crosswalk
                self.x_vel = 0.8
                self.angular_gain = 4
                print("Passing crosswalk")

    def reset_crosswalk_fields(self):
        self.stopping_for_crosswalk = False
        self.wait_at_crosswalk = False
        self.last_difference_check_time = None
        self.last_crosswalk_image = None
        self.wait_for_movement_on_crosswalk = None

    def path_following_procedure(self, lines):
        """

        :param lines:
        :return:
        """

        """
        Process lines into clusters and headings based on angles
        """
        angles = lines[:, 0, 1]
        angles = np.array(list([[-angle, np.pi - angle][angle > np.pi / 2]] for angle in angles), dtype=np.float32)

        headings, cluster_indexes = ipu.find_cardinal_clusters(angles)

        clusters = [np.array([lines[i] for i in cluster_indexes[0]]),
                    np.array([lines[i] for i in cluster_indexes[1]])]

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


if __name__ == '__main__':
    parking_bot = ParkingBot(verbose=False, do_visualization=True)
    ros.spin()
