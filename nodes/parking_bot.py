#!/usr/bin/env python


import rospy as ros
import cv2
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import global_variables as gv

from plate_reader import PlateReader
from path_follower import PathFollower

bridge = CvBridge()


def get_image(imgmsg):
    """
    Converts a ros imgmsg to a open-cv format image.
    :param imgmsg: imgmsg
    :return: open-cv image
    """
    cv_image = bridge.imgmsg_to_cv2(imgmsg)
    cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
    return cv_image


class ParkingBot:
    """
    Overarching class for the parking bot.
    """

    def __init__(self, verbose=True, do_visualization=True):

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

        self.homography_path = gv.path + "/assets/homography-sim_v4.npy"
        self.shape_path = gv.path + "/assets/shape-sim_v4.npy"

        self.path_follower = PathFollower(self.homography_path, self.shape_path,
                                          verbose=self.verbose, do_visualization=self.do_visualization)
        """
        Halt condition parameters
        """
        self.min_plate_confidence = 3.9

        """
        Init the subscriber
        """
        self.sub = ros.Subscriber('/rrbot/camera1/image_raw', Image, self.process_image)

    def process_image(self, imgmsg):
        """
        Process the image from the robot to find license plates and control robot.
        Unsubscribes from image subscriber when all 6 plates are found

        :param imgmsg: ros format Image message
        """
        img = get_image(imgmsg)

        """
        Analyze image to determine movement of robot
        """
        self.path_follower.process_image(img)

        """
        Analyze image to find plates and parking numbers
        """
        parking, license_text, prob = self.plate_reader.process_image(img)

        """
        Build up a dictionary of stall numbers -> license plate pairs saving with probability
        """
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
        time = self.path_follower.time_stamp
        if (self.parking_dict.__len__() == 6 and min(probabilities) > self.min_plate_confidence) \
                or int(time.secs / 60) > 4:
            self.path_follower.publish_move(0, 0)

            self.sub.unregister()

            print("Completed in {} minutes and {} seconds".format(int(time.secs / 60),
                                                                  time.secs % 60))
            print(self.parking_dict)
            print("Done")


if __name__ == '__main__':
    parking_bot = ParkingBot(verbose=False, do_visualization=True)
    ros.spin()
