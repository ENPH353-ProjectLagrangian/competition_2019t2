#!/usr/bin/env python
import rospy as ros
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge

import tensorflow as tf
from tensorflow import keras, Graph, Session
import cv2
import numpy as np

from plate_isolator_colour import PlateIsolatorColour as PlateIsolator
from letter_isolator_beta import LetterIsolatorBeta as LetterIsolator
import global_variables as gv

class PlateReader():
    """
    Given an image, the PlateReader object will track license-parking pairs that it sees    
    Resources used: https://blog.victormeunier.com/posts/keras_multithread/
    """
    def __init__(self, num_model_path, char_model_path, certainty_thresh=0.8, 
                 multithreaded=True):
        """
        Initialises PlateReader object

        @param: num_model_path - path to keras ML model which IDs numbers
        @param: num_model_path - path to keras ML model which IDs letters
        """
        thread_graph = Graph()
        with thread_graph.as_default():
            self.thread_session = Session()
            with self.thread_session.as_default():
                self.num_model = keras.models.load_model(num_model_path)
                self.char_model = keras.models.load_model(char_model_path)
                self.graph = tf.get_default_graph()
        # if (multithreaded):
        #     self.num_model._make_predict_function()
        #     self.char_model._make_predict_function()

        # key: parking spot (int)
        # val: (license, confidence lvl)
        self.parking_license_pairs = {}

        self.certainty_thresh = certainty_thresh
        self.plate_isolator = PlateIsolator()
        self.letter_isolator = LetterIsolator()

    def process_image(self, img):
        """
        Finds parking number and license plate (if image contains them)
        Saves the parking number (key), the license plate and confidence (val)

        If a plate is read more than once, the higher confidence result is kept

        @param img - the image in which we look for plates
        @return None if certainty read < certainty or plates not found
                else:
                parking spot (int), license plate (str), model certainty (dec)
        """
        parking_plate, license_plate = \
            self.plate_isolator.extract_plates(img)

        if (parking_plate is not None):
            try:
                p, spot_num, l0, l1, n0, n1 = \
                    self.letter_isolator.get_chars(parking_plate,
                                                   license_plate)

                # Get each number/letter on plates.
                # If any certainty falls below the threshold,
                # save it, but return None so that the driving knows:
                # It Should Do Better
                parking_num, prob_parking_num = \
                    self._get_num_and_prob(spot_num)

                letter_left, prob_letter_left = \
                    self._get_char_and_prob(l0)
                letter_right, prob_letter_right = \
                    self._get_char_and_prob(l1)
                tens_digit, prob_tens_digit = \
                    self._get_num_and_prob(n0)
                ones_digit, prob_ones_digit = \
                    self._get_num_and_prob(n1)

                # we'll quantify the certainty as the sum of probabilities
                # that a given result is correct
                certainty = prob_letter_left + prob_letter_right + \
                    prob_tens_digit + prob_ones_digit

                license_text = letter_left + letter_right + \
                    str(tens_digit) + str(ones_digit)

                # replace previous guess for plate reading if our certainty
                # increased or, if we had no guess, make one
                if ((parking_num not in self.parking_license_pairs) or
                        (certainty >
                            self.parking_license_pairs[parking_num][1])):
                    self.parking_license_pairs[parking_num] = \
                        (license_text, certainty)

                # if any of those conditions apply: return None so user knows
                # that a better picture is needed
                if (prob_parking_num < self.certainty_thresh or
                    prob_letter_left < self.certainty_thresh or
                    prob_letter_right < self.certainty_thresh or
                    prob_tens_digit < self.certainty_thresh or
                        prob_ones_digit < self.certainty_thresh):
                    return None, None, None

                return parking_num, license_text, certainty
            except AssertionError as e:
                print(e)
                print("letters could not be extracted from image")
                return None, None, None
        else:
            print("No plates found in image")
            return None, None, None

    def _get_num_and_prob(self, img):
        img = self._preprocess_image(img)
        with self.graph.as_default():
            with self.thread_session.as_default():
                prediction = self.num_model.predict(img)
                num = np.argmax(prediction)
                return num, prediction[0, num]
        print("something horrible happened")
        return None, None

    def _get_char_and_prob(self, img):
        img = self._preprocess_image(img)
        with self.graph.as_default():
            with self.thread_session.as_default():
                prediction = self.char_model.predict(img)
                index = np.argmax(prediction)
                return chr(index + 65), prediction[0, index]
        print("something horrible happened")
        return None, None

    def _preprocess_image(self, img):
        WIDTH = 100
        HEIGHT = 150

        # make img 3 channel bc we need that for the model
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        # rescale if too large
        if (img.shape[1] > WIDTH or img.shape[0] > HEIGHT):
            rescale_ratio = min(WIDTH / img.shape[1], HEIGHT / img.shape[0])
            img = cv2.resize(img, (int(img.shape[1] * rescale_ratio),
                                   int(img.shape[0] * rescale_ratio)))

        # pad the sides
        y_border = max(0, HEIGHT - img.shape[0])
        x_border = max(0, WIDTH - img.shape[1])

        img = cv2.copyMakeBorder(img, 0, y_border, 0, x_border,
                                 cv2.BORDER_CONSTANT, value=255)

        # normalise!!!!
        img = img / float(255)

        return np.expand_dims(img, axis=0)

    def get_all_license(self, print_str=False):
        """
        Returns each parking spot, paired with its license text
        @returns a string describing the parking spot license pairs
        """
        out = ""
        for key, val in self.parking_license_pairs:
            out += 'Parking {}: {}\n'.format(key, val[0])
        if (print_str):
            print(out)
        return out


def test(path, plate_reader):
    img = cv2.imread(path)
    parking_spot, text, _ = plate_reader.process_image(img)
    if (parking_spot is not None):
        print("Parking: {}, {}".format(parking_spot, text))


def main():
    path = 'Preprocessing/IsolatingPlates/plate_images/'
    plate_reader = PlateReader('num_model.h5', 'char_model.h5')
    for i in range(48):
        print("image {}".format(i))
        test(path + 'image{}.png'.format(i), plate_reader)


if __name__ == "__main__":
    main()