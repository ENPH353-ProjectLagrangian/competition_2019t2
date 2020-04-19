#!/usr/bin/env python
import tensorflow as tf
from tensorflow import keras, Graph, Session
import cv2
import numpy as np

from plate_isolator_colour import PlateIsolatorColour as PlateIsolator
from letter_isolator_beta import LetterIsolatorBeta as LetterIsolator


class PlateReader():
    """
    Given an image, the PlateReader object will track license-parking pairs
    Resources used: https://blog.victormeunier.com/posts/keras_multithread/

    It will maintain an updated dictionary of the license plates at each parking location,
    saving the most likely license plate text

    Works in ROS callback functions (plate recognition models deal with in multi-threaded fashion)
    """
    def __init__(self, num_model_path, char_model_path, certainty_thresh=0.7):
        """
        Initialises PlateReader object

        @param: num_model_path - path to keras ML model which IDs numbers
        @param: char_model_path - path to keras ML model which IDs letters
        @param: certainty_thresh - necessary probability threshold (per char) for
                                   process_image method to consider a plate found
        """
        # saving the graph and thread session is necessary in the ROS framework
        thread_graph = Graph()
        with thread_graph.as_default():
            self.thread_session = Session()
            with self.thread_session.as_default():
                self.num_model = keras.models.load_model(num_model_path)
                self.char_model = keras.models.load_model(char_model_path)
                self.graph = tf.get_default_graph()

        self.parking_license_pairs = {}

        self.certainty_thresh = certainty_thresh
        self.plate_isolator = PlateIsolator()
        self.letter_isolator = LetterIsolator()

    def process_image(self, img):
        """
        Finds parking number and license plate (if image contains them)
        Saves the parking number (key), license plate object (val)

        If a plate is read more than once, the higher confidence result is kept

        @param img - the image in which we look for plates
        @return None, None, None (if certainty read < certainty or plates not found)
                parking spot (int), license plate (str), model certainty (dec) (otherwise)
        """
        parking_plate, license_plate = \
            self.plate_isolator.extract_plates(img)

        if (parking_plate is not None):
            try:
                p, p_n0, p_n1, l0, l1, n0, n1 = \
                    self.letter_isolator.get_chars(parking_plate,
                                                   license_plate)

                # Use our our ML models to ID number/letters
                p_n0, prob_p_n0 = \
                    self._get_num_and_prob(p_n0)
                p_n1, prob_p_n1 = \
                    self._get_num_and_prob(p_n1)
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
                certainty = prob_p_n0, prob_p_n1, \
                    prob_letter_left + prob_letter_right + \
                    prob_tens_digit + prob_ones_digit

                parking_num = 10 * p_n0 + p_n1

                license_text = letter_left + letter_right + \
                    str(tens_digit) + str(ones_digit)

                # assumption is that the parking # is correct
                if (parking_num not in self.parking_license_pairs):
                    # create license plate object if none exists at this parking #
                    self.parking_license_pairs[parking_num] = \
                        LicensePlate(letter_left, prob_letter_left,
                                     letter_right, prob_letter_right,
                                     tens_digit, prob_tens_digit,
                                     ones_digit, prob_ones_digit)
                else:
                    # update license plate object so each character in the plate
                    # has the highest probability read value saved
                    self.parking_license_pairs[parking_num].\
                        update(letter_left, prob_letter_left,
                               letter_right, prob_letter_right,
                               tens_digit, prob_tens_digit,
                               ones_digit, prob_ones_digit)

                # if any character recognition falls below certainty threshold
                # return None, None, None so that user recognise the image was not adequate
                if (prob_p_n0 < self.certainty_thresh or
                    prob_p_n1 < self.certainty_thresh or
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
            return None, None, None

    def _get_num_and_prob(self, img):
        """
        Identifies the image of number
        @param: img - the image of the single digit number
        @return: the predicted number, the probability of it being right
        """
        img = self._preprocess_image(img)
        with self.graph.as_default():
            with self.thread_session.as_default():
                prediction = self.num_model.predict(img)
                num = np.argmax(prediction)
                return num, prediction[0, num]
        print("something horrible happened")
        return None, None

    def _get_char_and_prob(self, img):
        """
        Identifies the image of character
        @param: img - the image of the character
        @return: the predicted character, the probability of it being right
        """
        img = self._preprocess_image(img)
        with self.graph.as_default():
            with self.thread_session.as_default():
                prediction = self.char_model.predict(img)
                index = np.argmax(prediction)
                return chr(index + 65), prediction[0, index]
        print("something horrible happened")
        return None, None

    def _preprocess_image(self, img):
        """
        Image preprocessing (same as preprocessing used to train model)
            1. standardises image size
            2. Makes image 3 channel (required by model)
            3. Normalises channels
            4. Packages image in format expected by keras model
        @param img - the image to proprocess
        @return the preprocessed image
        """
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

    def __str__(self):
        """
        Returns each parking spot, paired with its license text
        @returns a string describing the parking spot license pairs
        """
        out = ""
        for key, val in self.parking_license_pairs.items():
            out += 'Parking {}: {}\n'.format(key, val)

        return out


class LicensePlate():
    """
    Saves the content of the plate, as well as the probability
    with which each character was predicted by the model

    Updates to contain the highest probability guesses
    """
    def __init__(self, c0, p_c0, c1, p_c1, n0, p_n0, n1, p_n1):
        """
        Creates plate object
        @param: c0   - left character
        @param: p_c0 - probability of left character 
        @param: c1   - right character
        @param: p_c1 - probability of right character 
        @param: n0   - left digit
        @param: p_n0 - probability of left digit 
        @param: n1   - right digit
        @param: p_n1 - probability of right digit 
        """
        self.c0 = (c0, p_c0)
        self.c1 = (c1, p_c1)
        self.n0 = (n0, p_n0)
        self.n1 = (n1, p_n1)

    def update(self, c0, p_c0, c1, p_c1, n0, p_n0, n1, p_n1):
        """
        Updates plate object to have (in each position of the plate)
        the character or digit which was given with highest probability

        @param: c0   - updated guess for the left character
        @param: p_c0 - probability of left character
        @param: c1   - updated guess for the right character
        @param: p_c1 - probability of right character 
        @param: n0   - updated guess for the left digit
        @param: p_n0 - probability of left digit 
        @param: n1   - updated guess for the right digit
        @param: p_n1 - probability of right digit 
        """
        if (p_c0 > self.c0[1]):
            self.c0 = (c0, p_c0)
        if (p_c1 > self.c1[1]):
            self.c1 = (c1, p_c1)
        if (p_n0 > self.n0[1]):
            self.n0 = (n0, p_n0)
        if (p_n1 > self.n1[1]):
            self.n1 = (n1, p_n1)

    def __str__(self):
        """
        String representation of the plate
        @ return "<predicted plate text> (<certainty of prediction>)"
        """
        total_cert = self.c0[1] + self.c1[1] + self.n0[1] + self.n1[1]
        return '{}{}{}{} ({})'.format(self.c0[0], self.c1[0], self.n0[0],
                                      self.n1[0], total_cert)


def test(path, plate_reader):
    """
    Testing function called by main,
    which runs a given image through a plate_reader object

    @param path - path to image
    @paam plate_reader - PlateReader object to test
    """
    img = cv2.imread(path)
    parking_spot, text, _ = plate_reader.process_image(img)


def main():
    """
    Testing function for if the plate reader is ran alone
    @Pre: requires the path below to exist, and the models to be present
    """
    path = 'Preprocessing/IsolatingPlates/images_new_format/'
    plate_reader = PlateReader('num_model.h5', 'char_model_new.h5')
    for i in range(69, 73):
        # print("image {}".format(i))
        test(path + 'image{}.png'.format(i), plate_reader)
    print(str(plate_reader))


if __name__ == "__main__":
    """
    If run from the command line, run tests
    """
    main()
