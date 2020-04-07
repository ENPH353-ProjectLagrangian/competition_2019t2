#!/usr/bin/env python3

import cv2
import numpy as np


class LetterIsolatorBeta:
    """
    Given coloured image of the parking and license plates
    Returns the binarised isolated letters

    Resources:
    https://stackoverflow.com/questions/34981144/split-text-lines-in-scanned-document
    https://www.pyimagesearch.com/2015/08/10/checking-your-opencv-version-using-python/
    """

    def __init__(self, bin_thresh_plate=60, testing=False):
        """
        @param binarisation_threshold: threshold which distinguishes between
                                       white and black in binary image
        @param img_width: width that input image gets scaled to
        """
        self.bin_thresh_plate = bin_thresh_plate
        self.is_testing = testing

    def get_chars(self, parking, license):
        """
        Given an image of a parking and license plate,
        returns each character as a separate image.
        @param img: greyscale image of the side of the car (with both plates)
        @return the cutout and binarised images for:
                P (for training purposes), parking spot number (parking plate)
                Letter0, Letter1, Num0, Num1 (license plate)
        @throws: assertion error if we don't find the right # of chars. Catch it, try again
        """
        parking_bin = self.binarise_plate(parking)
        license_bin = self.binarise_plate(license)
        self.display_test(parking_bin)
        self.display_test(license_bin)
        p, spot_num = self.get_chars_from_plate_hardcode(parking_bin, 2)
        self.display_test(p)
        self.display_test(spot_num)
        self.display_test(license_bin)
        c0, c1, n0, n1 = self.get_chars_from_plate_hardcode(license_bin, 4)

        self.display_test(c0)
        self.display_test(c1)
        self.display_test(n0)
        self.display_test(n1)
        return p, spot_num, c0, c1, n0, n1


# -------------------- Helpers -----------------------------

    def binarise_plate(self, img):
        """
        Binarises the image to isolate plates
        @param: input image
        @return: the binarised image (plate black, rest white)
        """
        grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.threshold(grey, self.bin_thresh_plate, 255,
                            cv2.THRESH_BINARY)[1]
        return cv2.morphologyEx(img, cv2.MORPH_OPEN, (6, 6))

    def display_test(self, img, duration=500):
        if (self.is_testing):
            cv2.imshow("testing", img)
            cv2.waitKey(duration)

    def get_chars_from_plate(self, img, expected_letters=2, thresh=200):
        """
        Returns an ordered list of letters imgs on plate (left to right)
        @param img - image from which letters are extracted
        @param expected_letters - number of expected letters
                                  (2 for parking, 4 for license plate)

        https://docs.opencv.org/3.4/da/d0c/tutorial_bounding_rects_circles.html
        """
        canny_out = cv2.Canny(img, thresh, 255)
        _, contours, _ = cv2.findContours(canny_out, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        contours_poly = [None]*len(contours)
        boundRect = [None]*len(contours)
        letter_rect = []
        width = img.shape[1]
        height = img.shape[0]

        # get all contours. Eliminate those which are the wrong size
        for i, c in enumerate(contours):
            contours_poly[i] = cv2.approxPolyDP(c, 3, True)
            boundRect[i] = cv2.boundingRect(contours_poly[i])
            if (width // 12 <= boundRect[i][2] <= width // 4
                    and height // 3 <= boundRect[i][3] <= 2 * height // 3):
                letter_rect.append(boundRect[i])

        list.sort(letter_rect, key=lambda rect: rect[0])

        # delete duplicates which are overlays of each other (take the larger area of the two)
        i = 1
        while i in range(1, len(letter_rect)):
            if (abs(letter_rect[i][0] - letter_rect[i - 1][0])
                    <= width // 12):
                if (letter_rect[i][2] * letter_rect[i][3] > letter_rect[i - 1][2] * letter_rect[i - 1][3]):
                    letter_rect[i - 1] = letter_rect[i]
                del letter_rect[i]
            else:
                i += 1

        # if self.is_testing:
        #     copy = img.copy()
        #     for i in range(len(letter_rect)):
        #         cv2.rectangle(img, (int(letter_rect[i][0]),
        #                             int(letter_rect[i][1])),
        #                            (int(letter_rect[i][0] + letter_rect[i][2]),
        #                             int(letter_rect[i][1] + letter_rect[i][3])),
        #                       (0), 2)
        #     for i in range(len(boundRect)):
        #         cv2.rectangle(copy, (int(boundRect[i][0]),
        #                             int(boundRect[i][1])),
        #                            (int(boundRect[i][0] + boundRect[i][2]),
        #                             int(boundRect[i][1] + boundRect[i][3])),
        #                       (0), 2)
        #     cv2.imshow("letter rect", img)
        #     cv2.imshow("bound rect", copy)
        #     print(boundRect)
        #     cv2.waitKey(2000)

        assert len(letter_rect) == expected_letters, \
            "letter_rect length: {}, {}".format(len(letter_rect), letter_rect)

        letters = [None] * len(letter_rect)
        for i, rect in enumerate(letter_rect):
            # print(rect)
            letters[i] = img[max(0, rect[1] - 2):min(img.shape[0] - 1, rect[1] + rect[3] + 2),
                             max(0, rect[0] - 2):min(img.shape[1] - 1, rect[0] + rect[2] + 2)]
        letters_new = [self._scale_and_blur(l) for l in letters]
        return letters_new

    def get_chars_from_plate_hardcode(self, img, expected_letters):
        w = img.shape[1]
        h = img.shape[0]
        y_top = h // 5
        y_bottom = 4 * h // 5
        if (expected_letters == 2):
            plate = self._scale_and_blur(img[y_top: y_bottom, 5 * w // 16: w // 2])
            license = self._scale_and_blur(img[y_top: y_bottom,
                                           w // 2: 11 * w // 16])
            return plate, license
        elif(expected_letters == 4):
            x_offset_outer = int(w * 0.35 / 5)
            x_offset_inner = int(w * 1.3 / 5)
            c0 = self._scale_and_blur(img[y_top:y_bottom,
                                          x_offset_outer:x_offset_inner])
            c1 = self._scale_and_blur(img[y_top:y_bottom,
                                          x_offset_inner: w // 2])
            n0 = self._scale_and_blur(img[y_top:y_bottom,
                                      w // 2: w - x_offset_inner])
            n1 = self._scale_and_blur(img[y_top:y_bottom,
                                      w - x_offset_inner: w - x_offset_outer])
            return c0, c1, n0, n1
        else:
            raise ValueError("Expected number of letters must be 2 or 4")

    def _scale_and_blur(self, img):
        MAX_IMG_WIDTH = 64

        width = img.shape[1]
        height = img.shape[0]
        scale_factor = MAX_IMG_WIDTH / width
        dim = (MAX_IMG_WIDTH, int(height * scale_factor))
        img = cv2.resize(img, dim)
        kernel = (5, 5)
        img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
        img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
        return cv2.GaussianBlur(img, (15, 15), 0)
