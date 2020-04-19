#!/usr/bin/env python

import cv2


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
        @param: parking - image of parking plate
        @param: license - image of license plate
        @return the cutout and binarised images for:
                P (for training purposes), parking spot number (parking plate)
                Letter0, Letter1, Num0, Num1 (license plate)
        @throws: assertion error if we don't find the right # of chars.
        """
        parking_bin = self._binarise_plate(parking)
        license_bin = self._binarise_plate(license)
        self._display_test(parking_bin)
        self._display_test(license_bin)
        p, p_n0, p_n1 = self._get_chars_from_plate_hardcode(parking_bin, 3)
        self._display_test(p)
        self._display_test(p_n0)
        self._display_test(p_n1)
        self._display_test(license_bin)
        c0, c1, n0, n1 = self._get_chars_from_plate_hardcode(license_bin, 4)

        self._display_test(c0)
        self._display_test(c1)
        self._display_test(n0)
        self._display_test(n1)
        return p, p_n0, p_n1, c0, c1, n0, n1

# -------------------- Helpers -----------------------------
    def _binarise_plate(self, img):
        """
        Binarises the image to isolate plates
        @param: img - image to binarise
        @return: the binarised image (plate black, rest white)
        """
        grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.threshold(grey, self.bin_thresh_plate, 255,
                            cv2.THRESH_BINARY)[1]
        return cv2.morphologyEx(img, cv2.MORPH_OPEN, (6, 6))

    def _display_test(self, img, duration=3000):
        """
        Testing purposes only: displays an image for desired time
        @param: img - the image to display
        @param: duration - duration for which to display image
        """
        if (self.is_testing):
            cv2.imshow("testing", img)
            cv2.waitKey(duration)

    def _get_chars_from_plate_hardcode(self, img, expected_letters):
        """
        Extracts characters from the plate

        @param: img - image to extract characters from
        @param: expected_letters - number of expected letters
                                  (for parking: 3, for license: 4)
        @return: each individual letter, processed and scaled
        """
        w = img.shape[1]
        h = img.shape[0]
        y_top = h // 5
        y_bottom = 4 * h // 5
        if (expected_letters == 3):
            plate = self._scale_and_blur(img[y_top: y_bottom,
                                             1 * w // 5: 3 * w // 7])
            n0 = self._scale_and_blur(img[y_top: y_bottom,
                                          3 * w // 7: 17 * w // 28])
            n1 = self._scale_and_blur(img[y_top: y_bottom,
                                          17 * w // 28:4 * w // 5])
            return plate, n0, n1
        elif(expected_letters == 4):
            x_offset_outer = int(w * 0.35 / 5)
            x_offset_inner_left = int(w * 1.55 / 5)
            x_offset_inner_right = int(w * 1.35 / 5)
            c0 = self._scale_and_blur(img[y_top:y_bottom,
                                          x_offset_outer:x_offset_inner_left])
            c1 = self._scale_and_blur(img[y_top:y_bottom,
                                          x_offset_inner_left: w // 2])
            n0 = self._scale_and_blur(img[y_top:y_bottom,
                                      w // 2: w - x_offset_inner_right])
            n1 = self._scale_and_blur(img[y_top:y_bottom,
                                      w - x_offset_inner_right:
                                      w - x_offset_outer])
            return c0, c1, n0, n1
        else:
            raise ValueError("Expected number of letters must be 2 or 4")

    def _scale_and_blur(self, img):
        """
        Processes image to approximate size,
        adds slight blur to regularise,
        regularise with morphological transformations.

        @param: img - image to process
        @return: processed image
        """
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
