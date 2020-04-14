#!/usr/bin/env python

import cv2
import numpy as np
from scipy.spatial import distance as dist


def _contour_length_tuple(c):
    return cv2.arcLength(c[0], True)


class PlateIsolatorColour:
    """
    The goal of this module is to pick out parking
    1. will pull out cars by colour

    Input: a clean image of the plates
           Random, likely terrible images picked up from Anki camera

    Resources:
    https://www.pyimagesearch.com/2014/08/04/opencv-python-color-detection/
    """

    def __init__(self, colour_bounds=None, testing=False):
        """
        Sets up our sift recognition based off of our pattern
        """
        # in order HSB, green, blue, yellow
        if colour_bounds is None:
            self.colour_bounds = [
                ([50, 0, 0], [80, 255, 255]),
                ([100, 130, 50], [120, 255, 170]),
                ([30, 0, 0], [40, 255, 255])
            ]
        else:
            self.colour_bounds = colour_bounds

        self.testing = testing

    def extract_plates(self, img, duration=1000):
        """
        Returns plates in order: parking, license, or None if no plates found
        """
        hsb = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        car_mask, car_colour = self.get_car_mask(hsb)
        if car_mask is None:
            if self.testing:
                print("no car found")
                cv2.imshow("image", img)
                cv2.waitKey(duration)
            return None, None

        parking_corners, license_corners = self.get_plate_corners(hsb, car_mask, car_colour)
        if (parking_corners is None or license_corners is None):
            if self.testing:
                cv2.imshow("image", img)
                cv2.imshow("mask", car_mask)
                cv2.waitKey(duration)
                print("no plate found")
            return None, None
        parking = self.cropped_image(img, parking_corners)
        license = self.cropped_image(img, license_corners)
        if self.testing:
            cv2.imshow("image", img)
            cv2.imshow("parking", parking)
            cv2.imshow("license", license)
            cv2.waitKey(duration)
            # print("success")

        return parking, license

    def get_car_mask(self, img, duration=3000):
        bound_num = 0

        mask = [None, None, None]

        # mask by each possible car colour
        for (lower, upper) in self.colour_bounds:
            # create numpy arrays from colour boundaries
            lower = np.array(lower, dtype="uint8")
            upper = np.array(upper, dtype="uint8")

            # find colours in the range, apply mask
            if self.testing:
                if (bound_num == 0):
                    title = "green"
                elif (bound_num == 1):
                    title = "blue"
                else:
                    title = "yellow"

            mask[bound_num] = cv2.inRange(img, lower, upper)

            # if self.testing:
            #     output = cv2.bitwise_and(img, img, mask=mask[bound_num])
            #     cv2.imshow(title, np.hstack([img, output]))
            #     cv2.waitKey(duration)

            bound_num += 1
        # Get the mask that had the largest contour (largest car), and
        # the contour of said car
        used_mask, car_contour = self.car_contour(mask)

        if car_contour is None:
            return None, None
        car_mask = np.zeros((img.shape[0], img.shape[1]), np.uint8)
        cv2.drawContours(car_mask, [car_contour], -1, (255), -1)

        if self.testing:
            cv2.imshow("image", img)
            cv2.imshow("car mask", car_mask)
            cv2.waitKey(duration)

        return car_mask, used_mask

    def get_plate_corners(self, img, mask, colour):
        """
        Returns 4 corners of the plates (with perspective still) as an np array
        Parking plate first, then license plate
        @param img - img in which we search for plate
        @param mask - the mask of the car
        @param colour - the colour of the car (that will also be part of
                        contour and thus should be filtered)

        Reference: https://stackoverflow.com/questions/44127342/detect-card-minarea-quadrilateral-from-contour-opencv
        """
        # 1. Get mask of ONLY the plates
        (lower, upper) = self.colour_bounds[colour]
        # create numpy arrays from colour boundaries
        lower = np.array(lower, dtype="uint8")
        upper = np.array(upper, dtype="uint8")
        colour_mask = cv2.inRange(img, lower, upper)
        colour_mask = cv2.bitwise_not(colour_mask)
        plate_mask = cv2.bitwise_and(colour_mask, colour_mask, mask=mask)
        if (self.testing):
            cv2.imshow("colour_mask", colour_mask)
            cv2.waitKey(2000)

        # 2. Use mask to get contours
        plate_mask = cv2.GaussianBlur(plate_mask, (5, 5), 1)  # regularise
        _, contours, _ = cv2.findContours(plate_mask, cv2.RETR_TREE,
                                          cv2.CHAIN_APPROX_SIMPLE)

        # 3. Discard contours with too small area
        MIN_AREA = int(0.75 * plate_mask.shape[0] / 10
                       * plate_mask.shape[1] / 10)  # experimentally determined
        good_contours = [c for c in contours if cv2.contourArea(c) > MIN_AREA]
        if (len(good_contours) != 2):
            return None, None

        # 4. pick out parking and license plates
        parking_contour = good_contours[0] \
            if good_contours[0][0, 0, 1] < good_contours[1][0, 0, 1] \
            else good_contours[1]
        license_contour = good_contours[1] \
            if good_contours[0][0, 0, 1] < good_contours[1][0, 0, 1] \
            else good_contours[0]

        # 4. Make a quadrilateral around the plate
        # (get a few tries w/ increasing precision)
        RETRIES_ALLOWED = 2
        tries = 0
        parking_poly = []
        license_poly = []

        while (tries < RETRIES_ALLOWED and len(parking_poly) != 4):
            precision = cv2.arcLength(parking_contour, True) // (8 + 2 * tries)
            parking_poly = cv2.approxPolyDP(parking_contour, precision, True)
            tries += 1

        tries = 0
        while (tries < RETRIES_ALLOWED and len(license_poly) != 4):
            precision = cv2.arcLength(parking_contour, True) // (8 + 2 * tries)
            license_poly = cv2.approxPolyDP(license_contour, precision, True)
            tries += 1

        # only works with quadrilaterals! Otherwise error happened
        if (len(parking_poly) != 4 or len(license_poly) != 4):
            print("length not 4!")
            return None, None

        if self.testing:
            parking_mask = np.zeros((img.shape[0], img.shape[1]), np.uint8)
            cv2.drawContours(parking_mask, [parking_poly], -1, (255))
            license_mask = np.zeros((img.shape[0], img.shape[1]), np.uint8)
            cv2.drawContours(license_mask, [license_poly], -1, (255))

            cv2.imshow("parking_mask", parking_mask)
            cv2.imshow("license_mask", license_mask)
            cv2.imshow("img", img)
            cv2.imshow("plate mask", plate_mask)
            cv2.waitKey(2000)

        return parking_poly[:, 0, :], license_poly[:, 0, :]

    def car_contour(self, mask):
        _, contours0, _ = cv2.findContours(mask[0], cv2.RETR_TREE,
                                           cv2.CHAIN_APPROX_SIMPLE)
        _, contours1, _ = cv2.findContours(mask[1], cv2.RETR_TREE,
                                           cv2.CHAIN_APPROX_SIMPLE)
        _, contours2, _ = cv2.findContours(mask[2], cv2.RETR_TREE,
                                           cv2.CHAIN_APPROX_SIMPLE)
        """
        guestimate area: experimentally determined
        """
        MIN_AREA = (int)(0.75 * mask[0].shape[1] / 6 * mask[0].shape[0] / 4)

        cont_0 = [(c, 0) for c in contours0 if cv2.contourArea(c) > MIN_AREA]
        cont_1 = [(c, 1) for c in contours1 if cv2.contourArea(c) > MIN_AREA]
        cont_2 = [(c, 2) for c in contours2 if cv2.contourArea(c) > MIN_AREA]

        good_contours = [c for c in (cont_0 + cont_1 + cont_2)
                         if (cv2.contourArea(c[0]) > MIN_AREA)]

        list.sort(good_contours, key=_contour_length_tuple)
        if (len(good_contours) == 0):
            return None, None

        car_contour = good_contours[0][0]
        car_colour = good_contours[0][1]

        if (len(good_contours) > 1 and self._contour_on_edge(car_contour,
                                                             mask[0])):
            car_contour = good_contours[1][0]
            car_colour = good_contours[1][1]

        return car_colour, car_contour

    def cropped_image(self, img, corners):
        ordered_corners = self._order_corners(corners)
        return self._four_point_transform(img, ordered_corners)

    def _four_point_transform(self, img, ordered_corners):
        # obtain a consistent order of the points and unpack them
        # individually
        (tl, tr, br, bl) = ordered_corners
        # compute the width of the new image, which will be the
        # maximum distance between bottom-right and bottom-left
        # x-coordiates or the top-right and top-left x-coordinates
        widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
        widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
        maxWidth = max(int(widthA), int(widthB))
        # compute the height of the new image, which will be the
        # maximum distance between the top-right and bottom-right
        # y-coordinates or the top-left and bottom-left y-coordinates
        heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
        heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
        maxHeight = max(int(heightA), int(heightB))
        # now that we have the dimensions of the new image, construct
        # the set of destination points to obtain a "birds eye view",
        # (i.e. top-down view) of the image, again specifying points
        # in the top-left, top-right, bottom-right, and bottom-left
        # order
        dst = np.array([
            [0, 0],
            [maxWidth - 1, 0],
            [maxWidth - 1, maxHeight - 1],
            [0, maxHeight - 1]], dtype="float32")
        # compute the perspective transform matrix and then apply it
        M = cv2.getPerspectiveTransform(ordered_corners, dst)
        warped = cv2.warpPerspective(img, M, (maxWidth, maxHeight))
        # return the warped image
        return warped

    def _order_corners(self, corners):
        """
        Helper function to generate our 4 points for perspective transform
        Important: points are generated in a consistent order! I'll do:
        1. top-left
        2. top-right
        3. bottom-right
        4. bottom-left

        From this blog post:
        https://www.pyimagesearch.com/2016/03/21/ordering-coordinates-clockwise-with-python-and-opencv/
        """

        # sort points by X
        xSorted = corners[np.argsort(corners[:, 0]), :]

        leftMost = xSorted[:2, :]
        rightMost = xSorted[2:, :]

        # grab top left and bottom left
        leftMost = leftMost[np.argsort(leftMost[:, 1]), :]
        (tl, bl) = leftMost
        D = dist.cdist(tl[np.newaxis], rightMost, "euclidean")[0]
        (br, tr) = rightMost[np.argsort(D)[::-1], :]

        return np.array([tl, tr, br, bl], dtype="float32")

    def _contour_on_edge(self, c, mask):
        # print(c[:, 0, 0])
        return (np.min(c[:, 0, 0]) < 2) or \
               (np.max(c[:, 0, 0]) > mask.shape[1] - 2)
