#!/usr/bin/env python

import cv2
import numpy as np
from scipy.spatial import distance as dist


def _contour_length_tuple(contour):
    """
    Used to sort contours by length
    (could be a lambda but for code clarity was chosen not to be)

    @param contour - contours
    @return arclength of contour
    """
    return cv2.arcLength(contour[0], True)


class PlateIsolatorColour:
    """
    The goal of this module is to pick out parking plates

    Provided with an image, it will find the plates (if they exist)

    Requires cars to fit the hue definition below
    (though colour bounds can be adjusted)

    Resources used:
    https://www.pyimagesearch.com/2014/08/04/opencv-python-color-detection/
    """

    def __init__(self, colour_bounds=None, testing=False):
        """
        Initialises the colour bounds based off of
        which cars are identified

        @param colour_bounds - colour boundaries of cars. If none, use default
        @param testing - determines whether intermediate images show
                         and if print statements are used
        """
        # HSB representations in the order: green, blue, yellow
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
        Picks out license plates from image
        @param: img - image in which to search for plates
        @param: duration - if testing set to true, display time for debugging
        @return: None, if no plates found
                 parking plate, license plate, otherwise
        """
        hsb = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        car_masks, car_colours = self._get_car_mask(hsb)
        if car_masks is None:
            # no cars found
            if self.testing:
                print("no car found")
                cv2.imshow("image", img)
                cv2.waitKey(duration)
            return None, None

        parking_corners, license_corners = self._get_plate_corners(hsb, car_masks[0], car_colours[0])
        if (parking_corners is None or license_corners is None):
            # no plates found on the first mask
            if (len(car_masks) == 2):
                """
                Note that two car contours are tried bc because when
                driving in the middle, the largest car contour spotted
                is often a car facing the wrong direction: therefore
                the second largest contour may be
                the one in which a plate can be found
                """
                parking_corners, license_corners = self._get_plate_corners(hsb, car_masks[1], car_colours[1])
            if (parking_corners is None and parking_corners is None):
                # couldn't find plate
                return None, None

        # extract plates from their outlines
        parking = self._cropped_image(img, parking_corners)
        license = self._cropped_image(img, license_corners)
        if self.testing:
            cv2.imshow("image", img)
            cv2.imshow("parking", parking)
            cv2.imshow("license", license)
            cv2.waitKey(duration)

        return parking, license

    def _get_car_mask(self, img, duration=3000):
        """
        Helper function which extracts (based on colour) up to 2 car masks

        @param: img - image in which car is being searched for
        @return: list of car masks (len 1 or 2)
                 list of ints representing colours used to get masks
        """
        bound_num = 0

        mask = [None, None, None]

        # mask by each possible car colour
        for (lower, upper) in self.colour_bounds:
            # create numpy arrays from colour boundaries
            lower = np.array(lower, dtype="uint8")
            upper = np.array(upper, dtype="uint8")

            # find colours in the range, apply mask
            mask[bound_num] = cv2.inRange(img, lower, upper)

            bound_num += 1

        """
        Get the two masks that had the largest contours (largest car), and
        the contour of said cars

        Note that two masks/contours are given because when driving in the
        middle, the largest car contour spotted is often a car facing the
        wrong direction: therefore the second largest contour may be the one
                         we want
        """
        used_masks, car_contours = self._car_contour(mask)

        if car_contours is None:
            return None, None

        car_masks = []

        for contour in car_contours:
            mask = np.zeros((img.shape[0], img.shape[1]), np.uint8)
            cv2.drawContours(mask, [contour], -1, (255), -1)
            car_masks.append(mask)

        if self.testing:
            cv2.imshow("image", img)
            cv2.imshow("car mask", car_masks[0])
            cv2.waitKey(duration)

        return car_masks, used_masks

    def _get_plate_corners(self, img, mask, colour):
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
        contours, _ = cv2.findContours(plate_mask, cv2.RETR_TREE,
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

    def _car_contour(self, masks):
        """
        Given a list of 3 masks (each created via filtering a different colour)
        searches for car contours based on a minimum area criteria.
        Larger area contours area prioritised
        (as they are more likely to yield readable plates)

        @param: mask - list of 3 mask in which to search for contours
        @return: list of (1 to 2) ints representing colours
                 (colour used to create masks of the top 2 contours)
                 list of (1 to 2) contours deemed the most likely car contours
        """
        contours0, _ = cv2.findContours(masks[0], cv2.RETR_TREE,
                                        cv2.CHAIN_APPROX_SIMPLE)
        contours1, _ = cv2.findContours(masks[1], cv2.RETR_TREE,
                                        cv2.CHAIN_APPROX_SIMPLE)
        contours2, _ = cv2.findContours(masks[2], cv2.RETR_TREE,
                                        cv2.CHAIN_APPROX_SIMPLE)
        """
        guestimate area: experimentally determined
        """
        MIN_AREA = (int)(0.75 * masks[0].shape[1] / 6 * masks[0].shape[0] / 4)

        # the second val in the tuples (0, 1, 2)
        # used to identify which colour was used to create the masks
        cont_0 = [(c, 0) for c in contours0 if cv2.contourArea(c) > MIN_AREA]
        cont_1 = [(c, 1) for c in contours1 if cv2.contourArea(c) > MIN_AREA]
        cont_2 = [(c, 2) for c in contours2 if cv2.contourArea(c) > MIN_AREA]

        good_contours = [c for c in (cont_0 + cont_1 + cont_2)
                         if (cv2.contourArea(c[0]) > MIN_AREA)]

        list.sort(good_contours, key=_contour_length_tuple)
        if (len(good_contours) == 0):
            return None, None

        # return our top 2 contour candidates
        car_contours = good_contours[0][0]
        car_colours = good_contours[0][1]

        if (len(good_contours) > 1):
            if (self._contour_on_edge(car_contours, masks[0])):
                car_contours = [good_contours[1][0], good_contours[0][0]]
                car_colours = [good_contours[1][1], good_contours[0][1]]
            else:
                car_contours = [good_contours[0][0], good_contours[1][0]]
                car_colours = [good_contours[0][1], good_contours[1][1]]
        else:
            car_contours = [car_contours]
            car_colours = [car_colours]

        return car_colours, car_contours

    def _cropped_image(self, img, corners):
        """
        Picks the subset of img described by the four corners
        in corners. Uses a perspective transform to render it a rectangle

        Used to pick out license plates based on the corners of their contours

        @param: img - the image to crop
        @param: corner - 4 corners to crop around
        @return: cropped image
        """
        ordered_corners = self._order_corners(corners)
        return self._four_point_transform(img, ordered_corners)

    def _four_point_transform(self, img, ordered_corners):
        """
        Takes the 4 point transform of the image based on ordered corners
        provided.
        Returns the perspective transform of the quadrilateral defined by
        the corners, such that the quadrilateral is transformed to a rectangle

        @param: img - image to tack the transform of
        @param: ordered_corners - ordered corners of quadrilateral
        @return: perspective transform of the image
        """
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
        Points will be ordered as such:
        1. top-left
        2. top-right
        3. bottom-right
        4. bottom-left

        From this blog post:
        https://www.pyimagesearch.com/2016/03/21/ordering-coordinates-clockwise-with-python-and-opencv/

        @param: corners - corners to order
        @return: the ordered corners
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
        """
        @param: c - contour
        @param: mask - mask of full image
        @return: true if c is on the edge of mask, false otherwise
        """
        return (np.min(c[:, 0, 0]) < 2) or \
               (np.max(c[:, 0, 0]) > mask.shape[1] - 2)
