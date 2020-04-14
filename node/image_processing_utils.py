import cv2
import numpy as np
import copy


def get_hough_lines(img, img_type="edges", threshold=125, probabilistic=False, min_line_length=100, max_line_gap=10):
    """

    @param img: image to generate lines on
    @param threshold: threshold to pass to the Houghlines generator
    @param probabilistic: if true uses the probabalistic hough transform see cv2.HoughLinesP (
    @param min_line_length: minimum length of a line, passed to cv2.HoughLinesP
    @param max_line_gap: maximum gap in a line, passed to cv2.HoughLinesP
    @param verbose: Whether to print information to the console (Currently unimplemented)

    @return: ndarray of lines. By default each line is represented by the length and angle of a vector
    (with origin (0,0)) normal to and intersecting the line. If probabalistic is true lines are instead
    represented by a pair of end points of a line segment,
    """
    if img_type == "color":
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)

    elif img_type == "gray":
        edges = cv2.Canny(img, 50, 150, apertureSize=3)
    elif img_type == "edges":
        edges = img
    else:
        raise ValueError("Img type " + img_type + " not supported")

    if probabilistic:
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180,
                                threshold, min_line_length, max_line_gap)

    else:
        lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold)

    return lines


def draw_lines(img, lines, probabilistic=False, color=(0, 0, 255), size=2):
    length = max((img.shape[0], img.shape[1]))

    for line in lines:
        if probabilistic:
            for x1, y1, x2, y2 in line:
                cv2.line(img, (x1, y1), (x2, y2), color, size)

        else:
            for rho, theta in line:
                a = np.cos(theta)
                b = np.sin(theta)
                x0 = a * rho
                y0 = b * rho
                x1 = int(x0 + length * (-b))
                y1 = int(y0 + length * a)
                x2 = int(x0 - length * (-b))
                y2 = int(y0 - length * a)

                cv2.line(img, (x1, y1), (x2, y2), color, size)


def find_cardinal_clusters(angles):
    # The K-means algorithm is sometimes unreliable.
    # But because we know the distribution of the angles already we can create a much more reliable one

    good_buckets = np.array([[0], [np.pi / 2]])
    min_include = -1

    half_bucket = np.pi * 5 / 180

    # First split the angles into buckets 90 degrees apart. Find the set of buckets that include the most points.
    # This is a course move through the state space buckets are 10 degrees wide spaced 5 degrees apart.

    for theta_0 in np.linspace(0, np.pi / 2.0, 18, endpoint=False):
        buckets = [[], [], []]
        for angle in angles:

            if theta_0 - half_bucket <= angle[0] <= theta_0 + half_bucket:
                buckets[0].append(angle)
            elif theta_0 - half_bucket - np.pi / 2.0 <= angle[0] <= theta_0 + half_bucket - np.pi / 2.0:
                buckets[1].append(angle)
            elif theta_0 - half_bucket + np.pi / 2.0 <= angle[0] <= theta_0 + half_bucket + np.pi / 2.0:
                buckets[2].append(angle)

        include = buckets[0].__len__() + buckets[1].__len__()

        if include > min_include:
            min_include = include
            good_buckets = copy.copy(buckets)

    #  Now we take a mean value to find the actual angle from the course buckets.

    theta = 0

    n = 0.0

    if good_buckets[0].__len__() != 0:
        theta += np.average(good_buckets[0])
        n += 1.0
    if good_buckets[1].__len__() != 0:
        theta += np.average(good_buckets[1]) - np.pi / 2
        n += 1.0
    if good_buckets[2].__len__() != 0:
        theta += np.average(good_buckets[2]) + np.pi / 2
        n += 1.0

    if n != 0:
        theta *= 1 / n

    #  Remake buckets at the refined center with indices.
    final_bucket = [[], []]

    for i in range(angles.__len__()):
        if theta - half_bucket <= angles[i][0] <= theta + half_bucket:
            final_bucket[0].append(i)
        elif theta - half_bucket - np.pi / 2.0 <= angles[i][0] <= theta + half_bucket - np.pi / 2.0 or \
                theta - half_bucket + np.pi / 2.0 <= angles[i][0] <= theta + half_bucket + np.pi / 2.0:
            final_bucket[1].append(i)

    headings = np.array([[theta], [theta + np.pi / 2.0]])
    clusters = np.array(final_bucket)

    # Normalize angles (pi=0, pi/2=-pi/2)
    headings = np.array(list([[heading[0], heading[0] - np.pi][heading[0] > np.pi / 2]] for heading in headings))

    if abs(headings[0]) > abs(headings[1]):
        headings = headings[::-1]
        clusters = clusters[::-1]
    return headings, clusters


def detect_crosswalk(img):
    parameters = cv2.SimpleBlobDetector_Params()
    detector = cv2.SimpleBlobDetector_create(parameters)

    return detector.detect(img[:, :, 0])
