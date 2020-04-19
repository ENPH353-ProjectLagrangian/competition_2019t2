import cv2
import numpy as np
import copy


def draw_lines(img, lines, color=(0, 0, 255), size=2):
    """
    Draws lines on image. modifies img.

    :param img: Image to be drawn upon.
    :param lines: Lines written in hesse normal form to be drawn on image.
    :param color: Color of drawn lines.
    :param size: Size of drawn lines.

    """
    length = max((img.shape[0], img.shape[1]))

    for line in lines:
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
    """
    Custom clustering algorithm clusters groups of angles into two orthogonal clusters.
    Angles outside these clusters will be disregarded.
    Angles close to 180 degrees apart will be moved into the same cluster.

    :param angles: Array of angles to be analyzed
    :return: headings: Size two list of mean angles of orthogonal clusters
             clusters: Two arrays of indices of angles that fit into respective clusters.
    """

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

    #  There is an ambiguity in which direction is "forward" when theta is close to pi/2.
    #  -1 is right turn, 1 is left turn.
    turn_bias = -1  # Starts off at -1 could be changed by advanced algorithms
    # but probably shouldn't for this naive approach.

    # The margin of what is considered "close" to an angle
    tolerance = np.pi / 60  # 3 degrees

    head_abs = np.abs(headings[1])
    if np.abs(head_abs - 1.57) < tolerance:
        headings[1] = turn_bias * head_abs

    return headings, clusters


def find_crosswalk(img):
    """
    Find the center of crosswalks seen by robot.
    :param img: image to analyze
    :return: Pixel coordinates of center of crosswalk in image or None if no crosswalk is detected.
    """
    hsv_img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    lower_red = np.array([110, 150, 150])
    upper_red = np.array([130, 255, 255])
    # Threshold the HSV image to get only red parts
    mask = cv2.inRange(hsv_img, lower_red, upper_red)

    if mask.max() == 0:
        return None
    else:
        moments = cv2.moments(mask, binaryImage=True)
        center = (int(moments["m10"] / moments["m00"]), int(moments["m01"] / moments["m00"]))
        return center
