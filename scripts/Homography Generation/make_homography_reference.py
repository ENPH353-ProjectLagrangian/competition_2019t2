"""
Alters the property of the homography reference image to used in generate_homography.py.
For instance the size of the image and how "high up" in the sky the new perspective is.

"""
import sys
import cv2 as cv

# Loads an image, in our case an asymetric circle pattern
src = cv.imread("ref_img.png")

# The boarder represents how "zoomed out" the new perspective is.
top = int(0.5 * src.shape[0])  # shape[0] = rows
bottom = 30
left = int(0.35 * src.shape[1])  # shape[1] = cols
right = left
value = [255, 255, 255]

dst = cv.copyMakeBorder(src, top, bottom, left, right, cv.BORDER_CONSTANT, None, value)

# Scale the resolution of the image.
scale_percent = 30  # percent of original size
width = int(dst.shape[1] * scale_percent / 100)
height = int(dst.shape[0] * scale_percent / 100)
dim = (width, height)
# resize image
resized = cv.resize(dst, dim, interpolation=cv.INTER_AREA)[: int(height * .88), :]

print('Resized Dimensions : ', resized.shape)

cv.imwrite("new_ref.jpg", resized)

cv.imshow("Resized image", resized)
cv.waitKey(1000)
