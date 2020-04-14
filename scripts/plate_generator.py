#!/usr/bin/env python

import cv2
import csv
import numpy as np
import os
import pyqrcode
import random
import string

from random import randint
from PIL import Image, ImageFont, ImageDraw

path = os.path.dirname(os.path.realpath(__file__)) + "/"
texture_path = '../media/materials/textures/'
NUM_PLATES = 16

with open(path + "plates.csv", 'w') as plates_file:
    csvwriter = csv.writer(plates_file)
    positions = range(1, NUM_PLATES+1)
    random.shuffle(positions)

    for i in range(0, NUM_PLATES):

        ### Generate license plates
        
        # Pick two random letters
        plate_alpha = ""
        for _ in range(0, 2):
            plate_alpha += (random.choice(string.ascii_uppercase))
        num = randint(0, 99)

        # Pick two random numbers
        plate_num = "{:02d}".format(num)

        # Save plate to file
        csvwriter.writerow([plate_alpha+plate_num])

        ### Create a plate from template image

        #Load template
        blank_plate = cv2.imread(path+'blank_template.png')

        # To use monospaced font for the license plate we need to use the PIL
        # package.
        # Convert into a PIL image (this is so we can use the monospaced fonts)
        blank_plate_pil = Image.fromarray(blank_plate)
        # Get a drawing context
        draw = ImageDraw.Draw(blank_plate_pil)
        monospace = ImageFont.truetype("/usr/share/fonts/truetype/ubuntu/UbuntuMono-R.ttf", 120)
        X_POS = 25
        Y_POS = 450
        draw.text((X_POS, Y_POS), plate_alpha, (0,0,0), font=monospace)
        draw.text((X_POS + 135, Y_POS), plate_num, (0,0,0), font=monospace)

        # Create QR code image
        # spot_name = "P" + str(i)
        # qr = pyqrcode.create(spot_name+"_" + plate_alpha + plate_num)
        # qrname = path + "QRCode_" + str(i) + ".png"
        # qr.png(qrname, scale=20)
        # QR_img = cv2.imread(qrname)
        # QR_img = cv2.resize(QR_img, (600, 600), interpolation=cv2.INTER_AREA)

        # Create parking spot label
        s = "P{:02d}".format(positions[i])
        draw.text((65, 20), s, (0, 0, 0), font=monospace)

        # Convert back to OpenCV image and save
        blank_plate = np.array(blank_plate_pil)
        spot_w_plate = blank_plate

        # Merge labelled or unlabelled images and save
        # labelled = np.concatenate((QR_img, spot_w_plate), axis=0)
        # unlabelled = np.concatenate((255 * np.ones(shape=[600, 600, 3],
        #                             dtype=np.uint8), spot_w_plate), axis=0)
        unlabelled = blank_plate
        # cv2.imwrite(os.path.join(path + texture_path + "labelled/",
        #                          "plate_" + str(i) + ".png"), labelled)
        cv2.imwrite(os.path.join(path+texture_path+"unlabelled/",
                                 "plate_" + str(i) + ".png"), unlabelled)
