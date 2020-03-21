#!/usr/bin/env python

import rospy
import cv2 as cv
import time
from pynput.keyboard import Key, Listener
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

from geometry_msgs.msg import Twist

rospy.init_node('topic_publisher')
pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)
capture_period = rospy.Rate(50)

capture_mode = "periodic"
lin_speed = 1
ang_speed = 2

left_flag = 0
right_flag = 0
back_flag = 0
for_flag = 0

bridge = CvBridge()

move = Twist()

def get_image(img):
    
    # Try to convert 
    try:
        cv_image = bridge.imgmsg_to_cv2(img)
    except CvBridgeError as e:
        print(e)

    return cv_image

def save_image(img):

    cv_img = get_image(img)
    cv.imwrite("images\\image_" + str(int(time.time())) + ".png", cv_img)
    
def on_press(key):
    global left_flag, right_flag, for_flag, back_flag, shift_flag, lin_speed, ang_speed
    
    if key == Key.up:
        for_flag = 1

    elif key == Key.down:
        back_flag = 1

    elif key == Key.left:
        left_flag = 1

    elif key == Key.right:
        right_flag = 1
    
    move.linear.x =  lin_speed * (for_flag - back_flag)
    move.angular.z = ang_speed * (left_flag - right_flag)
    pub.publish(move)

def on_release(key):
    global left_flag, right_flag, for_flag, back_flag, shift_flag, lin_speed, ang_speed

    if key == Key.up:
        for_flag = 0

    elif key == Key.down:
        back_flag = 0

    elif key == Key.left:
        left_flag = 0

    elif key == Key.right:
        right_flag = 0
    
    elif key == "q":
        lin_speed += 0.5

    elif key == "w" and lin_speed >= 0.5:
        lin_speed -= 0.5

    elif key == "e":
        ang_speed += 0.5

    elif key == "r" and ang_speed >= 0.5:
        ang_speed -= 0.5

    elif key == Key.space and capture_mode == "on_key":
        image_capture()
    
    move = Twist()
    move.linear.x = lin_speed * (for_flag - back_flag)
    move.angular.z = ang_speed * (left_flag - right_flag)
    pub.publish(move)

    if key == Key.esc:
        # Stop listener
        return False


rospy.Subscriber('/rrbot/camera1/image_raw', Image, save_image)

with Listener(on_press=on_press, on_release=on_release) as listener:
    listener.join()
