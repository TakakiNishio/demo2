#!/usr/bin/env python
import cv2
import colorsys
from PIL import Image as PIL_Image
import numpy as np
import os
import copy
import rospkg


class ColorRecognition():

    def __init__(self):

        size = 200, 200, 3
        self.dominant_color_display = np.zeros(size, dtype=np.uint8)

    def get_dominant_color(self, cv_image):

        image = PIL_Image.fromarray(cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB))
        image = image.convert('RGBA')
        image.thumbnail((200, 200))
        max_score = None
        dominant_color = None

        for count, (r, g, b, a) in image.getcolors(image.size[0] * image.size[1]):
            if a == 0:
                continue
            saturation = colorsys.rgb_to_hsv(r / 255.0, g / 255.0, b / 255.0)[1]
            y = min(abs(r * 2104 + g * 4130 + b * 802 + 4096 + 131072) >> 13, 235)
            y = (y - 16.0) / (235 - 16)
            if y > 0.9:
                continue
            score = (saturation + 0.1) * count
            if score > max_score:
                max_score = score
                dominant_color = [b, g, r]

        return dominant_color


    def bgr_to_hsv(self, bgr_color):
        hsv = cv2.cvtColor(np.array([[[bgr_color[0], bgr_color[1], bgr_color[2]]]],
                                    dtype=np.uint8), cv2.COLOR_BGR2HSV)[0][0]
        return [hsv[0], hsv[1], hsv[2]]


    def hsv_to_bgr(self, hsv_color):
        bgr = cv2.cvtColor(np.array([[[hsv_color[0], hsv_color[1], hsv_color[2]]]],
                                    dtype=np.uint8),cv2.COLOR_HSV2BGR)[0][0]
        return [bgr[0], bgr[1], bgr[2]]


    def color_recognition(self, hsv_color):

        h = hsv_color[0]
        s = hsv_color[1]
        v = hsv_color[2]

        print "H: "+str(h)+" S: "+str(s)+" V: "+str(v)

        # if h <= 70 and s <= 50 and v <= 100:
        if h <= 50 and v <= 50:
            return "black"

        if (165 <= h and h <= 180) or (0 <= h and h <= 10):
            return "red"

        elif 50 <= h and h <= 85:
            return "green"

        elif 95 <= h and h <= 130:
            return "blue"

        else:
            return "others"


    def __call__(self, image):

        new_image = image
        # display_image = copy.deepcopy(new_image)

        width, height = image.shape[1::-1]

        top = int(height/2)
        bottom = int(2*height/3)

        left = int(width/3)
        right = int(2*width/3)

        image = image[top:bottom, left:right]

        dominant_bgr = self.get_dominant_color(image)
        dominant_hsv = self.bgr_to_hsv(dominant_bgr)

        self.dominant_color_display[:] = dominant_bgr

        recognition_result = self.color_recognition(dominant_hsv)

        # cv2.imshow('image',display_image)
        # cv2.imshow("dominant", self.dominant_color_display)
        # cv2.waitKey(3)

        return recognition_result, dominant_bgr
