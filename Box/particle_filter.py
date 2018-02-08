#!/usr/bin/env python
import cv2
import cv2.cv as cv
import numpy as np
import colorsys
from PIL import Image
import copy


def get_dominant_color(image):

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


def generate_color_range(dominant_hsv, h_range, v_th):

    if dominant_hsv[2] < v_th:
        # for black color tracking
        _LOWER_COLOR = np.array([dominant_hsv[0]-10,dominant_hsv[1]-40,dominant_hsv[2]-10])
        _UPPER_COLOR = np.array([dominant_hsv[0]+10,dominant_hsv[1]+40,dominant_hsv[2]+20])

    else:

        if dominant_hsv[0] < h_range:
            low_h = 0
        else:
            low_h = dominant_hsv[0]-h_range

        if dominant_hsv[0]+h_range > 179 :
            high_h = dominant_hsv[0]
        else:
            high_h = dominant_hsv[0] + h_range

            _LOWER_COLOR = np.array([low_h,80,80])
            _UPPER_COLOR = np.array([high_h,255,255])

    return _LOWER_COLOR, _UPPER_COLOR


class ParticleFilter:

    def __init__(self,particle_N, image_size):

        self.SAMPLEMAX = particle_N
        self.height = image_size[0]
        self.width = image_size[1]

    def initialize(self):
        self.Y = np.random.random(self.SAMPLEMAX) * self.height
        self.X = np.random.random(self.SAMPLEMAX) * self.width

    # Need adjustment for tracking object velocity
    def modeling(self):
        self.Y += np.random.random(self.SAMPLEMAX) * 100 - 50 # 2:1
        self.X += np.random.random(self.SAMPLEMAX) * 100 - 50


    def normalize(self, weight):
        return weight / np.sum(weight)

    def resampling(self, weight):
        index = np.arange(self.SAMPLEMAX)
        sample = []

        # choice by weight
        for i in range(self.SAMPLEMAX):
            idx = np.random.choice(index, p=weight)
            sample.append(idx)
        return sample

    def calcLikelihood(self, image):
        # white space tracking
        mean, std = 250.0, 10.0
        intensity = []

        for i in range(self.SAMPLEMAX):
            y, x = self.Y[i], self.X[i]
            if y >= 0 and y < self.height and x >= 0 and x < self.width:
                intensity.append(image[int(y),int(x)])
            else:
                intensity.append(-1)

        # normal distribution
        weights = 1.0 / np.sqrt(2 * np.pi * std) * np.exp(-(np.array(intensity) - mean)**2 /(2 * std**2))
        weights[intensity == -1] = 0
        weights = self.normalize(weights)
        return weights

    def filtering(self, image):
        self.modeling()
        weights = self.calcLikelihood(image)
        index = self.resampling(weights)
        self.Y = self.Y[index]
        self.X = self.X[index]

        # return COG
        return np.sum(self.Y) / float(len(self.Y)), np.sum(self.X) / float(len(self.X))
