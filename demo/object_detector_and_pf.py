#!/usr/bin/env python
import cv2
import cv2.cv as cv
import numpy as np
import colorsys
from PIL import Image
import copy
import argparse
import json
import collections as cl
from collections import deque

from color_detection import *


class Initializer:

    # def __init__(self):
    #     self.robot_workspace_done = False
    #     self.goal_box1_done = False
    #     self.goal_box2_done = False

    def clear(self):

        self.robot_workspace_done = False
        self.goal_box1_done = False
        self.goal_box2_done = False

        self.press_A_cnt = 0

        self.robot_workspace = [0]*4 # top,bottom,left,right
        self.goal_box1 = [0]*4 # top,bottom,left,right
        self.goal_box2 = [0]*4 # top,bottom,left,right

    def finish_robot_workspace_initialization(self,image):
        self.robot_workspace_done = True

    def finish_goal_box1_initialization(self):
        self.goal_box1_done = True

    def mouse_event(self, event, x, y, flags, param):

        if event == cv2.EVENT_LBUTTONUP:
            print "left x: " +str(x) + " y: " + str(y)

            if not self.robot_workspace_done:
                self.robot_workspace[0] = y
                self.robot_workspace[2] = x
            elif not self.goal_box1_done:
                self.goal_box1[0] = y
                self.goal_box1[2] = x
            else:
                self.goal_box2[0] = y
                self.goal_box2[2] = x

        elif event == cv2.EVENT_RBUTTONUP:
            print "right x: " +str(x) + " y: " + str(y)

            if not self.robot_workspace_done:
                self.robot_workspace[1] = y
                self.robot_workspace[3] = x
            elif not self.goal_box1_done:
                self.goal_box1[1] = y
                self.goal_box1[3] = x
            else:
                self.goal_box2[1] = y
                self.goal_box2[3] = x

    def display_initialzation_result(self, image):

        cv2.circle(image, (self.robot_workspace[2], self.robot_workspace[0]), 10, (0, 0, 255), -1)
        cv2.circle(image, (self.robot_workspace[3], self.robot_workspace[1]), 10, (255, 0, 0), -1)

        cv2.circle(image, (self.goal_box1[2], self.goal_box1[0]), 10, (0, 0, 255), -1)
        cv2.circle(image, (self.goal_box1[3], self.goal_box1[1]), 10, (255, 0, 0), -1)

        cv2.circle(image, (self.goal_box2[2], self.goal_box2[0]), 10, (0, 0, 255), -1)
        cv2.circle(image, (self.goal_box2[3], self.goal_box2[1]), 10, (255, 0, 0), -1)

        if self.robot_workspace_done:
            cv2.rectangle(image,
                          (self.robot_workspace[2], self.robot_workspace[0]),
                          (self.robot_workspace[3], self.robot_workspace[1]),
                          (255, 100, 100), 3)
            cv2.putText(image, "robot_workspace",
                        (self.robot_workspace[2]+20, self.robot_workspace[0]+50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 100, 100), 5)

        if self.goal_box1_done:
            cv2.rectangle(image,
                          (self.goal_box1[2], self.goal_box1[0]),
                          (self.goal_box1[3], self.goal_box1[1]),
                          (0, 0, 0), 3)
            cv2.putText(image, "Box1",
                        (self.goal_box1[2]+20, self.goal_box1[0]+50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)

        if self.goal_box2_done:
            cv2.rectangle(image,
                          (self.goal_box2[2], self.goal_box2[0]),
                          (self.goal_box2[3], self.goal_box2[1]),
                          (0, 0, 0), 3)
            cv2.putText(image, "Box2",
                        (self.goal_box2[2]+20, self.goal_box2[0]+50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)



def find_object(robot_workspace, img):

    img_contour = copy.deepcopy(img)

    #1. crop image
    # top,bottom,left,right
    img = img[robot_workspace[0]:robot_workspace[1], robot_workspace[2]:robot_workspace[3]]

    #2. gray image
    grayed = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    #3. blur image
    # blur parameter
    k_size = 11
    g_blur = cv2.GaussianBlur(grayed,(k_size,k_size),0)

    #4. binary image
    # binary parameters
    max_value = 255
    under_thresh = 95

    _, binary = cv2.threshold(g_blur, under_thresh, max_value, cv2.THRESH_BINARY)
    binary_inv = cv2.bitwise_not(binary)

    #5. recognize contour and rectangle
    contour, _ = cv2.findContours(binary_inv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # area threshold
    min_area = 2000
    max_area = 6000

    object_contour = [cnt for cnt in contour if cv2.contourArea(cnt) < max_area and cv2.contourArea(cnt) > min_area]
    object_rec_list = []


    for i in range(len(object_contour)):

        object_rec = cv2.boundingRect(object_contour[i])

        object_top = object_rec[1] + robot_workspace[0]
        object_left = object_rec[0] + robot_workspace[2]
        object_bottom = object_top + object_rec[3]
        object_right = object_left + object_rec[2]

        cv2.rectangle(img_contour,
                      (object_left, object_top),
                      (object_right, object_bottom),
                      (255, 100, 100), 2)

        object_rec_list.append([object_top, object_bottom, object_left, object_right])

    cv2.imshow("binary", binary)
    cv2.imshow("objects", img_contour)

    return object_rec_list


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



if __name__ == '__main__':

    # video
    ap = argparse.ArgumentParser()
    ap.add_argument('--video_file', '-v', type=str, default=False,help='path to video')
    ap.add_argument('--camera_ID', '-c', type=int, default=0,help='camera ID')
    ap.add_argument('--init', '-init', action="store_true")
    ap.add_argument('--display_each_steps', '-d', action="store_true")
    args = vars(ap.parse_args())

    if not args["video_file"] == False:
        cap = cv2.VideoCapture(args["video_file"])
    else:
        cap = cv2.VideoCapture(args["camera_ID"])

    ret, frame = cap.read()
    frame = cv2.flip(frame,-1)
    w, h = frame.shape[1::-1]
    image_size = (h, w)

    # 0. robot_workspace initialization
    if args["init"]:

        workspace_info = cl.OrderedDict()

        print "-----------------------"
        print "Step0 : initialization of the teaching space"

        initializer = Initializer()
        initializer.clear()
        cv2.namedWindow("initialize", cv2.WINDOW_NORMAL)
        cv2.setMouseCallback("initialize", initializer.mouse_event)
        original_frame = copy.deepcopy(frame)
        initializer.clear()

        while (True):

            frame = copy.deepcopy(original_frame)
            initializer.display_initialzation_result(frame)
            cv2.imshow("initialize", frame)

            key = cv2.waitKey(1) & 0xFF

            if key == ord("c"):
                initializer.clear()
                print "clear"

            if key == ord("a"):

                if initializer.press_A_cnt == 0:
                    initializer.robot_workspace_done = True
                    initializer.press_A_cnt += 1
                    print "Finished robot_workspace initialization"

                elif initializer.press_A_cnt == 1:
                    initializer.goal_box1_done = True
                    initializer.press_A_cnt += 1
                    print "Finished goal_box1 initialization"

                elif initializer.press_A_cnt == 2:
                    initializer.goal_box2_done = True
                    initializer.press_A_cnt += 1
                    print "Finished goal_box2 initialization"

                else:
                    print "Initialization Completed. press key Q."

            if key == ord("q"):
                break

        if not args["display_each_steps"]:
            cv2.destroyAllWindows()

        # top,bottom,left,right
        robot_workspace = initializer.robot_workspace
        goal_box1 = initializer.goal_box1
        goal_box2 = initializer.goal_box2

        workspace_info["robot_workspace"] = robot_workspace
        workspace_info["goal_box1"] = goal_box1
        workspace_info["goal_box2"] = goal_box2

        fw = open("workspace_info.json","w")

        json.dump(workspace_info,fw,indent=4)
        fw.close

        print "saved json file."


    # 1. object detection
    print "-----------------------"
    print "Step1 : object detection"
    print

    if not args["init"]:

        # load initial values of workspace
        print "Open json file."
        json_file = open("workspace_info.json", "r")
        json_data = json.load(json_file)
        json_file.close

        robot_workspace = json_data["robot_workspace"]
        goal_box1 = json_data["goal_box1"]
        goal_box2 = json_data["goal_box2"]

    while True:

        ret, frame = cap.read()

        if ret == False:
            break

        frame = cv2.flip(frame,-1)
        object_rec_list = find_object(robot_workspace, frame)

        key = cv2.waitKey(10) & 0xFF

        if key == ord("q"):
            break

    print "found objects: " + str(len(object_rec_list))

    if not args["display_each_steps"]:
        cv2.destroyAllWindows()


    # 2. get a dominant color of the object
    print "-----------------------"
    print "Step2: dominant color recognition"
    print

    color_recognition = ColorRecognition()

    object_cnt = 0
    object_color_bgr_list = []
    object_color_str_list = []

    for object_rec in object_rec_list:

        object_cnt += 1
        object_image  = frame[object_rec[0]:object_rec[1], object_rec[2]:object_rec[3]]
        object_color_str, object_color_bgr = color_recognition(object_image)
        object_color_bgr_list.append(object_color_bgr)
        object_color_str_list.append(object_color_str)
        print "---> " + object_color_str
        print
        cv2.imshow(object_color_str, object_image)

    cv2.waitKey(0)

    if not args["display_each_steps"]:
        cv2.destroyAllWindows()

    # 3. particle filter
    print "-----------------------"
    print "Step3: tracking with particle filter"
    print
    print "teaching commands:"
    print

    particle_N = 250
    trajectory_length = 30
    object_size = 300
    h_range = 10
    v_th = 50
    box_area_margin = 10

    PF_list = []
    trajectory_points_list = []
    _LOWER_COLOR_list = []
    _UPPER_COLOR_list = []
    low_bgr_list = []
    high_bgr_list = []

    object_N = len(object_color_bgr_list)
    teaching_command = [False]*object_N
    command_recorded_flag = [False]*object_N

    for i in range(object_N):

        pf = ParticleFilter(particle_N, image_size)
        pf.initialize()

        PF_list.append(pf)

        trajectory_points_list.append(deque(maxlen=trajectory_length))

        dominant_hsv = color_recognition.bgr_to_hsv(object_color_bgr_list[i])
        _LOWER_COLOR, _UPPER_COLOR = generate_color_range(dominant_hsv, h_range, v_th)
        _LOWER_COLOR_list.append(_LOWER_COLOR)
        _UPPER_COLOR_list.append(_UPPER_COLOR)
        low_bgr_list.append(color_recognition.hsv_to_bgr(_LOWER_COLOR))
        high_bgr_list.append(color_recognition.hsv_to_bgr(_UPPER_COLOR))

    cv2.namedWindow("tracking result", cv2.WINDOW_NORMAL)

    while True:

        ret, frame = cap.read()

        if ret == False:
            break

        frame = cv2.flip(frame,-1)
        frame_size = frame.shape

        result_frame = copy.deepcopy(frame)

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        for i in range(object_N):

            # Threshold the HSV image to get only a color
            mask = cv2.inRange(hsv, _LOWER_COLOR_list[i], _UPPER_COLOR_list[i])

            # Start Tracking
            y, x = PF_list[i].filtering(mask)

            p_range_x = np.max(PF_list[i].X)-np.min(PF_list[i].X)
            p_range_y = np.max(PF_list[i].Y)-np.min(PF_list[i].Y)

            for j in range(PF_list[i].SAMPLEMAX):
                cv2.circle(result_frame, (int(PF_list[i].X[j]), int(PF_list[i].Y[j])), 2,
                           (int(object_color_bgr_list[i][0]),
                            int(object_color_bgr_list[i][1]),
                            int(object_color_bgr_list[i][2])), -1)

            if p_range_x < object_size and p_range_y < object_size:

                center = (int(x), int(y))

                # goal_box1 = [0]*4 # top,bottom,left,right
                # goal_box2 = [0]*4 # top,bottom,left,right

                if goal_box1[2] - box_area_margin < center[0] and \
                   center[0] < goal_box1[3] + box_area_margin and \
                   goal_box1[0] - box_area_margin < center[1] and \
                   center[1] < goal_box1[1] + box_area_margin:
                    if not command_recorded_flag[i]:
                        command = object_color_str_list[i] + " --> Box1"
                        teaching_command[i] = command
                        print command
                        command_recorded_flag[i] = True

                if goal_box2[2] - box_area_margin < center[0] and \
                   center[0] < goal_box2[3] + box_area_margin and \
                   goal_box2[0] - box_area_margin < center[1] and \
                   center[1] < goal_box2[1] + box_area_margin:
                    if not command_recorded_flag[i]:
                        command = object_color_str_list[i] + " --> Box2"
                        teaching_command[i] = command
                        print command
                        command_recorded_flag[i] = True


                cv2.circle(result_frame, center, 8, (0, 255, 255), -1)
                trajectory_points_list[i].appendleft(center)

                for k in range(1, len(trajectory_points_list[i])):
                    if trajectory_points_list[i][k - 1] is None or trajectory_points_list[i][k] is None:
                        continue
                    cv2.line(result_frame, trajectory_points_list[i][k-1], trajectory_points_list[i][k],
                             (int(high_bgr_list[i][0]),int(high_bgr_list[i][1]),int(high_bgr_list[i][2])), thickness=3)
            else:
                trajectory_points_list[i] = deque(maxlen=trajectory_length)

        cv2.imshow("tracking result", result_frame)

        if cv2.waitKey(10) & 0xFF == 27:
            break

    # print teaching_command
    print

    cap.release()

    if args["display_each_steps"]:
            cv2.waitKey(0)

    cv2.destroyAllWindows()
