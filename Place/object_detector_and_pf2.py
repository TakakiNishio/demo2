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
from initializer import *
from find_objects import *
from particle_filter import *

import rospy
import rospkg
from motoman_interaction_msgs.msg import PickingInteraction


class SendCommand:

    def __init__(self):

        self.command_pub = rospy.Publisher('/picking_interaction', PickingInteraction, queue_size=1)
        self.command_msg = PickingInteraction()

    def publish_command(self, command):
        self.command_msg.tag = command[0]
        self.command_msg.num = command[1]
        self.command_pub.publish(self.command_msg)
        print
        print self.command_msg
        print "is published !!"

    def send(self, command_list):

        command_message = []
        for command in command_list:
            if command != None:
                command_message.append(command)
        print
        print "confirmed command"
        print command_message

        return command_message


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

    rospack = rospkg.RosPack()
    json_path = rospack.get_path('teaching_space')+'/json/'

    rospy.init_node("teaching_space")
    send_command = SendCommand()

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

        fw = open(json_path + "workspace_info.json","w")

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
        json_file = open(json_path + "workspace_info.json", "r")
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
    raw_command = [None, None]
    str_command = [None]*object_N
    str_command_list = [None]*object_N
    command = [[0 for i in range(2)] for j in range(object_N)]
    command_list = [[0 for i in range(2)] for j in range(object_N)]

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
                    str_command[i] = object_color_str_list[i] + " --> Box1"
                    command[i][0] = object_color_str_list[i]
                    command[i][1] = 1
                    # raw_command = [command[i][0], command[i][1]]

                if goal_box2[2] - box_area_margin < center[0] and \
                   center[0] < goal_box2[3] + box_area_margin and \
                   goal_box2[0] - box_area_margin < center[1] and \
                   center[1] < goal_box2[1] + box_area_margin:
                    str_command[i] = object_color_str_list[i] + " --> Box2"
                    command[i][0] = object_color_str_list[i]
                    command[i][1] = 2
                    # raw_command = [command[i][0], command[i][1]]

                # Update command
                if str_command_list[i] != str_command[i] :
                    str_command_list[i] = str_command[i]
                    command_list[i] = command[i]
                    raw_command = [command[i][0], command[i][1]]
                    print "update:"
                    print command[i]

                cv2.circle(result_frame, center, 8, (0, 255, 255), -1)

                cv2.putText(result_frame, "("+str(center[0])+", "+str(center[0])+")",
                        (center[0]+20, center[1]+50),cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                        (int(object_color_bgr_list[i][0]),
                         int(object_color_bgr_list[i][1]),
                         int(object_color_bgr_list[i][2])), 5)

                trajectory_points_list[i].appendleft(center)

                for k in range(1, len(trajectory_points_list[i])):
                    if trajectory_points_list[i][k - 1] is None or \
                       trajectory_points_list[i][k] is None:
                        continue
                    cv2.line(result_frame, trajectory_points_list[i][k-1],
                             trajectory_points_list[i][k],
                             (int(high_bgr_list[i][0]),int(high_bgr_list[i][1]),
                              int(high_bgr_list[i][2])), thickness=3)
            else:
                trajectory_points_list[i] = deque(maxlen=trajectory_length)

        cv2.imshow("tracking result", result_frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("a"):
            send_command.publish_command(raw_command)
            # send_command.send(command_list)
            cv2.waitKey(0)
            # break

        if key == 27:
            break

    send_command.send(command_list)

    cap.release()

    if args["display_each_steps"]:
            cv2.waitKey(0)

    cv2.destroyAllWindows()
