#!/usr/bin/env python
import cv2
import cv2.cv as cv


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
