#!/usr/bin/env python
import cv2
import cv2.cv as cv
import copy


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
