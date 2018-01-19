# python library
import numpy as np
from numpy.random import *
from matplotlib import pyplot as plt
import argparse

# OpenCV
import cv2

# python scripts
import path as p


# load picture data
def find_object(img):

    #1. read image
    #print img.shape

    #2. gray image
    grayed = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    #3. blur image
    # blur parameter
    k_size = 11
    g_blur = cv2.GaussianBlur(grayed,(k_size,k_size),0)

    #4. binary image
    # binary parameters
    under_thresh = 105
    max_value = 255

    _, binary = cv2.threshold(g_blur, under_thresh, max_value, cv2.THRESH_BINARY)
    binary_inv = cv2.bitwise_not(binary)

    #5. recognize contour and rectangle
    contour, _ = cv2.findContours(binary_inv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    img_contour = np.copy(img)

    print len(contour[0])

    # area threshold
    # test1
    # min_area = 300
    # max_area = 2200

    # test2
    min_area = 2000
    max_area = 65000

    object_contour = [cnt for cnt in contour if cv2.contourArea(cnt) < max_area and cv2.contourArea(cnt) > min_area]
    cv2.drawContours(img_contour, object_contour, -1, (255,0,255),2)

    object_rec = []

    for i in range(len(object_contour)):
        object_rec.append(cv2.boundingRect(object_contour[i]))
        print 'x:'+str(object_rec[i][0])+' y:'+str(object_rec[i][1])+' w:'+str(object_rec[i][2])+' h:'+str(object_rec[i][3])
        cv2.rectangle(img_contour, (object_rec[i][0], object_rec[i][1]), (object_rec[i][0] + object_rec[i][2], object_rec[i][1] + object_rec[i][3]), (255, 100, 100), 2)

    if len(object_rec)  == 0:
        print "error: could not find objects."
    else:
        print "amount of rectangles: "+str(len(object_rec))

    #print object_rec

    cv2.imshow("rgb", img)
    cv2.imshow("gray", grayed)
    cv2.imshow("blurred", g_blur)
    cv2.imshow("binary", binary)
    cv2.imshow("contour", img_contour)
    cv2.waitKey(0)

    return object_rec


if __name__ == '__main__':

    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", required = True, help = "Path to the image")
    args = vars(ap.parse_args())

    cv_img = cv2.imread(args["image"])

    object_area = find_object(cv_img)

    plt.show()
