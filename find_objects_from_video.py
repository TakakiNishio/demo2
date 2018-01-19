# python library
import numpy as np
import colorsys
from PIL import Image
import copy
import argparse


# OpenCV
import cv2
import cv2.cv as cv


def get_dominant_color(image):
    """
    Find a PIL image's dominant color, returning an (r, g, b) tuple.
    """
    image = image.convert('RGBA')
    # Shrink the image, so we don't spend too long analysing color
    # frequencies. We're not interpolating so should be quick.
    image.thumbnail((200, 200))
    max_score = None
    dominant_color = None

    for count, (r, g, b, a) in image.getcolors(image.size[0] * image.size[1]):
        # Skip 100% transparent pixels
        if a == 0:
            continue
        # Get color saturation, 0-1
        saturation = colorsys.rgb_to_hsv(r / 255.0, g / 255.0, b / 255.0)[1]
        # Calculate luminance - integer YUV conversion from
        # http://en.wikipedia.org/wiki/YUV
        y = min(abs(r * 2104 + g * 4130 + b * 802 + 4096 + 131072) >> 13, 235)
        # Rescale luminance from 16-235 to 0-1
        y = (y - 16.0) / (235 - 16)
        # Ignore the brightest colors
        if y > 0.9:
            continue
        # Calculate the score, preferring highly saturated colors.
        # Add 0.1 to the saturation so we don't completely ignore grayscale
        # colors by multiplying the count by zero, but still give them a low
        # weight.
        score = (saturation + 0.1) * count
        if score > max_score:
            max_score = score
            dominant_color = [b, g, r]

    return dominant_color


def bgr_to_hsv(bgr_color):
    hsv = cv2.cvtColor(np.array([[[bgr_color[0], bgr_color[1], bgr_color[2]]]],
                                dtype=np.uint8), cv2.COLOR_BGR2HSV)[0][0]
    return [hsv[0], hsv[1], hsv[2]]


def hsv_to_bgr(hsv_color):
    bgr = cv2.cvtColor(np.array([[[hsv_color[0], hsv_color[1], hsv_color[2]]]],
                                dtype=np.uint8),cv2.COLOR_HSV2BGR)[0][0]
    return [bgr[0], bgr[1], bgr[2]]


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
    # under_thresh = 90
    max_value = 255

    under_thresh = 95

    _, binary = cv2.threshold(g_blur, under_thresh, max_value, cv2.THRESH_BINARY)
    binary_inv = cv2.bitwise_not(binary)

    #5. recognize contour and rectangle
    contour, _ = cv2.findContours(binary_inv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    img_contour = np.copy(img)

    # area threshold

    # test1
    min_area = 300
    max_area = 2200

    # test2
    # min_area = 2000
    # max_area = 65000

    object_contour = [cnt for cnt in contour if cv2.contourArea(cnt) < max_area and cv2.contourArea(cnt) > min_area]
    cv2.drawContours(img_contour, object_contour, -1, (255,0,255),2)

    object_rec = []

    if len(object_rec)  == 0:
        print "could not find objects."
    else:
        print "amount of rectangles: "+str(len(object_rec))

    for i in range(len(object_contour)):
        object_rec.append(cv2.boundingRect(object_contour[i]))
        print 'x:'+str(object_rec[i][0])+' y:'+str(object_rec[i][1])+' w:'+str(object_rec[i][2])+' h:'+str(object_rec[i][3])
        cv2.rectangle(img_contour, (object_rec[i][0], object_rec[i][1]), (object_rec[i][0] + object_rec[i][2], object_rec[i][1] + object_rec[i][3]), (255, 100, 100), 2)

    cv2.imshow("binary", binary)

    return object_rec, img_contour


if __name__ == '__main__':

    ap = argparse.ArgumentParser()
    ap.add_argument("-v", "--video", required = True, help = "Path to the video")
    args = vars(ap.parse_args())

    cap = cv2.VideoCapture(args["video"])
    # cap = cv2.VideoCapture(0)

    while True:

        ret, frame = cap.read()

        frame = cv2.flip(frame,-1)

        if ret == False:
            break

        print frame.shape

        # test1
        top = 130
        left = 140
        bottom = 380
        right = 480

        # test2
        # top = 0
        # left = 130
        # bottom = 315
        # right = 630

        cropped_frame = frame[top:bottom, left:right]

        object_area, img_contour = find_object(cropped_frame)
        # object_area, img_contour = find_object(frame)

        cv2.imshow("contour", img_contour)

        key = cv2.waitKey(20) & 0xFF

        if key == 27:
            break

    cap.release()
