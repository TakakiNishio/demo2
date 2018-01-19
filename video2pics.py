import numpy as np
import matplotlib.pyplot as plt
import argparse

import cv2


def video_to_images(video_path, image_path):

    # cap = cv2.VideoCapture(video_path)
    cap = cv2.VideoCapture(0)

    cnt = 0

    cv2.namedWindow('video_frame', cv2.WINDOW_NORMAL)
    cv2.namedWindow('saved_frame', cv2.WINDOW_NORMAL)

    while(True):
        ret, frame = cap.read()
        key = cv2.waitKey(25) & 0xFF

        if ret is not True:
            break

        cv2.imshow('video_frame', frame)

        if key == ord('a'):
            print cnt

        if key == ord('s'):
            cnt += 1
            cv2.imshow('saved_frame', frame)
            cv2.imwrite(image_path+str(cnt)+'.jpg', frame)

        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':

    ap = argparse.ArgumentParser()
    ap.add_argument("-v", "--video", required = True, help = "Path to the video")
    args = vars(ap.parse_args())

    video_path = args["video"]
    image_path = 'images/'

    print video_path

    video_to_images(video_path, image_path)
