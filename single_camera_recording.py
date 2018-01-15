#python library
import numpy as np
import argparse
import os

#OpenCV
import cv2


# FPS counter
class FrameRate:
    def __init__(self):
        self._count     = 0
        self._fps       = 0
        self._freq      = 1000 / cv2.getTickFrequency()
        self._tmStart   = cv2.getTickCount()
        self._tmNow     = cv2.getTickCount()
    def get(self):
        self._count     += 1
        self._tmNow      = cv2.getTickCount()
        tmDiff           = (self._tmNow - self._tmStart) * self._freq
        if tmDiff >= 1000 :
            self._tmStart    = self._tmNow
            self._fps        = self._count
            self._count      = 0
        return self._fps


# preparing to save the video
def initWriter(video_name, w, h, fps, save_path):
    #fourcc = cv2.cv.CV_FOURCC('D','I','B',' ')
    #fourcc = cv2.cv.CV_FOURCC('D','I','V','X')
    fourcc = cv2.cv.CV_FOURCC('F','L','V','1')
    rec = cv2.VideoWriter(save_path+str(video_name)+'.avi', \
                          fourcc, fps, (w, h))
    return rec


# capture the video
def capture(camera_ID, video_number):

    # camera parameters
    width = 320
    height = 240
    brightness = 0.498039215803
    contrast = 0.6
    saturation = 0.3
    fps = 30

    # bey
    brightness = 0.6
    saturation = 0.6

    # OpnenCampus
    width = 640
    height = 480
    # width = 320
    # height = 240
    brightness = 0.501960813999
    # brightness = 0.8
    contrast = 0.1254902035
    saturation = 0.109803922474


    # settings of FPS counter
    count_fps = False
    gFrameRate = FrameRate()
    fontcolor = (255,255,255)
    fontface  = cv2.FONT_HERSHEY_SIMPLEX
    fontthick = 2

    cap = cv2.VideoCapture(camera_ID)

    print '\n'+'dafault values'
    print 'width: '+str(cap.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH))
    print 'height: '+str(cap.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT))

    #print 'frame rate: '+str(cap.get(cv2.cv.CV_CAP_PROP_FPS))

    print 'brightness: '+str(cap.get(cv2.cv.CV_CAP_PROP_BRIGHTNESS))
    print 'contrast: '+str(cap.get(cv2.cv.CV_CAP_PROP_CONTRAST))
    print 'saturation: '+str(cap.get(cv2.cv.CV_CAP_PROP_SATURATION))+'\n'

    cap.set(cv2.cv.CV_CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.cv.CV_CAP_PROP_BRIGHTNESS, brightness)
    cap.set(cv2.cv.CV_CAP_PROP_CONTRAST, contrast)
    cap.set(cv2.cv.CV_CAP_PROP_SATURATION, saturation)
    cap.set(cv2.cv.CV_CAP_PROP_FPS, fps)

    save_path = 'videos/'

    if not os.path.isdir(save_path):
        os.makedirs(save_path)

    rec = initWriter(video_number, width, height, fps, save_path)
    cv2.namedWindow('camera:'+str(camera_ID), cv2.WINDOW_NORMAL)
    save_flag = False

    while(True):

        ret, frame = cap.read()

        if ret == False:
            print 'error'
            break

        if count_fps == True:

            fps = gFrameRate.get()
            fps_str = '%4d' % fps
            cv2.putText(frame, fps_str, (10,25), fontface, 1.0, fontcolor, fontthick , cv2.CV_AA)

        k = cv2.waitKey(1)

        if k & 0xFF == ord('q'):
            break

        if k & 0xFF == ord('s'):
            print 'start recording'
            save_flag = True

        if save_flag:
            rec.write(frame)

        frame = cv2.flip(frame, -1)

        cv2.imshow('camera:'+str(camera_ID), frame)

    cap.release()
    rec.release()


#main
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Camera streaming')
    parser.add_argument('--ID', '-i', type=int, default=0,help='Camera ID')
    parser.add_argument('--video_name', '-n', type=str, default=1,help='video file name')
    args = parser.parse_args()

    capture(args.ID, args.video_name)
    #cv2.dstroyAllWindows()
