import pyjevois
if pyjevois.pro: import libjevoispro as jevois
else: import libjevois as jevois
import cv2
import numpy as np
import mediapipe as mp

## Human body pose detection using MediaPipe
#
# Detect human body pose using MediaPipe in Python
#
# This code is derived from sample_pose.py at https://github.com/Kazuhito00/mediapipe-python-sample
#
# @author Laurent Itti
# 
# @videomapping JVUI 0 0 30.0 CropScale=RGB24@512x288:YUYV 1920 1080 30.0 JeVois PyPoseDetector
# @email itti\@usc.edu
# @address University of Southern California, HNB-07A, 3641 Watt Way, Los Angeles, CA 90089-2520, USA
# @copyright Copyright (C) 2021 by Laurent Itti, iLab and the University of Southern California
# @mainurl http://jevois.org
# @supporturl http://jevois.org/doc
# @otherurl http://iLab.usc.edu
# @license GPL v3
# @distribution Unrestricted
# @restrictions None
# @ingroup modules
class PyPoseDetector:
    # ###################################################################################################
    ## Constructor
    def __init__(self):
        # Instantiate a JeVois Timer to measure our processing framerate:
        self.timer = jevois.Timer("pose", 100, jevois.LOG_INFO)

        # Instantiate mediapipe pose detector. model_complexity should be 0, 1, or 2:
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(model_complexity = 1, min_detection_confidence = 0.5,
                                      min_tracking_confidence = 0.5)
        self.use_brect = False; # true to show a bounding rectangle around each hand
        
    # ###################################################################################################
    ## Process function with GUI output
    def processGUI(self, inframe, helper):
        # Start a new display frame, gets its size and also whether mouse/keyboard are idle:
        idle, winw, winh = helper.startFrame()

        # Draw full-resolution input frame from camera:
        x, y, w, h = helper.drawInputFrame("c", inframe, False, False)
        helper.itext('JeVois-Pro Body Pose Skeleton Detection', 0, -1)

        # Get the next camera image at processing resolution (may block until it is captured):
        image = inframe.getCvRGBp()
        iw, ih = image.shape[1], image.shape[0]
        
        # Start measuring image processing time:
        self.timer.start()

        # Detect pose:
        results = self.pose.process(image)

        # Draw results:
        if results.pose_landmarks is not None:
            draw_landmarks(helper, iw, ih, results.pose_landmarks, visibility_th = 0.5)
            if self.use_brect:
                brect = calc_bounding_rect(iw, ih, results.pose_landmarks)
                helper.drawRect(brect[0], brect[1], brect[2], brect[3], 0x6040ffff, True)

        # Write frames/s info from our timer:
        fps = self.timer.stop()
        helper.iinfo(inframe, fps, winw, winh);

        # End of frame:
        helper.endFrame()
        
# ###################################################################################################
def calc_bounding_rect(iw, ih, landmarks):
    landmark_array = np.empty((0, 2), int)

    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * iw), iw - 1)
        landmark_y = min(int(landmark.y * ih), ih - 1)
        landmark_point = [np.array((landmark_x, landmark_y))]
        landmark_array = np.append(landmark_array, landmark_point, axis=0)
    x, y, w, h = cv2.boundingRect(landmark_array)
    return [x, y, x + w, y + h]

# ###################################################################################################
def draw_line(helper, lp1, lp2, visibility_th, col):
    if lp1[0] > visibility_th and lp2[0] > visibility_th:
        helper.drawLine(lp1[1][0], lp1[1][1], lp2[1][0], lp2[1][1], col)
    
# ###################################################################################################
def draw_landmarks(helper, iw, ih, landmarks, visibility_th):
    lp = []
    col = 0xff00ff00

    for index, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * iw), iw - 1)
        landmark_y = min(int(landmark.y * ih), ih - 1)
        landmark_z = landmark.z
        lp.append([landmark.visibility, (landmark_x, landmark_y)])

        if landmark.visibility >= visibility_th:
            helper.drawCircle(landmark_x, landmark_y, 5, col, True)
            helper.drawText(landmark_x - 7, landmark_y - 7, "z:" + str(round(landmark_z, 3)), 0xffff8080)

    if len(lp) > 0:
        draw_line(helper, lp[1], lp[2], visibility_th, col)
        draw_line(helper, lp[2], lp[3], visibility_th, col)
        draw_line(helper, lp[4], lp[5], visibility_th, col)
        draw_line(helper, lp[5], lp[6], visibility_th, col)
        draw_line(helper, lp[9], lp[10], visibility_th, col)
        draw_line(helper, lp[11], lp[12], visibility_th, col)
        draw_line(helper, lp[11], lp[13], visibility_th, col)
        draw_line(helper, lp[13], lp[15], visibility_th, col)
        draw_line(helper, lp[12], lp[14], visibility_th, col)
        draw_line(helper, lp[14], lp[16], visibility_th, col)
        draw_line(helper, lp[15], lp[17], visibility_th, col)
        draw_line(helper, lp[17], lp[19], visibility_th, col)
        draw_line(helper, lp[19], lp[21], visibility_th, col)
        draw_line(helper, lp[21], lp[15], visibility_th, col)
        draw_line(helper, lp[16], lp[18], visibility_th, col)
        draw_line(helper, lp[18], lp[20], visibility_th, col)
        draw_line(helper, lp[20], lp[22], visibility_th, col)
        draw_line(helper, lp[22], lp[16], visibility_th, col)
        draw_line(helper, lp[11], lp[23], visibility_th, col)
        draw_line(helper, lp[12], lp[24], visibility_th, col)
        draw_line(helper, lp[23], lp[24], visibility_th, col)
        
        if len(lp) > 25:
            draw_line(helper, lp[23], lp[25], visibility_th, col)
            draw_line(helper, lp[25], lp[27], visibility_th, col)
            draw_line(helper, lp[27], lp[29], visibility_th, col)
            draw_line(helper, lp[29], lp[31], visibility_th, col)
            draw_line(helper, lp[24], lp[26], visibility_th, col)
            draw_line(helper, lp[26], lp[28], visibility_th, col)
            draw_line(helper, lp[28], lp[30], visibility_th, col)
            draw_line(helper, lp[30], lp[32], visibility_th, col)
