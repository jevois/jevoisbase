import pyjevois
if pyjevois.pro: import libjevoispro as jevois
else: import libjevois as jevois
import cv2
import numpy as np
import mediapipe as mp

## Hand detection using MediaPipe
#
# Detect hands using MediaPipe in Python
#
# This code is derived from sample_hand.py at https://github.com/Kazuhito00/mediapipe-python-sample
#
# @author Laurent Itti
# 
# @videomapping JVUI 0 0 30.0 CropScale=RGB24@512x288:YUYV 1920 1080 30.0 JeVois PyHandDetector
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
class PyHandDetector:
    # ###################################################################################################
    ## Constructor
    def __init__(self):
        # Instantiate a JeVois Timer to measure our processing framerate:
        self.timer = jevois.Timer("hand", 100, jevois.LOG_INFO)

        # Instantiate mediapipe hand detector:
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(max_num_hands = 2, min_detection_confidence = 0.7,
                                         min_tracking_confidence = 0.5)
        self.use_brect = True; # true to show a bounding rectangle around each hand
        
    # ###################################################################################################
    ## Process function with GUI output
    def processGUI(self, inframe, helper):
        # Start a new display frame, gets its size and also whether mouse/keyboard are idle:
        idle, winw, winh = helper.startFrame()

        # Draw full-resolution input frame from camera:
        x, y, w, h = helper.drawInputFrame("c", inframe, False, False)
        
        # Get the next camera image at processing resolution (may block until it is captured):
        image = inframe.getCvRGBp()
        iw, ih = image.shape[1], image.shape[0]
        
        # Start measuring image processing time:
        self.timer.start()

        # Detect hands:
        results = self.hands.process(image)

        # Draw results:
        if results.multi_hand_landmarks is not None:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                cx, cy = calc_palm_moment(iw, ih, hand_landmarks)
                draw_landmarks(helper, iw, ih, cx, cy, hand_landmarks, handedness)
                if self.use_brect:
                    brect = calc_bounding_rect(iw, ih, hand_landmarks)
                    helper.drawRect(brect[0], brect[1], brect[2], brect[3], 0x6040ffff, True)

        # Write frames/s info from our timer:
        fps = self.timer.stop()
        helper.iinfo(inframe, fps, winw, winh);

        # End of frame:
        helper.endFrame()
        



# ###################################################################################################
def calc_palm_moment(iw, ih, landmarks):
    palm_array = np.empty((0, 2), int)

    for index, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * iw), iw - 1)
        landmark_y = min(int(landmark.y * ih), ih - 1)

        landmark_point = [np.array((landmark_x, landmark_y))]

        if index == 0:
            palm_array = np.append(palm_array, landmark_point, axis=0)
        if index == 1:
            palm_array = np.append(palm_array, landmark_point, axis=0)
        if index == 5:
            palm_array = np.append(palm_array, landmark_point, axis=0)
        if index == 9:
            palm_array = np.append(palm_array, landmark_point, axis=0)
        if index == 13:
            palm_array = np.append(palm_array, landmark_point, axis=0)
        if index == 17:
            palm_array = np.append(palm_array, landmark_point, axis=0)

    M = cv2.moments(palm_array)
    cx, cy = 0, 0
    if M['m00'] != 0:
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
    return cx, cy

# ###################################################################################################
def calc_bounding_rect(iw, ih, landmarks):
    landmark_array = np.empty((0, 2), int)

    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * iw), iw - 1)
        landmark_y = min(int(landmark.y * ih), ih - 1)
        landmark_point = [np.array((landmark_x, landmark_y))]
        landmark_array = np.append(landmark_array, landmark_point, axis=0)
    x, y, ww, hh = cv2.boundingRect(landmark_array)
    return [x, y, x + ww, y + hh]

# ###################################################################################################
def draw_landmarks(helper, iw, ih, cx, cy, landmarks, handedness):
    lpx = []
    lpy = []
    col = 0xff00ff00
        
    for index, landmark in enumerate(landmarks.landmark):
        if landmark.visibility < 0 or landmark.presence < 0:
            continue

        landmark_x = min(int(landmark.x * iw), iw - 1)
        landmark_y = min(int(landmark.y * ih), ih - 1)
        # landmark_z = landmark.z

        lpx.append(landmark_x)
        lpy.append(landmark_y)

        if index == 0:
            helper.drawCircle(landmark_x, landmark_y, 5, col, True)
        if index == 1:
            helper.drawCircle(landmark_x, landmark_y, 5, col, True)
        if index == 2:
            helper.drawCircle(landmark_x, landmark_y, 5, col, True)
        if index == 3:
            helper.drawCircle(landmark_x, landmark_y, 5, col, True)
        if index == 4:
            helper.drawCircle(landmark_x, landmark_y, 5, col, False)
            helper.drawCircle(landmark_x, landmark_y, 12, col, True)
        if index == 5:
            helper.drawCircle(landmark_x, landmark_y, 5, col, True)
        if index == 6:
            helper.drawCircle(landmark_x, landmark_y, 5, col, True)
        if index == 7:
            helper.drawCircle(landmark_x, landmark_y, 5, col, True)
        if index == 8:
            helper.drawCircle(landmark_x, landmark_y, 5, col, False)
            helper.drawCircle(landmark_x, landmark_y, 12, col, True)
        if index == 9:
            helper.drawCircle(landmark_x, landmark_y, 5, col, True)
        if index == 10:
            helper.drawCircle(landmark_x, landmark_y, 5, col, True)
        if index == 11:
            helper.drawCircle(landmark_x, landmark_y, 5, col, True)
        if index == 12:
            helper.drawCircle(landmark_x, landmark_y, 5, col, False)
            helper.drawCircle(landmark_x, landmark_y, 12, col, True)
        if index == 13:
            helper.drawCircle(landmark_x, landmark_y, 5, col, True)
        if index == 14:
            helper.drawCircle(landmark_x, landmark_y, 5, col, True)
        if index == 15:
            helper.drawCircle(landmark_x, landmark_y, 5, col, True)
        if index == 16:
            helper.drawCircle(landmark_x, landmark_y, 5, col, False)
            helper.drawCircle(landmark_x, landmark_y, 12, col, True)
        if index == 17:
            helper.drawCircle(landmark_x, landmark_y, 5, col, True)
        if index == 18:
            helper.drawCircle(landmark_x, landmark_y, 5, col, True)
        if index == 19:
            helper.drawCircle(landmark_x, landmark_y, 5, col, True)
        if index == 20:
            helper.drawCircle(landmark_x, landmark_y, 5, col, False)
            helper.drawCircle(landmark_x, landmark_y, 12, col, True)

    if len(lpx) > 0:
        helper.drawLine(lpx[2], lpy[2], lpx[3], lpy[3], col)
        helper.drawLine(lpx[3], lpy[3], lpx[4], lpy[4], col)

        helper.drawLine(lpx[5], lpy[5], lpx[6], lpy[6], col)
        helper.drawLine(lpx[6], lpy[6], lpx[7], lpy[7], col)
        helper.drawLine(lpx[7], lpy[7], lpx[8], lpy[8], col)

        helper.drawLine(lpx[9], lpy[9], lpx[10], lpy[10], col)
        helper.drawLine(lpx[10], lpy[10], lpx[11], lpy[11], col)
        helper.drawLine(lpx[11], lpy[11], lpx[12], lpy[12], col)

        helper.drawLine(lpx[13], lpy[13], lpx[14], lpy[14], col)
        helper.drawLine(lpx[14], lpy[14], lpx[15], lpy[15], col)
        helper.drawLine(lpx[15], lpy[15], lpx[16], lpy[16], col)

        helper.drawLine(lpx[17], lpy[17], lpx[18], lpy[18], col)
        helper.drawLine(lpx[18], lpy[18], lpx[19], lpy[19], col)
        helper.drawLine(lpx[19], lpy[19], lpx[20], lpy[20], col)

        helper.drawLine(lpx[0], lpy[0], lpx[1], lpy[1], col)
        helper.drawLine(lpx[1], lpy[1], lpx[2], lpy[2], col)
        helper.drawLine(lpx[2], lpy[2], lpx[5], lpy[5], col)
        helper.drawLine(lpx[5], lpy[5], lpx[9], lpy[9], col)
        helper.drawLine(lpx[9], lpy[9], lpx[13], lpy[13], col)
        helper.drawLine(lpx[13], lpy[13], lpx[17], lpy[17], col)
        helper.drawLine(lpx[17], lpy[17], lpx[0], lpy[0], col)

    if len(lpx) > 0:
        helper.drawCircle(cx, cy, 5, 0xffff8080, True)
        # If camera view was not flipped horizontally, swap left/right hands:
        if handedness.classification[0].label[0] == 'L': hnd = 'R'
        else: hnd = 'L'
        helper.drawText(cx - 2, cy - 3, hnd, 0xffff0000)
