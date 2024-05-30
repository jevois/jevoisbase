import pyjevois
if pyjevois.pro: import libjevoispro as jevois
else: import libjevois as jevois
import cv2
import numpy as np
import mediapipe as mp

## Face mesh detection using MediaPipe
#
# Detect face landmarks using MediaPipe in Python
#
# This code is derived from sample_facemesh.py at https://github.com/Kazuhito00/mediapipe-python-sample
#
# @author Laurent Itti
# 
# @videomapping JVUI 0 0 30.0 CropScale=RGB24@512x288:YUYV 1920 1080 30.0 JeVois PyFaceMesh
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
class PyFaceMesh:
    # ###################################################################################################
    ## Constructor
    def __init__(self):
        # Instantiate a JeVois Timer to measure our processing framerate:
        self.timer = jevois.Timer("mesh", 100, jevois.LOG_INFO)

        # Instantiate mediapipe face mesh:
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(max_num_faces = 1, min_detection_confidence = 0.7,
                                                    min_tracking_confidence = 0.5)
        self.use_brect = True; # true to show a bounding rectangle
        
    # ###################################################################################################
    ## Process function with GUI output
    def processGUI(self, inframe, helper):
        # Start a new display frame, gets its size and also whether mouse/keyboard are idle:
        idle, winw, winh = helper.startFrame()

        # Draw full-resolution input frame from camera:
        x, y, w, h = helper.drawInputFrame("c", inframe, False, False)
        helper.itext('JeVois-Pro Face Landmarks Detection')
        
        # Get the next camera image at processing resolution (may block until it is captured):
        image = inframe.getCvRGBp()
        iw, ih = image.shape[1], image.shape[0]
        
        # Start measuring image processing time:
        self.timer.start()

        # Detect face landmarks:
        results = self.face_mesh.process(image)

        # Draw results:
        if results.multi_face_landmarks is not None:
            for face_landmarks in results.multi_face_landmarks:
                self.draw_landmarks(helper, iw, ih, face_landmarks)
                if self.use_brect:
                    brect = self.calc_bounding_rect(iw, ih, face_landmarks)
                    helper.drawRect(brect[0], brect[1], brect[2], brect[3], 0x6040ffff, True)

        # Write frames/s info from our timer:
        fps = self.timer.stop()
        helper.iinfo(inframe, fps, winw, winh);

        # End of frame:
        helper.endFrame()
 
    # ###################################################################################################
    def calc_bounding_rect(self, iw, ih, landmarks):
        landmark_array = np.empty((0, 2), int)

        for _, landmark in enumerate(landmarks.landmark):
            landmark_x = min(int(landmark.x * iw), iw - 1)
            landmark_y = min(int(landmark.y * ih), ih - 1)
            landmark_point = [np.array((landmark_x, landmark_y))]
            landmark_array = np.append(landmark_array, landmark_point, axis=0)
        x, y, w, h = cv2.boundingRect(landmark_array)
        return [x, y, x + w, y + h]

    # ###################################################################################################
    def draw_landmarks(self, helper, iw, ih, landmarks):
        lpx = []
        lpy = []
        col = 0xff00ff00

        for index, landmark in enumerate(landmarks.landmark):
            if landmark.visibility < 0 or landmark.presence < 0: continue

            landmark_x = min(int(landmark.x * iw), iw - 1)
            landmark_y = min(int(landmark.y * ih), ih - 1)
            lpx.append(landmark_x)
            lpy.append(landmark_y)

            helper.drawCircle(landmark_x, landmark_y, 1, col, False)

        if len(lpx) > 0:
            helper.drawLine(lpx[55], lpy[55], lpx[65], lpy[65], col)
            helper.drawLine(lpx[65], lpy[65], lpx[52], lpy[52], col)
            helper.drawLine(lpx[52], lpy[52], lpx[53], lpy[53], col)
            helper.drawLine(lpx[53], lpy[53], lpx[46], lpy[46], col)
            
            helper.drawLine(lpx[285], lpy[285], lpx[295], lpy[295], col)
            helper.drawLine(lpx[295], lpy[295], lpx[282], lpy[282], col)
            helper.drawLine(lpx[282], lpy[282], lpx[283], lpy[283], col)
            helper.drawLine(lpx[283], lpy[283], lpx[276], lpy[276], col)
            
            helper.drawLine(lpx[133], lpy[133], lpx[173], lpy[173], col)
            helper.drawLine(lpx[173], lpy[173], lpx[157], lpy[157], col)
            helper.drawLine(lpx[157], lpy[157], lpx[158], lpy[158], col)
            helper.drawLine(lpx[158], lpy[158], lpx[159], lpy[159], col)
            helper.drawLine(lpx[159], lpy[159], lpx[160], lpy[160], col)
            helper.drawLine(lpx[160], lpy[160], lpx[161], lpy[161], col)
            helper.drawLine(lpx[161], lpy[161], lpx[246], lpy[246], col)
            
            helper.drawLine(lpx[246], lpy[246], lpx[163], lpy[163], col)
            helper.drawLine(lpx[163], lpy[163], lpx[144], lpy[144], col)
            helper.drawLine(lpx[144], lpy[144], lpx[145], lpy[145], col)
            helper.drawLine(lpx[145], lpy[145], lpx[153], lpy[153], col)
            helper.drawLine(lpx[153], lpy[153], lpx[154], lpy[154], col)
            helper.drawLine(lpx[154], lpy[154], lpx[155], lpy[155], col)
            helper.drawLine(lpx[155], lpy[155], lpx[133], lpy[133], col)
            
            helper.drawLine(lpx[362], lpy[362], lpx[398], lpy[398], col)
            helper.drawLine(lpx[398], lpy[398], lpx[384], lpy[384], col)
            helper.drawLine(lpx[384], lpy[384], lpx[385], lpy[385], col)
            helper.drawLine(lpx[385], lpy[385], lpx[386], lpy[386], col)
            helper.drawLine(lpx[386], lpy[386], lpx[387], lpy[387], col)
            helper.drawLine(lpx[387], lpy[387], lpx[388], lpy[388], col)
            helper.drawLine(lpx[388], lpy[388], lpx[466], lpy[466], col)
            
            helper.drawLine(lpx[466], lpy[466], lpx[390], lpy[390], col)
            helper.drawLine(lpx[390], lpy[390], lpx[373], lpy[373], col)
            helper.drawLine(lpx[373], lpy[373], lpx[374], lpy[374], col)
            helper.drawLine(lpx[374], lpy[374], lpx[380], lpy[380], col)
            helper.drawLine(lpx[380], lpy[380], lpx[381], lpy[381], col)
            helper.drawLine(lpx[381], lpy[381], lpx[382], lpy[382], col)
            helper.drawLine(lpx[382], lpy[382], lpx[362], lpy[362], col)
            
            helper.drawLine(lpx[308], lpy[308], lpx[415], lpy[415], col)
            helper.drawLine(lpx[415], lpy[415], lpx[310], lpy[310], col)
            helper.drawLine(lpx[310], lpy[310], lpx[311], lpy[311], col)
            helper.drawLine(lpx[311], lpy[311], lpx[312], lpy[312], col)
            helper.drawLine(lpx[312], lpy[312], lpx[13], lpy[13], col)
            helper.drawLine(lpx[13], lpy[13], lpx[82], lpy[82], col)
            helper.drawLine(lpx[82], lpy[82], lpx[81], lpy[81], col)
            helper.drawLine(lpx[81], lpy[81], lpx[80], lpy[80], col)
            helper.drawLine(lpx[80], lpy[80], lpx[191], lpy[191], col)
            helper.drawLine(lpx[191], lpy[191], lpx[78], lpy[78], col)
            
            helper.drawLine(lpx[78], lpy[78], lpx[95], lpy[95], col)
            helper.drawLine(lpx[95], lpy[95], lpx[88], lpy[88], col)
            helper.drawLine(lpx[88], lpy[88], lpx[178], lpy[178], col)
            helper.drawLine(lpx[178], lpy[178], lpx[87], lpy[87], col)
            helper.drawLine(lpx[87], lpy[87], lpx[14], lpy[14], col)
            helper.drawLine(lpx[14], lpy[14], lpx[317], lpy[317], col)
            helper.drawLine(lpx[317], lpy[317], lpx[402], lpy[402], col)
            helper.drawLine(lpx[402], lpy[402], lpx[318], lpy[318], col)
            helper.drawLine(lpx[318], lpy[318], lpx[324], lpy[324], col)
            helper.drawLine(lpx[324], lpy[324], lpx[308], lpy[308], col)

