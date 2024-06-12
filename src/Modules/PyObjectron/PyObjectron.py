import pyjevois
if pyjevois.pro: import libjevoispro as jevois
else: import libjevois as jevois
import cv2
import numpy as np
import mediapipe as mp

## 3D object detection using MediaPipe
#
# Detect objects and draw estimated 3D bounding boxes, using MediaPipe in Python
#
# Only works for a few pre-trained objects: 'Shoe', 'Chair', 'Cup', 'Camera', with Shoe selected by default. So
# point the camera to your shoes and see what happens...
# 
# This code is derived from sample_objectron.py at https://github.com/Kazuhito00/mediapipe-python-sample
#
# @author Laurent Itti
# 
# @videomapping JVUI 0 0 30.0 CropScale=RGB24@512x288:YUYV 1920 1080 30.0 JeVois PyObjectron
# @email itti\@usc.edu
# @address University of Southern California, HNB-07A, 3641 Watt Way, Los Angeles, CA 90089-2520, USA
# @copyright Copyright (C) 2022 by Laurent Itti, iLab and the University of Southern California
# @mainurl http://jevois.org
# @supporturl http://jevois.org/doc
# @otherurl http://iLab.usc.edu
# @license GPL v3
# @distribution Unrestricted
# @restrictions None
# @ingroup modules
class PyObjectron:
    # ###################################################################################################
    ## Constructor
    def __init__(self):
        # Parameters:
        max_num_objects = 5
        min_detection_confidence = 0.5
        min_tracking_confidence = 0.99
        self.model_name = 'Shoe' # 'Shoe', 'Chair', 'Cup', 'Camera'

        # Instantiate a JeVois Timer to measure our processing framerate:
        self.timer = jevois.Timer("objectron", 30, jevois.LOG_DEBUG)

        # Instantiate mediapipe model:
        self.mp_objectron = mp.solutions.objectron
        self.objectron = self.mp_objectron.Objectron(
            static_image_mode = False,
            max_num_objects = max_num_objects,
            min_detection_confidence = min_detection_confidence,
            min_tracking_confidence = min_tracking_confidence,
            model_name = self.model_name)

    # ###################################################################################################
    ## Process function with GUI output
    def processGUI(self, inframe, helper):
        # Start a new display frame, gets its size and also whether mouse/keyboard are idle:
        idle, winw, winh = helper.startFrame()

        # Draw full-resolution input frame from camera:
        x, y, w, h = helper.drawInputFrame("c", inframe, False, False)
        helper.itext('JeVois-Pro 3D Object Detection', 0, -1)
        helper.itext('Detecting: ' + self.model_name + ' - edit source code to change')
        
        # Get the next camera image at processing resolution (may block until it is captured):
        image = inframe.getCvRGBp()
        iw, ih = image.shape[1], image.shape[0]

        # Start measuring image processing time:
        self.timer.start()
        
        # Run graph:
        results = self.objectron.process(image)

        # Draw results:
        if results.detected_objects is not None:
            for detected_object in results.detected_objects:
                self.draw_landmarks(helper, iw, ih, detected_object.landmarks_2d)
                self.draw_axis(helper, iw, ih, detected_object.rotation, detected_object.translation, 0.1)
                
        # Write frames/s info from our timer:
        fps = self.timer.stop()
        helper.iinfo(inframe, fps, winw, winh);

        # End of frame:
        helper.endFrame()


    # ###################################################################################################
    def draw_landmarks(self, helper, iw, ih, landmarks):
        col = 0xff00ff00
        idx_to_coordinates = {}

        for index, landmark in enumerate(landmarks.landmark):
            lm = helper.i2d(landmark.x * iw, landmark.y * ih, "c")
            helper.drawCircle(lm.x, lm.y, 5, col, True)
            idx_to_coordinates[index] = lm

        # See https://github.com/google/mediapipe/blob/master/mediapipe/modules/objectron/calculators/box.h
        # for the 8 vertex locations. Code here derived from mediapipe/python/solutions/drawing_utils.py
        connections = self.mp_objectron.BOX_CONNECTIONS
        if connections:
            num_landmarks = len(landmarks.landmark)
            for connection in connections:
                start_idx = connection[0]
                end_idx = connection[1]
                if not (0 <= start_idx < num_landmarks and 0 <= end_idx < num_landmarks):
                    raise ValueError(f'Landmark index is out of range. Invalid connection '
                                     f'from landmark #{start_idx} to landmark #{end_idx}.')
                if start_idx in idx_to_coordinates and end_idx in idx_to_coordinates:
                    helper.drawLine(idx_to_coordinates[start_idx].x, idx_to_coordinates[start_idx].y,
                                    idx_to_coordinates[end_idx].x, idx_to_coordinates[end_idx].y, col)

    # ###################################################################################################
    def draw_axis(self, helper, iw, ih, rotation, translation, axis_length):
        focal_length = (1.0, 1.0)
        principal_point = (0.0, 0.0)
        axis_world = np.float32([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
        axis_cam = np.matmul(rotation, axis_length * axis_world.T).T + translation
        x = axis_cam[..., 0]
        y = axis_cam[..., 1]
        z = axis_cam[..., 2]
        
        # Project 3D points to NDC space.
        fx, fy = focal_length
        px, py = principal_point
        x_ndc = np.clip(-fx * x / (z + 1e-5) + px, -1., 1.)
        y_ndc = np.clip(-fy * y / (z + 1e-5) + py, -1., 1.)

        # Convert from NDC space to image space.
        x_im = (1 + x_ndc) * 0.5 * iw
        y_im = (1 - y_ndc) * 0.5 * ih

        # Draw, converting coords from low-res processing frame to full-res display frame:
        orig = helper.i2d(x_im[0], y_im[0], "c")
        xa = helper.i2d(x_im[1], y_im[1], "c")
        ya = helper.i2d(x_im[2], y_im[2], "c")
        za = helper.i2d(x_im[3], y_im[3], "c")
        helper.drawLine(orig.x, orig.y, xa.x, xa.y, 0xff0000ff)
        helper.drawLine(orig.x, orig.y, ya.x, ya.y, 0xff00ff00)
        helper.drawLine(orig.x, orig.y, za.x, za.y, 0xffff0000)

