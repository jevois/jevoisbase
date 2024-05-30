import pyjevois
if pyjevois.pro: import libjevoispro as jevois
else: import libjevois as jevois
import cv2
import numpy as np

from lpd_yunet import LPD_YuNet

## Detect license plates on NPU using YuNet TIM-VX
#
# This module runs on the JeVois-Pro NPU using a quantized deep neural network. It is derived from
# https://github.com/opencv/opencv_zoo/tree/master/models/license_plate_detection_yunet
# See LICENSE for license information.
#
# Please note that the model is trained with Chinese license plates, so the detection results of other license plates
# with this model may be limited. See the screenshots of this module for examples, or search for "Chinese license plate"
# on the web for more images.
#
# This module is mainly intended as a tutorial for how to run quantized int8 DNNs on the NPU using OpenCV and TIM-VX,
# here achieved through only small modifications of code from https://github.com/opencv/opencv_zoo - in particular,
# the core class for this model, LPD_YuNet, was not modified at all, and only the demo code was edited to use the
# JeVois GUI Helper for fast OpenGL drawing as opposed to slow drawings into OpenCV images.
#
# @author Laurent Itti
# 
# @displayname PyLicensePlate
# @videomapping JVUI 0 0 30.0 CropScale=RGB24@512x288:YUYV 1920 1080 30.0 JeVois PyLicensePlate
# @videomapping JVUI 0 0 30.0 CropScale=RGB24@256x144:YUYV 1920 1080 30.0 JeVois PyLicensePlate
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
class PyLicensePlate:
    # ###################################################################################################
    ## Constructor
    def __init__(self):
        # Instantiate the model
        self.modelname = "license_plate_detection_lpd_yunet_2022may-int8-quantized.onnx"

        self.model = LPD_YuNet(modelPath = "/jevoispro/share/npu/other/" + self.modelname,
                               confThreshold = 0.9,
                               nmsThreshold = 0.3,
                               topK = 5000,
                               keepTopK = 750,
                               backendId = cv2.dnn.DNN_BACKEND_TIMVX,
                               targetId = cv2.dnn.DNN_TARGET_NPU)

        # Instantiate a timer for framerate:
        self.timer = jevois.Timer('PyLicensePlate', 30, jevois.LOG_DEBUG)

        # Keep track of frame size:
        self.h = 0
        self.w = 0
        
    # ####################################################################################################
    ## Process function with GUI output
    def processGUI(self, inframe, helper):
        # Start a new display frame, gets its size and also whether mouse/keyboard are idle:
        idle, winw, winh = helper.startFrame()

        # Draw full-resolution input frame from camera:
        dx, dy, dw, dh = helper.drawInputFrame("c", inframe, False, False)
        helper.itext('JeVois-Pro License Plate Detection')

        # Get the next camera image at processing resolution (may block until it is captured):
        frame = inframe.getCvBGRp()
        h, w, _ = frame.shape

        # Start measuring image processing time:
        self.timer.start()

        # Resize model if needed:
        if self.w != w or self.h != h:
            self.model.setInputSize([w, h])
            self.h = h
            self.w = w
        
        # Inference
        dets = self.model.infer(frame)

        # Draw the detections:
        for det in dets:
            bbox = det[:-1]
            x1, y1, x2, y2, x3, y3, x4, y4 = bbox

            # Draw the border of license plate
            helper.drawLine(float(x1), float(y1), float(x2), float(y2), 0xff0000ff) # color format is ABGR
            helper.drawLine(float(x2), float(y2), float(x3), float(y3), 0xff0000ff)
            helper.drawLine(float(x3), float(y3), float(x4), float(y4), 0xff0000ff)
            helper.drawLine(float(x4), float(y4), float(x1), float(y1), 0xff0000ff)
        
        # Write frames/s info from our timer:
        fps = self.timer.stop()
        helper.iinfo(inframe, fps, winw, winh);
        helper.itext("JeVois-Pro - " + self.modelname)

        # End of frame:
        helper.endFrame()
