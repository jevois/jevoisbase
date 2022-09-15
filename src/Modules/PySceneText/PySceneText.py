import pyjevois
if pyjevois.pro: import libjevoispro as jevois
else: import libjevois as jevois
import cv2
import numpy as np

from crnn import CRNN
from db import DB

## Detect and decode English or Chinese text on NPU using TIM-VX
#
# This module runs on the JeVois-Pro NPU using a quantized deep neural network. It is derived from
# https://github.com/opencv/opencv_zoo/tree/master/models/text_recognition_crnn
# See LICENSE for license information.
#
# By default, English is used, but you can change that to Chinese in the constructor
#
# This module is mainly intended as a tutorial for how to run quantized int8 DNNs on the NPU using OpenCV and TIM-VX,
# here achieved through only small modifications of code from https://github.com/opencv/opencv_zoo - in particular,
# the core class for this model, LPD_YuNet, was not modified at all, and only the demo code was edited to use the
# JeVois GUI Helper for fast OpenGL drawing as opposed to slow drawings into OpenCV images.
#
# @author Laurent Itti
# 
# @displayname PySceneText
# @videomapping JVUI 0 0 30.0 CropScale=RGB24@512x288:YUYV 1920 1080 30.0 JeVois PySceneText
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
class PySceneText:
    # ###################################################################################################
    ## Constructor
    def __init__(self):
        # Select a language here:
        lang = "English"
        #lang = "Chinese"
        
        # Instantiate the model
        if lang == "English":
            self.modelname = "text_recognition_CRNN_EN_2021sep-act_int8-wt_int8-quantized.onnx"
            self.charset = "charset_36_EN.txt"
            self.tdname = "text_detection_DB_IC15_resnet18_2021sep.onnx"
        elif lang == "Chinese":
            self.modelname = "text_recognition_CRNN_CN_2021nov-act_int8-wt_int8-quantized.onnx"
            self.charset = "charset_3944_CN.txt"
            self.tdname = "text_detection_DB_TD500_resnet18_2021sep.onnx"
        else:
            jevois.LFATAL("Invalid language selected")

        root = "/jevoispro/share/npu/other/"
        
        self.recognizer = CRNN(modelPath = root + self.modelname, charsetPath = root + self.charset)

        # Instantiate DB for text detection
        self.detector = DB(modelPath = root + self.tdname,
                           inputSize = [512, 288],
                           binaryThreshold = 0.3,
                           polygonThreshold = 0.5,
                           maxCandidates = 200,
                           unclipRatio = 2.0,
                           backendId = cv2.dnn.DNN_BACKEND_TIMVX,
                           targetId = cv2.dnn.DNN_TARGET_NPU)

        # Instantiate a timer for framerate:
        self.timer = jevois.Timer('PySceneText', 10, jevois.LOG_DEBUG)

        # Keep track of frame size:
        self.w, self.h = (0, 0)
        
    # ####################################################################################################
    ## Process function with GUI output
    def processGUI(self, inframe, helper):
        # Start a new display frame, gets its size and also whether mouse/keyboard are idle:
        idle, winw, winh = helper.startFrame()

        # Draw full-resolution input frame from camera:
        dx, dy, dw, dh = helper.drawInputFrame("c", inframe, False, False)
        
        # Get the next camera image at processing resolution (may block until it is captured):
        frame = inframe.getCvBGRp()
        h, w, _ = frame.shape

        # Start measuring image processing time:
        self.timer.start()

        w,h = (512,288)
        frame = cv2.resize(frame, [w, h])

        # Input size must be a multiple of 32 in this model
        if (w % 32) != 0 or (h % 32) != 0:
            jevois.LFATAL("Input width and height must be multiples of 32")
        
        # Resize model if needed:
        if self.w != w or self.h != h:
            self.detector.setInputSize([w, h])
            self.w, self.h = (w, h)

        # Inference
        results = self.detector.infer(frame)
        texts = []
        for box, score in zip(results[0], results[1]):
            texts.append(self.recognizer.infer(frame, box.reshape(8)))

        # Draw results:
        pts = np.array(results[0]).astype(np.single)
        helper.drawPoly(pts, 0xff0000ff, True)
        for box, text in zip(results[0], texts):
            helper.drawText(float(box[1][0]), float(box[1][1]), text, 0xff0000ff);
        
        # Write frames/s info from our timer:
        fps = self.timer.stop()
        helper.iinfo(inframe, fps, winw, winh);
        helper.itext("JeVois-Pro - Scene text detection and decoding", 0, -1)
        helper.itext("Detection:   " + self.tdname, 0, -1)
        helper.itext("Charset:     " + self.charset, 0, -1)
        helper.itext("Recognition: " + self.modelname, 0, -1)

        # End of frame:
        helper.endFrame()
