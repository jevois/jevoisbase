import libjevois as jevois
import cv2 as cv
import numpy as np

## Human facial emotion recognition using OpenCV Deep Neural Networks (DNN)
#
# This module runs an emotion classification deep neural network using the OpenCV DNN library. The network is from the
# FER+ emotion recognition project, "Training Deep Networks for Facial Expression Recognition with Crowd-Sourced Label
# Distribution" arXiv:1608.01041
#
# The module outputs a score from 0 to 100 for each of: neutral, happiness, surprise, sadness, anger, disgust, fear,
# contempt.
#
# @author Laurent Itti
# 
# @videomapping YUYV 320 336 15.0 YUYV 320 336 15.0 JeVois PyEmotion
# @email itti@usc.edu
# @address 880 W 1st St Suite 807, Los Angeles CA 90012, USA
# @copyright Copyright (C) 2019 by Laurent Itti
# @mainurl http://jevois.org
# @supporturl http://jevois.org
# @otherurl http://jevois.org
# @license GPL v3
# @distribution Unrestricted
# @restrictions None
# @ingroup modules
class PyEmotion:
    # ####################################################################################################
    ## Constructor
    def __init__(self):
        self.confThreshold = 0.2 # Confidence threshold (0..1), higher for stricter confidence.
        self.inpWidth = 64       # Resized image width passed to network
        self.inpHeight = 64      # Resized image height passed to network
        self.scale = 1.0         # Value scaling factor applied to input pixels
        self.mean = [104, 117, 123] # Mean BGR value subtracted from input image
        self.rgb = False         # True if model expects RGB inputs, otherwise it expects BGR

        # You should not have to edit anything beyond this point.
        backend = cv.dnn.DNN_BACKEND_DEFAULT
        target = cv.dnn.DNN_TARGET_CPU
        self.classes = [ "neutral", "happiness", "surprise", "sadness", "anger", "disgust",
                         "fear", "contempt", "unknown", "NF" ]
        modelname = '/jevois/share/opencv-dnn/classification/emotion_ferplus.onnx'
        configname = ''
        model = 'Fer+ ONNX'

        # Load the network
        self.net = cv.dnn.readNet(modelname, configname)
        self.net.setPreferableBackend(backend)
        self.net.setPreferableTarget(target)
        self.timer = jevois.Timer('Neural emotion', 10, jevois.LOG_DEBUG)
        self.model = model
        
    # ####################################################################################################
    ## JeVois main processing function
    def process(self, inframe, outframe):
        frame = inframe.getCvBGR()
        self.timer.start()
        
        frameHeight = frame.shape[0]
        frameWidth = frame.shape[1]

        # Create a 4D blob from a frame.
        gframe = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        blob = cv.dnn.blobFromImage(gframe, self.scale, (self.inpWidth, self.inpHeight), self.mean, self.rgb, crop=True)

        # Run the model
        self.net.setInput(blob)
        out = self.net.forward()

        # Get a class with a highest score:
        out = out.flatten()
        classId = np.argmax(out)
        confidence = out[classId]

        # Create dark-gray (value 80) image for the bottom panel, 96 pixels tall and show top-1 class:
        msgbox = np.zeros((96, frame.shape[1], 3), dtype = np.uint8) + 80
        rlabel = ' '
        if (confidence > self.confThreshold):
            rlabel = '%s: %.2f' % (self.classes[classId] if self.classes else 'Class #%d' % classId, confidence*100)

        cv.putText(msgbox, rlabel, (3, 15), cv.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv.LINE_AA)

        # Put efficiency information.
        cv.putText(frame, 'JeVois Emotion DNN - ' + self.model, (3, 15),
                   cv.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv.LINE_AA)
        t, _ = self.net.getPerfProfile()
        fps = self.timer.stop()
        label = fps + ', %dms' % (t * 1000.0 / cv.getTickFrequency())
        cv.putText(frame, label, (3, frameHeight-5), cv.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv.LINE_AA)
        
        # Stack bottom panel below main image:
        frame = np.vstack((frame, msgbox))

        # Send output frame to host:
        outframe.sendCv(frame)
