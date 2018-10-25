import libjevois as jevois
import cv2 as cv
import numpy as np
import sys

## Object recognition using OpenCV Deep Neural Networks (DNN)
#
# This module runs an object classification deep neural network using the OpenCV DNN library. Classification
# (recognition) networks analyze a central portion of the whole scene and produce identity labels and confidence scores
# about what the object in the field of view might be.
#
# This module supports detection networks implemented in TensorFlow, Caffe,
# Darknet, Torch, ONNX, etc as supported by the OpenCV DNN module.
#
# Included with the standard JeVois distribution are:
#
# - SqueezeNet v1.1, Caffe model
# - more to come, please contribute!
#
# See the module's constructor (__init__) code and select a value for \b model to switch network.
#
# Object category names for models trained on ImageNet are at
# https://github.com/jevois/jevoisbase/blob/master/share/opencv-dnn/classification/synset_words.txt
#
# Sometimes it will make mistakes! The performance of SqueezeNet v1.1 is about 56.1% correct (mean average precision,
# top-1) on the ImageNet test set.
#
# This module is adapted from the sample OpenCV code:
# https://github.com/opencv/opencv/blob/master/samples/dnn/classification.py
#
# More pre-trained models are available on github in opencv_extra
#
#
# @author Laurent Itti
# 
# @videomapping YUYV 320 264 30.0 YUYV 320 240 30.0 JeVois PyClassificationDNN
# @email itti@usc.edu
# @address 880 W 1st St Suite 807, Los Angeles CA 90012, USA
# @copyright Copyright (C) 2018 by Laurent Itti
# @mainurl http://jevois.org
# @supporturl http://jevois.org
# @otherurl http://jevois.org
# @license GPL v3
# @distribution Unrestricted
# @restrictions None
# @ingroup modules
class PyClassificationDNN:
    # ####################################################################################################
    ## Constructor
    def __init__(self):
        self.confThreshold = 0.2 # Confidence threshold (0..1), higher for stricter confidence.
        self.inpWidth = 227      # Resized image width passed to network
        self.inpHeight = 227     # Resized image height passed to network
        self.scale = 1.0         # Value scaling factor applied to input pixels
        self.mean = [104, 117, 123] # Mean BGR value subtracted from input image
        self.rgb = True          # True if model expects RGB inputs, otherwise it expects BGR

        # Select one of the models:
        model = 'SqueezeNet'            # SqueezeNet v1.1, Caffe model

        # You should not have to edit anything beyond this point.
        backend = cv.dnn.DNN_BACKEND_DEFAULT
        target = cv.dnn.DNN_TARGET_CPU
        self.classes = None
        classnames = None
        if (model == 'SqueezeNet'):
            classnames = '/jevois/share/opencv-dnn/classification/synset_words.txt'
            modelname = '/jevois/share/opencv-dnn/classification/squeezenet_v1.1.caffemodel'
            configname = '/jevois/share/opencv-dnn/classification/squeezenet_v1.1.prototxt'

        # Load names of classes
        if classnames:
            with open(classnames, 'rt') as f:
                self.classes = f.read().rstrip('\n').split('\n')
        
        # Load a network
        self.net = cv.dnn.readNet(modelname, configname)
        self.net.setPreferableBackend(backend)
        self.net.setPreferableTarget(target)
        self.timer = jevois.Timer('Neural classification', 10, jevois.LOG_DEBUG)
        self.model = model
        
    # ####################################################################################################
    ## JeVois main processing function
    def process(self, inframe, outframe):
        frame = inframe.getCvBGR()
        self.timer.start()
        
        frameHeight = frame.shape[0]
        frameWidth = frame.shape[1]

        # Create a 4D blob from a frame.
        blob = cv.dnn.blobFromImage(frame, self.scale, (self.inpWidth, self.inpHeight), self.mean, self.rgb, crop=True)

        # Run a model
        self.net.setInput(blob)
        out = self.net.forward()

        # Get a class with a highest score:
        out = out.flatten()
        classId = np.argmax(out)
        confidence = out[classId]

        # Create dark-gray (value 80) image for the bottom panel, 24 pixels tall and show top-1 class:
        msgbox = np.zeros((24, frame.shape[1], 3), dtype = np.uint8) + 80
        rlabel = ' '
        if (confidence > self.confThreshold):
            rlabel = '%s: %.2f' % (self.classes[classId] if self.classes else 'Class #%d' % classId, confidence*100)

        cv.putText(msgbox, rlabel, (3, 15), cv.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv.LINE_AA)

        # Put efficiency information.
        cv.putText(frame, 'JeVois Classification DNN - ' + self.model, (3, 15),
                   cv.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv.LINE_AA)
        t, _ = self.net.getPerfProfile()
        fps = self.timer.stop()
        label = fps + ', %dms' % (t * 1000.0 / cv.getTickFrequency())
        cv.putText(frame, label, (3, frameHeight-5), cv.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv.LINE_AA)
        
        # Stack bottom panel below main image:
        frame = np.vstack((frame, msgbox))

        # Send output frame to host:
        outframe.sendCv(frame)
