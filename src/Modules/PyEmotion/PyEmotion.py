import libjevois as jevois
import cv2 as cv
import numpy as np

## Human facial emotion recognition using OpenCV Deep Neural Networks (DNN)
#
# This module runs an emotion classification deep neural network using the OpenCV DNN library. The network is from the
# FER+ emotion recognition project, "Training Deep Networks for Facial Expression Recognition with Crowd-Sourced Label
# Distribution" arXiv:1608.01041
#
# The module outputs a score from -1000 to 1000 for each of: neutral, happiness, surprise, sadness, anger, disgust,
# fear, contempt.
#
# Note that this module does not include any face detection. Hence it always assumes that there is a face well centered
# in the image. You should enhance this module with first applying a face detector (see, e.g.,
# \jvmod{PyDetectionDNN}) and to only run the emotion recognition network on the detected faces.
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
        self.inpWidth = 64        # Resized image width passed to network
        self.inpHeight = 64       # Resized image height passed to network
        self.scale = 1.0          # Value scaling factor applied to input pixels
        self.mean = [127,127,127] # Mean BGR value subtracted from input image
        self.rgb = False          # True if model expects RGB inputs, otherwise it expects BGR

        # This network takes a while to load from microSD. To avoid timouts at construction,
        # we will load it in process() instead.
        
        self.timer = jevois.Timer('Neural emotion', 10, jevois.LOG_DEBUG)

    # ####################################################################################################
    ## JeVois main processing function
    def process(self, inframe, outframe):
        font = cv.FONT_HERSHEY_PLAIN
        siz = 0.8
        white = (255, 255, 255)
        
        # Load the network if needed:
        if not hasattr(self, 'net'):
            backend = cv.dnn.DNN_BACKEND_DEFAULT
            target = cv.dnn.DNN_TARGET_CPU
            self.classes = [ "neutral", "happiness", "surprise", "sadness", "anger", "disgust",
                             "fear", "contempt" ]
            self.model = 'FER+ ONNX'
            self.net = cv.dnn.readNet('/jevois/share/opencv-dnn/classification/emotion_ferplus.onnx', '')
            self.net.setPreferableBackend(cv.dnn.DNN_BACKEND_DEFAULT)
            self.net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)
                
        # Get the next frame from the camera sensor:
        frame = inframe.getCvBGR()
        self.timer.start()
        
        frameHeight = frame.shape[0]
        frameWidth = frame.shape[1]
        mid = int((frameWidth - 110) / 2) + 110 # x coord of midpoint of our bars
        leng = frameWidth - mid - 6             # max length of our bars
        maxconf = 999

        # Create a 4D blob from a frame.
        gframe = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        blob = cv.dnn.blobFromImage(gframe, self.scale, (self.inpWidth, self.inpHeight), self.mean, self.rgb, crop=True)

        # Run the model
        self.net.setInput(blob)
        out = self.net.forward()

        # Create dark-gray (value 80) image for the bottom panel, 96 pixels tall and show top-1 class:
        msgbox = np.zeros((96, frame.shape[1], 3), dtype = np.uint8) + 80

        # Show the scores for each class:
        out = out.flatten()
        for i in range(8):
            conf = out[i] * 100
            if conf > maxconf: conf = maxconf
            if conf < -maxconf: conf = -maxconf
            cv.putText(msgbox, self.classes[i] + ':', (3, 11*(i+1)), font, siz, white, 1, cv.LINE_AA)
            rlabel = '%+6.1f' % conf
            cv.putText(msgbox, rlabel, (76, 11*(i+1)), font, siz, white, 1, cv.LINE_AA)
            cv.line(msgbox, (mid, 11*i+6), (mid + int(conf*leng/maxconf), 11*i+6), white, 4)
        
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
