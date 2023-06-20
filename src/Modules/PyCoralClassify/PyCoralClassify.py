import pyjevois
if pyjevois.pro: import libjevoispro as jevois
else: import libjevois as jevois
import cv2 as cv
import numpy as np
from PIL import Image
from pycoral.utils import edgetpu
from pycoral.adapters import classify
from pycoral.adapters import common
from pycoral.utils.dataset import read_label_file
import time

## Object recognition using Coral Edge TPU
#
# This module runs an object classification deep neural network using the Coral TPU library. It only works on JeVois-Pro
# platform equipped with an Edge TPU add-on card. Classification (recognition) networks analyze a central portion of the
# whole scene and produce identity labels and confidence scores about what the object in the field of view might be.
#
# This module supports networks implemented in TensorFlow-Lite and ported to Edge TPU/
#
# Included with the standard JeVois distribution are:
#
# - MobileNetV3
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
# This module is adapted from the sample code:
# https://github.com/google-coral/pycoral/blob/master/examples/classify_image.py
#
# More pre-trained models are available at https://coral.ai/models/
#
#
# @author Laurent Itti
# 
# @videomapping YUYV 320 264 30.0 YUYV 320 240 30.0 JeVois PyClassificationDNN
# @email itti@usc.edu
# @address 880 W 1st St Suite 807, Los Angeles CA 90012, USA
# @copyright Copyright (C) 2020 by Laurent Itti
# @mainurl http://jevois.org
# @supporturl http://jevois.org
# @otherurl http://jevois.org
# @license GPL v3
# @distribution Unrestricted
# @restrictions None
# @ingroup modules
class PyCoralClassify:
    # ####################################################################################################
    ## Constructor
    def __init__(self):
        if jevois.getNumInstalledTPUs() == 0:
            jevois.LFATAL("A Google Coral EdgeTPU is required for this module (PCIe M.2 2230 A+E or USB)")

        self.threshold = 0.2 # Confidence threshold (0..1), higher for stricter confidence.
        self.rgb = True      # True if model expects RGB inputs, otherwise it expects BGR
        
        # Select one of the models:
        self.model = 'MobileNetV3'

        # You should not have to edit anything beyond this point.
        if (self.model == 'MobileNetV3'):
            classnames = 'imagenet_labels.txt'
            modelname = 'tf2_mobilenet_v3_edgetpu_1.0_224_ptq_edgetpu.tflite'

        # Load names of classes:
        sdir = pyjevois.share + '/coral/classification/'
        self.labels = read_label_file(sdir + classnames)
        
        # Load network:
        self.interpreter = edgetpu.make_interpreter(sdir + modelname)
        #self.interpreter = edgetpu.make_interpreter(*modelname.split('@'))
        self.interpreter.allocate_tensors()
        self.timer = jevois.Timer('Coral classification', 10, jevois.LOG_DEBUG)
        
    # ####################################################################################################
    ## JeVois main processing function
    def process(self, inframe, outframe):
        frame = inframe.getCvRGB() if self.rgb else inframe.getCvBGR()
        self.timer.start()
        
        h = frame.shape[0]
        w = frame.shape[1]

        # Set the input:
        size = common.input_size(self.interpreter)
        image = Image.fromarray(frame).resize(size, Image.ANTIALIAS)
        common.set_input(self.interpreter, image)
  
        # Run the model
        start = time.perf_counter()
        self.interpreter.invoke()
        inference_time = time.perf_counter() - start
        
        # Get classes with high enough scores:
        classes = classify.get_classes(self.interpreter, 1, self.threshold)

        # Create dark-gray (value 80) image for the bottom panel, 24 pixels tall and show top-1 class:
        msgbox = np.zeros((24, w, 3), dtype = np.uint8) + 80
        for c in classes:
            rlabel = '%s: %.2f' % (self.labels.get(c.id, c.id), c.score)
            cv.putText(msgbox, rlabel, (3, 15), cv.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv.LINE_AA)

        # Put efficiency information:
        cv.putText(frame, 'JeVois Coral Classification - ' + self.model, (3, 15),
                   cv.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv.LINE_AA)

        fps = self.timer.stop()
        label = fps + ', %dms' % (inference_time * 1000.0)
        cv.putText(frame, label, (3, h-5), cv.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv.LINE_AA)
        
        # Stack bottom panel below main image:
        frame = np.vstack((frame, msgbox))

        # Send output frame to host:
        if self.rgb: outframe.sendCvRGB(frame)
        else: outframe.sendCv(frame)

    # ###################################################################################################
    ## Process function with GUI output
    def processGUI(self, inframe, helper):
        # Start a new display frame, gets its size and also whether mouse/keyboard are idle:
        idle, winw, winh = helper.startFrame()

        # Draw full-resolution input frame from camera:
        x, y, w, h = helper.drawInputFrame("c", inframe, False, False)
        
        # Get the next camera image at processing resolution (may block until it is captured):
        frame = inframe.getCvRGBp() if self.rgb else inframe.getCvBGRp()
        
        # Start measuring image processing time:
        self.timer.start()
        
        # Set the input:
        size = common.input_size(self.interpreter)
        image = Image.fromarray(frame).resize(size, Image.ANTIALIAS)
        common.set_input(self.interpreter, image)
  
        # Run the model
        start = time.perf_counter()
        self.interpreter.invoke()
        inference_time = time.perf_counter() - start
        
        # Get classes with high enough scores:
        classes = classify.get_classes(self.interpreter, 1, self.threshold)

        # Put efficiency information:
        helper.itext('JeVois-Pro Python Coral Classification - %s - %dms/inference' %
                     (self.model, inference_time * 1000.0), 0, -1)

        # Report top-scoring classes:
        for c in classes:
            rlabel = '%s: %.2f' % (self.labels.get(c.id, c.id), c.score)
            helper.itext(rlabel, 0, -1)

        # Write frames/s info from our timer:
        fps = self.timer.stop()
        helper.iinfo(inframe, fps, winw, winh);

        # End of frame:
        helper.endFrame()
