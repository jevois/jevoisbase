import pyjevois
if pyjevois.pro: import libjevoispro as jevois
else: import libjevois as jevois
import cv2
import numpy as np
import mediapipe as mp

## Selfie segmentation using MediaPipe
#
# Segment out face and upper body and show them on top of an image background, using MediaPipe in Python
#
# This code is derived from sample_selfie_segmentation.py at https://github.com/Kazuhito00/mediapipe-python-sample
#
# @author Laurent Itti
# 
# @videomapping JVUI 0 0 30.0 CropScale=RGB24@512x288:YUYV 1920 1080 30.0 JeVois PySelfie
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
class PySelfie:
    # ###################################################################################################
    ## Constructor
    def __init__(self):
        # Parameters:
        model_selection = 1 # which model to use internally: 0: 256x256, 1: 256x144
        alpha = 150         # Transparency alpha values for processGUI, higher is less transparent
        tidx = 1            # Class index of transparent foreground

        # Instantiate a JeVois Timer to measure our processing framerate:
        self.timer = jevois.Timer("selfie", 30, jevois.LOG_DEBUG)

        # Instantiate mediapipe model:
        self.mp_selfie_segmentation = mp.solutions.selfie_segmentation
        self.selfie_segmentation = self.mp_selfie_segmentation.SelfieSegmentation(model_selection = model_selection)

        # Create a colormap for mask:
        self.cmapRGBA = self.create_pascal_label_colormapRGBA(alpha, tidx)

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
        
        # Run graph:
        results = self.selfie_segmentation.process(image)
        
        # Draw the mask on top of our image, OpenGL will do the alpha blending:
        mask = self.cmapRGBA[results.segmentation_mask.astype(np.uint8)]
        helper.drawImage("m", mask, True, False, True)

        # Write frames/s info from our timer:
        fps = self.timer.stop()
        helper.iinfo(inframe, fps, winw, winh);

        # End of frame:
        helper.endFrame()

    # ####################################################################################################
    def create_pascal_label_colormapRGBA(self, alpha, tidx):
        """Creates a label colormap used in PASCAL VOC segmentation benchmark.
        Returns:
        A Colormap for visualizing segmentation results.
        """
        colormap = np.zeros((256, 4), dtype=int)
        indices = np.arange(256, dtype=int)

        for shift in reversed(range(8)):
            for channel in range(3):
                colormap[:, channel] |= ((indices >> channel) & 1) << shift
                indices >>= 3
                
        colormap[:, 3] = alpha
        colormap[tidx, 3] = 0 # force fully transparent for entry tidx
        return colormap.astype(np.uint8)
  
