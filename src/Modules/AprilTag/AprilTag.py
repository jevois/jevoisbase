import pyjevois
if pyjevois.pro: import libjevoispro as jevois
else: import libjevois as jevois

import cv2
import numpy as np
import apriltag

## Detect AprilTag robotic fiducial markers in Python
#
# This module detects AprilTag markers, which are small 2D barcode-like patterns used in many robotics applications.
# The code here is derived from https://pyimagesearch.com/2020/11/02/apriltag-with-python/
#
# On host, make sure the apriltag library is installed with 'pip3 install apriltag'; it is pre-installed on platform.
#
# If you need full 3D pose revovery, see our other module DemoArUco which also supports AprilTag.
#
# @author Laurent Itti
# 
# @displayname AprilTag
# @videomapping JVUI 0 0 30.0 CropScale=GREY@1024x576 YUYV 1920 1080 30.0 JeVois AprilTag
# @email itti\@usc.edu
# @address University of Southern California, HNB-07A, 3641 Watt Way, Los Angeles, CA 90089-2520, USA
# @copyright Copyright (C) 2024 by Laurent Itti, iLab and the University of Southern California
# @mainurl http://jevois.org
# @supporturl http://jevois.org/doc
# @otherurl http://iLab.usc.edu
# @license GPL v3
# @distribution Unrestricted
# @restrictions None
# @ingroup modules
class AprilTag:
    # ###################################################################################################
    ## Constructor
    def __init__(self):
        # Instantiate a JeVois Timer to measure our processing framerate:
        self.timer = jevois.Timer("apriltag", 100, jevois.LOG_INFO)
        self.families = [ 'tag36h11', 'tag36h10', 'tag36artoolkit', 'tag25h9', 'tag25h7', 'tag16h5' ]
        self.detector = None
        
    # ###################################################################################################
    ## JeVois optional extra init once the instance is fully constructed
    def init(self):
        # Create some parameters that users can adjust in the GUI:
        self.pc = jevois.ParameterCategory("AprilTag Parameters", "")

        self.dic = jevois.Parameter(self, 'dictionary', 'str', 'Dictionary to use. One of: ' +
                                    ', '.join(self.families), 'tag36h11', self.pc)
        self.dic.setCallback(self.setDic);
        
    # ###################################################################################################
    # Instantiate a detector each time dictionary is changed:
    def setDic(self, dicname):
        # Nuke any current detector instance:
        self.detector = None
        self.options = None
        
        # Check that family supported, see https://github.com/swatbotics/apriltag/blob/master/core/apriltag_family.c
        if dicname in self.families:
            self.options = apriltag.DetectorOptions(families = dicname)
            self.detector = apriltag.Detector(self.options)
        else:
            jevois.LFATAL('Unsupported AprilTag family. Must be one of: ' + ', '.join(self.families))

    # ###################################################################################################
    ## Process function with no USB output
    def processNoUSB(self, inframe):
        # Get the next camera image as grayscale and lower resolution for processing (may block until it is captured):
        ingray = inframe.getCvGRAYp()

        # Detect AprilTag markers:
        if self.detector:
            results = self.detector.detect(ingray)

            # Loop over the AprilTag detection results and send serial messages:
            for r in results:
                jevois.sendSerial('ATAG' + r.tag_id + ' ' + str(r.center[0]) + ' ' + str(r.center[1]))

    # ###################################################################################################
    ## Process function with GUI output on JeVois-Pro
    def processGUI(self, inframe, helper):
        # Start a new display frame, gets its size and also whether mouse/keyboard are idle:
        idle, winw, winh = helper.startFrame()

        # Draw full-resolution input frame from camera:
        x, y, w, h = helper.drawInputFrame("c", inframe, False, False)
        helper.itext('JeVois-Pro AprilTag detection')
        
        # Get the next camera image as grayscale and lower resolution for processing (may block until it is captured):
        ingray = inframe.getCvGRAYp()

        # Start measuring image processing time (NOTE: does not account for input conversion time):
        self.timer.start()

        # Detect AprilTag markers:
        if self.detector:
            results = self.detector.detect(ingray)
            col = 0xffffff7f # ARGB color for our drawings

            # Loop over the AprilTag detection results and draw them + send serial messages:
            for r in results:
                # Draw tag outline:
                (ptA, ptB, ptC, ptD) = r.corners.astype(float)
                helper.drawLine(ptA[0], ptA[1], ptB[0], ptB[1], col)
                helper.drawLine(ptB[0], ptB[1], ptC[0], ptC[1], col)
                helper.drawLine(ptC[0], ptC[1], ptD[0], ptD[1], col)
                helper.drawLine(ptD[0], ptD[1], ptA[0], ptA[1], col)

                # Draw a circle at the center of the AprilTag:
                helper.drawCircle(float(r.center[0]), float(r.center[1]), 5, col, True)

                # Show tag ID:
                helper.drawText(float(r.center[0] + 7), float(r.center[1] + 7), str(r.tag_id), col)

                # Send a serial message:
                jevois.sendSerial('ATAG' + r.tag_id + ' ' + str(r.center[0]) + ' ' + str(r.center[1]))
                
        # Write frames/s info from our timer:
        fps = self.timer.stop()
        helper.iinfo(inframe, fps, winw, winh);

        # End of frame:
        helper.endFrame()
