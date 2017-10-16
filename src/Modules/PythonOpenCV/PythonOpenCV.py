import libjevois as jevois
import cv2
import numpy as np

## Simple example of image processing using OpenCV in Python on JeVois
#
# This module by default simply converts the input image to a grayscale OpenCV image, and then applies the Canny
# edge detection algorithm. Try to edit it to do something else (note that the videomapping associated with this
# module has grayscale image outputs, so that is what you should output).
#
# See http://jevois.org/tutorials for tutorials on getting started with programming JeVois in Python without having
# to install any development software on your host computer.
#
# @author Laurent Itti
# 
# @displayname Python OpenCV
# @videomapping GRAY 640 480 20.0 YUYV 640 480 20.0 JeVois PythonOpenCV
# @email itti\@usc.edu
# @address University of Southern California, HNB-07A, 3641 Watt Way, Los Angeles, CA 90089-2520, USA
# @copyright Copyright (C) 2017 by Laurent Itti, iLab and the University of Southern California
# @mainurl http://jevois.org
# @supporturl http://jevois.org/doc
# @otherurl http://iLab.usc.edu
# @license GPL v3
# @distribution Unrestricted
# @restrictions None
# @ingroup modules
class PythonOpenCV:
    # ###################################################################################################
    ## Constructor
    def __init__(self):
        # Instantiate a JeVois Timer to measure our processing framerate:
        self.timer = jevois.Timer("canny", 100, jevois.LOG_INFO)
        
    # ###################################################################################################
    ## Process function with no USB output
    #def process(self, inframe):
    #    jevois.LFATAL("process with no USB output not implemented yet in this module")

    # ###################################################################################################
    ## Process function with USB output
    def process(self, inframe, outframe):
        # Get the next camera image (may block until it is captured) and convert it to OpenCV GRAY:
        inimggray = inframe.getCvGRAY()
        
        # Start measuring image processing time (NOTE: does not account for input conversion time):
        self.timer.start()
        
        # Detect edges using the Canny algorithm from OpenCV:
        edges = cv2.Canny(inimggray, 100, 200, apertureSize = 3)
        
        # Write frames/s info from our timer into the edge map (NOTE: does not account for output conversion time):
        fps = self.timer.stop()
        height, width = edges.shape
        cv2.putText(edges, fps, (3, height - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 255, 1, cv2.LINE_AA)
        
        # Convert our GRAY output image to video output format and send to host over USB:
        outframe.sendCvGRAY(edges)
    
    # ###################################################################################################
    ## Parse a serial command forwarded to us by the JeVois Engine, return a string
    #def parseSerial(self, str):
    #    return "ERR: Unsupported command"
    
    # ###################################################################################################
    ## Return a string that describes the custom commands we support, for the JeVois help message
    #def supportedCommands(self):
    #    return ""
