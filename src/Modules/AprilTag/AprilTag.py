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
        
        self.calib = jevois.Parameter(self, 'calibrate', 'bool',
                                      'Calibration mode (you need a calibration chessboard)', False, self.pc)
        self.calib.setCallback(self.calibChanged)

        self.crows = jevois.Parameter(self, 'calibration_rows', 'uint',
                                          'Number of chessboard inner corners in the vertical direction', 6, self.pc)

        self.ccols = jevois.Parameter(self, 'calibration_cols', 'uint',
                                          'Number of chessboard inner corners in the horizontal direction', 8, self.pc)

        self.csiz = jevois.Parameter(self, 'calibration_checksize', 'float',
                                        'Check size in user-chosen units', 50.0, self.pc)
        self.ipoints = []
        self.calibration_msg = None
        self.iter = 0 # to get a variety of viewpoints, we will skip some frames

    # ###################################################################################################
    # Instantiate a detector each time dictionary is changed:
    def setDic(self, dicname):
        # Check that family supported, see https://github.com/swatbotics/apriltag/blob/master/core/apriltag_family.c
        if dicname in self.families:
            self.options = apriltag.DetectorOptions(families = dicname)
            self.detector = apriltag.Detector(self.options)
        else:
            jevois.LFATAL('Unsupported AprilTag family. Must be one of: ' + ', '.join(self.families))

    # ###################################################################################################
    # Reset calibration data each time some calibration param is changed
    def calibChanged(self, val):
        if val:
            self.crows.freeze(True)
            self.ccols.freeze(True)
            self.csiz.freeze(True)
        else:
            self.crows.freeze(False)
            self.ccols.freeze(False)
            self.csiz.freeze(False)
            
        self.ipoints = []
        self.calibration_msg = None
        self.iter = 0 # to get a variety of viewpoints, we will skip some frames
        
    # ###################################################################################################
    ## Process function with no USB output
    #def processNoUSB(self, inframe):
    #    jevois.LFATAL("process with no USB output not implemented yet in this module")

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

        # Run calibration or detection?
        if self.calib.get():
            self.runCalibration(ingray, helper)
            
        # Detect AprilTag markers:
        elif self.detector:
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
    
    # ###################################################################################################
    # Run chessboard calibration
    # Code from https://github.com/swatbotics/apriltag/blob/master/python/calibrate_camera.py
    # Here, we will grab 20 good checkerboards, giving a few seconds in between to move it, then run the calib.
    def runCalibration(self, gray, helper):
        # If calibration was complete, just show results:
        if self.calibration_msg is not None:
            helper.itext(self.calibration_msg)
            return
        
        col = 0xffff7f7f # ARGB color for our drawings

        if self.crows.get() < self.ccols.get(): patternsize = (int(self.ccols.get()), int(self.crows.get()))
        else: patternsize = (int(self.crows.get()), int(self.ccols.get()))
        sz = self.csiz.get()

        x = np.arange(patternsize[0])*sz
        y = np.arange(patternsize[1])*sz

        xgrid, ygrid = np.meshgrid(x, y)
        zgrid = np.zeros_like(xgrid)
        opoints = np.dstack((xgrid, ygrid, zgrid)).reshape((-1, 1, 3)).astype(np.float32)
        imagesize = (gray.shape[1], gray.shape[0])

        helper.itext('Found {} / 20 chessboards so far...'.format(len(self.ipoints)))
        
        self.iter += 1
        if self.iter >= 100:
            self.iter = 0
            helper.itext('Detecting corners...')
            retval, corners = cv2.findChessboardCorners(gray, patternsize)

            if not retval:
                helper.itext('No chessboard found...')
                return

            for c in corners:
                helper.drawCircle(float(c[0][0]), float(c[0][1]), 5, col, True)

            self.ipoints.append(corners)
            
        helper.itext('Next snapshot in {}...'.format(100-self.iter))

        if len(self.ipoints) < 20: return

        flags = (cv2.CALIB_ZERO_TANGENT_DIST | cv2.CALIB_FIX_K1 | cv2.CALIB_FIX_K2 |
                 cv2.CALIB_FIX_K3 | cv2.CALIB_FIX_K4 | cv2.CALIB_FIX_K5 | cv2.CALIB_FIX_K6)

        opoints = [opoints] * len(self.ipoints)

        retval, K, dcoeffs, rvecs, tvecs = cv2.calibrateCamera(opoints, self.ipoints, imagesize, cameraMatrix=None,
                                                               distCoeffs = np.zeros(5), flags = flags)
        #if np.all(dcoeffs == 0):
    
        fx = K[0,0]
        fy = K[1,1]
        cx = K[0,2]
        cy = K[1,2]
        params = (fx, fy, cx, cy)
        self.calibration_msg = 'Calibrated fx, fy, cx, cy = {}'.format(repr(params))
