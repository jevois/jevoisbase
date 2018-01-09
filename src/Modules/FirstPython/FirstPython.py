import libjevois as jevois
import cv2
import numpy as np
import math # for cos, sin, etc

## Simple example of FIRST Robotics image processing pipeline using OpenCV in Python on JeVois
#
# This module is a simplified version of the C++ module \jvmod{FirstVision}. It is available with \jvversion{1.6.2} or
# later.
#
# This module implements a simple color-based object detector using OpenCV in Python. Its main goal is to also
# demonstrate full 6D pose recovery of the detected object, in Python.
#
# This module isolates pixels within a given HSV range (hue, saturation, and value of color pixels), does some cleanups,
# and extracts object contours. It is looking for a rectangular U shape of a specific size (set by parameters \p owm and
# \p ohm for object width and height in meters). See screenshots for an example of shape. It sends information about
# detected objects over serial.
#
# This module usually works best with the camera sensor set to manual exposure, manual gain, manual color balance, etc
# so that HSV color values are reliable. See the file \b script.cfg file in this module's directory for an example of
# how to set the camera settings each time this module is loaded.
#
# This module is provided for inspiration. It has no pretension of actually solving the FIRST Robotics vision problem
# in a complete and reliable way. It is released in the hope that FRC teams will try it out and get inspired to
# develop something much better for their own robot.
#
#  Using this module
#  -----------------
#
#  Check out [this tutorial](http://jevois.org/tutorials/UserFirstVision.html) first, for the \jvmod{FirstVision} module
#  writtent in C++ and also check out the doc for \jvmod{FirstVision}. Then you can just dive in and start editing the
#  python code of \jvmod{FirstPython}.
#
# See http://jevois.org/tutorials for tutorials on getting started with programming JeVois in Python without having
# to install any development software on your host computer.
#
# Trying it out
# -------------
#
# Edit the module's file at JEVOIS:/modules/JeVois/FirstPython/FirstPython.py and set the parameters \p self.owm and \p
# self.ohm to the physical width and height of your U-shaped object in meters. You should also review and edit the other
# parameters in the module's constructor, such as the range of HSV colors.
#
# @author Laurent Itti
# 
# @displayname FIRST Python
# @videomapping YUYV 640 252 60.0 YUYV 320 240 60.0 JeVois FirstPython
# @videomapping YUYV 320 252 60.0 YUYV 320 240 60.0 JeVois FirstPython
# @email itti\@usc.edu
# @address University of Southern California, HNB-07A, 3641 Watt Way, Los Angeles, CA 90089-2520, USA
# @copyright Copyright (C) 2018 by Laurent Itti, iLab and the University of Southern California
# @mainurl http://jevois.org
# @supporturl http://jevois.org/doc
# @otherurl http://iLab.usc.edu
# @license GPL v3
# @distribution Unrestricted
# @restrictions None
# @ingroup modules
class FirstPython:
    # ###################################################################################################
    ## Constructor
    def __init__(self):
        # HSV color range to use:
        #
        # H: 0=red/do not use because of wraparound, 30=yellow, 45=light green, 60=green, 75=green cyan, 90=cyan,
        #      105=light blue, 120=blue, 135=purple, 150=pink
        # S: 0 for unsaturated (whitish discolored object) to 255 for fully saturated (solid color)
        # V: 0 for dark to 255 for maximally bright
        self.HSVmin = np.array([ 20,  50, 180], dtype=np.uint8)
        self.HSVmax = np.array([ 80, 255, 255], dtype=np.uint8)

        # Measure your U-shaped object (in meters) and set its size here:
        self.owm = 0.280 # width in meters
        self.ohm = 0.175 # height in meters

        # Other processing parameters:
        self.epsilon = 0.015               # Shape smoothing factor (higher for smoother)
        self.hullarea = ( 20*20, 300*300 ) # Range of object area (in pixels) to track
        self.hullfill = 50                 # Max fill ratio of the convex hull (percent)
        self.ethresh = 900                 # Shape error threshold (lower is stricter for exact shape)
        self.margin = 5                    # Margin from from frame borders (pixels)
    
        # Instantiate a JeVois Timer to measure our processing framerate:
        self.timer = jevois.Timer("FirstPython", 100, jevois.LOG_INFO)

        # CAUTION: The constructor is a time-critical code section. Taking too long here could upset USB timings and/or
        # video capture software running on the host computer. Only init the strict minimum here, and do not use OpenCV,
        # read files, etc
        
    # ###################################################################################################
    ## Load camera calibration from JeVois share directory
    def loadCameraCalibration(self, w, h):
        cpf = "/jevois/share/camera/calibration{}x{}.yaml".format(w, h)
        fs = cv2.FileStorage(cpf, cv2.FILE_STORAGE_READ)
        if (fs.isOpened()):
            self.camMatrix = fs.getNode("camera_matrix").mat()
            self.distCoeffs = fs.getNode("distortion_coefficients").mat()
            jevois.LINFO("Loaded camera calibration from {}".format(cpf))
        else:
            jevois.LFATAL("Failed to read camera parameters from file [{}]".format(cpf))

    # ###################################################################################################
    ## Detect objects within our HSV range
    def detect(self, imgbgr, outimg = None):
        maxn = 5 # max number of objects we will consider
        h, w, chans = imgbgr.shape

        # Convert input image to HSV:
        imghsv = cv2.cvtColor(imgbgr, cv2.COLOR_BGR2HSV)

        # Isolate pixels inside our desired HSV range:
        imgth = cv2.inRange(imghsv, self.HSVmin, self.HSVmax)
        str = "H={}-{} S={}-{} V={}-{} ".format(self.HSVmin[0], self.HSVmax[0], self.HSVmin[1],
                                                self.HSVmax[1], self.HSVmin[2], self.HSVmax[2])

        # Create structuring elements for morpho maths:
        if not hasattr(self, 'erodeElement'):
            self.erodeElement = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
            self.dilateElement = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        
        # Apply morphological operations to cleanup the image noise:
        imgth = cv2.erode(imgth, self.erodeElement)
        imgth = cv2.dilate(imgth, self.dilateElement)

        # Detect objects by finding contours:
        im2, contours, hierarchy = cv2.findContours(imgth, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        str += "N={} ".format(len(contours))

        # Only consider the 5 biggest objects by area:
        contours = sorted(contours, key = cv2.contourArea, reverse = True)[:maxn]
        hlist = [ ] # list of hulls of good objects, which we will return
        str2 = ""
        beststr2 = ""
        
        # Identify the "good" objects:
        for c in contours:
            # Keep track of our best detection so far:
            if len(str2) > len(beststr2): beststr2 = str2
            str2 = ""

            # Compute contour area:
            area = cv2.contourArea(c, oriented = False)

            # Compute convex hull:
            rawhull = cv2.convexHull(c, clockwise = True)
            rawhullperi = cv2.arcLength(rawhull, closed = True)
            hull = cv2.approxPolyDP(rawhull, epsilon = self.epsilon * rawhullperi * 3.0, closed = True)

            # Is it the right shape?
            if (hull.shape != (4,1,2)): continue # 4 vertices for the rectangular convex outline (shows as a trapezoid)
            str2 += "H" # Hull is quadrilateral
          
            huarea = cv2.contourArea(hull, oriented = False)
            if huarea < self.hullarea[0] or huarea > self.hullarea[1]: continue
            str2 += "A" # Hull area ok
          
            hufill = area / huarea * 100.0
            if hufill > self.hullfill: continue
            str2 += "F" # Fill is ok
          
            # Check object shape:
            peri = cv2.arcLength(c, closed = True)
            approx = cv2.approxPolyDP(c, epsilon = self.epsilon * peri, closed = True)
            if len(approx) < 7 or len(approx) > 9: continue  # 8 vertices for a U shape
            str2 += "S" # Shape is ok

            # Compute contour serr:
            serr = 100.0 * cv2.matchShapes(c, approx, cv2.CONTOURS_MATCH_I1, 0.0)
            if serr > self.ethresh: continue
            str2 += "E" # Shape error is ok
          
            # Reject the shape if any of its vertices gets within the margin of the image bounds. This is to avoid
            # getting grossly incorrect 6D pose estimates as the shape starts getting truncated as it partially exits
            # the camera field of view:
            reject = 0
            for v in c:
                if v[0,0] < self.margin or v[0,0] >= w-self.margin or v[0,1] < self.margin or v[0,1] >= h-self.margin:
                   reject = 1
                   break
               
            if reject == 1: continue
            str2 += "M" # Margin ok
          
            # Re-order the 4 points in the hull if needed: In the pose estimation code, we will assume vertices ordered
            # as follows:
            #
            #    0|        |3
            #     |        |
            #     |        |
            #    1----------2

            # v10+v23 should be pointing outward the U more than v03+v12 is:
            v10p23 = complex(hull[0][0,0] - hull[1][0,0] + hull[3][0,0] - hull[2][0,0],
                             hull[0][0,1] - hull[1][0,1] + hull[3][0,1] - hull[2][0,1])
            len10p23 = abs(v10p23)
            v03p12 = complex(hull[3][0,0] - hull[0][0,0] + hull[2][0,0] - hull[1][0,0],
                             hull[3][0,1] - hull[0][0,1] + hull[2][0,1] - hull[1][0,1])
            len03p12 = abs(v03p12)

            # Vector from centroid of U shape to centroid of its hull should also point outward of the U:
            momC = cv2.moments(c)
            momH = cv2.moments(hull)
            vCH = complex(momH['m10'] / momH['m00'] - momC['m10'] / momC['m00'],
                          momH['m01'] / momH['m00'] - momC['m01'] / momC['m00'])
            lenCH = abs(vCH)

            if len10p23 < 0.1 or len03p12 < 0.1 or lenCH < 0.1: continue
            str2 += "V" # Shape vectors ok

            good = (v10p23.real * vCH.real + v10p23.imag * vCH.imag) / (len10p23 * lenCH)
            bad = (v03p12.real * vCH.real + v03p12.imag * vCH.imag) / (len03p12 * lenCH)

            # We reject upside-down detections as those are likely to be spurious:
            if vCH.imag >= -2.0: continue
            str2 += "U" # U shape is upright
        
            # Fixup the ordering of the vertices if needed:
            if bad > good: hull = np.roll(hull, shift = 1, axis = 0)
                
            # This detection is a keeper:
            str2 += " OK"
            hlist.append(hull)

        if len(str2) > len(beststr2):  beststr2 = str2
        
        # Display any results requested by the users:
        if outimg is not None and outimg.valid():
            if (outimg.width == w * 2): jevois.pasteGreyToYUYV(imgth, outimg, w, 0)
            jevois.writeText(outimg, str + beststr2, 3, h+1, jevois.YUYV.White, jevois.Font.Font6x10)

        return hlist

    # ###################################################################################################
    ## Estimate 6D pose of each of the quadrilateral objects in hlist:
    def estimatePose(self, hlist):
        rvecs = []
        tvecs = []
        
        # set coordinate system in the middle of the object, with Z pointing out
        objPoints = np.array([ ( -self.owm * 0.5, -self.ohm * 0.5, 0 ),
                               ( -self.owm * 0.5,  self.ohm * 0.5, 0 ),
                               (  self.owm * 0.5,  self.ohm * 0.5, 0 ),
                               (  self.owm * 0.5, -self.ohm * 0.5, 0 ) ])

        for detection in hlist:
            det = np.array(detection, dtype=np.float).reshape(4,2,1)
            (ok, rv, tv) = cv2.solvePnP(objPoints, det, self.camMatrix, self.distCoeffs)
            if ok:
                rvecs.append(rv)
                tvecs.append(tv)
            else:
                rvecs.append(np.array([ (0.0), (0.0), (0.0) ]))
                tvecs.append(np.array([ (0.0), (0.0), (0.0) ]))

        return (rvecs, tvecs)        
        
    # ###################################################################################################
    ## Send serial messages, one per object
    def sendAllSerial(self, w, h, hlist, rvecs, tvecs):
        idx = 0
        for c in hlist:
            # Compute quaternion: FIXME need to check!
            tv = tvecs[idx]
            axis = rvecs[idx]
            angle = (axis[0] * axis[0] + axis[1] * axis[1] + axis[2] * axis[2]) ** 0.5

            # This code lifted from pyquaternion from_axis_angle:
            mag_sq = axis[0] * axis[0] + axis[1] * axis[1] + axis[2] * axis[2]
            if (abs(1.0 - mag_sq) > 1e-12): axis = axis / (mag_sq ** 0.5)
            theta = angle / 2.0
            r = math.cos(theta)
            i = axis * math.sin(theta)
            q = (r, i[0], i[1], i[2])

            jevois.sendSerial("D3 {} {} {} {} {} {} {} {} {} {} FIRST".
                              format(np.asscalar(tv[0]), np.asscalar(tv[1]), np.asscalar(tv[2]),  # position
                                     self.owm, self.ohm, 1.0,                                     # size
                                     r, np.asscalar(i[0]), np.asscalar(i[1]), np.asscalar(i[2]))) # pose
            idx += 1
                              
    # ###################################################################################################
    ## Draw all detected objects in 3D
    def drawDetections(self, outimg, hlist, rvecs = None, tvecs = None):
        # Show trihedron and parallelepiped centered on object:
        hw = self.owm * 0.5
        hh = self.ohm * 0.5
        dd = -0.5 * max(self.owm, self.ohm)
        i = 0
        empty = np.array([ (0.0), (0.0), (0.0) ])
            
        for obj in hlist:
            # skip those for which solvePnP failed:
            if np.array_equal(rvecs[i], empty):
                i += 1
                continue
            
            # Project axis points:
            axisPoints = np.array([ (0.0, 0.0, 0.0), (hw, 0.0, 0.0), (0.0, hh, 0.0), (0.0, 0.0, dd) ])
            imagePoints, jac = cv2.projectPoints(axisPoints, rvecs[i], tvecs[i], self.camMatrix, self.distCoeffs)
            
            # Draw axis lines:
            jevois.drawLine(outimg, int(imagePoints[0][0,0] + 0.5), int(imagePoints[0][0,1] + 0.5),
                            int(imagePoints[1][0,0] + 0.5), int(imagePoints[1][0,1] + 0.5),
                            2, jevois.YUYV.MedPurple)
            jevois.drawLine(outimg, int(imagePoints[0][0,0] + 0.5), int(imagePoints[0][0,1] + 0.5),
                            int(imagePoints[2][0,0] + 0.5), int(imagePoints[2][0,1] + 0.5),
                            2, jevois.YUYV.MedGreen)
            jevois.drawLine(outimg, int(imagePoints[0][0,0] + 0.5), int(imagePoints[0][0,1] + 0.5),
                            int(imagePoints[3][0,0] + 0.5), int(imagePoints[3][0,1] + 0.5),
                            2, jevois.YUYV.MedGrey)
          
            # Also draw a parallelepiped:
            cubePoints = np.array([ (-hw, -hh, 0.0), (hw, -hh, 0.0), (hw, hh, 0.0), (-hw, hh, 0.0),
                                    (-hw, -hh, dd), (hw, -hh, dd), (hw, hh, dd), (-hw, hh, dd) ])
            cu, jac2 = cv2.projectPoints(cubePoints, rvecs[i], tvecs[i], self.camMatrix, self.distCoeffs)

            # Round all the coordinates and cast to int for drawing:
            cu = np.rint(cu)
          
            # Draw parallelepiped lines:
            jevois.drawLine(outimg, int(cu[0][0,0]), int(cu[0][0,1]), int(cu[1][0,0]), int(cu[1][0,1]),
                            1, jevois.YUYV.LightGreen)
            jevois.drawLine(outimg, int(cu[1][0,0]), int(cu[1][0,1]), int(cu[2][0,0]), int(cu[2][0,1]),
                            1, jevois.YUYV.LightGreen)
            jevois.drawLine(outimg, int(cu[2][0,0]), int(cu[2][0,1]), int(cu[3][0,0]), int(cu[3][0,1]),
                            1, jevois.YUYV.LightGreen)
            jevois.drawLine(outimg, int(cu[3][0,0]), int(cu[3][0,1]), int(cu[0][0,0]), int(cu[0][0,1]),
                            1, jevois.YUYV.LightGreen)
            jevois.drawLine(outimg, int(cu[4][0,0]), int(cu[4][0,1]), int(cu[5][0,0]), int(cu[5][0,1]),
                            1, jevois.YUYV.LightGreen)
            jevois.drawLine(outimg, int(cu[5][0,0]), int(cu[5][0,1]), int(cu[6][0,0]), int(cu[6][0,1]),
                            1, jevois.YUYV.LightGreen)
            jevois.drawLine(outimg, int(cu[6][0,0]), int(cu[6][0,1]), int(cu[7][0,0]), int(cu[7][0,1]),
                            1, jevois.YUYV.LightGreen)
            jevois.drawLine(outimg, int(cu[7][0,0]), int(cu[7][0,1]), int(cu[4][0,0]), int(cu[4][0,1]),
                            1, jevois.YUYV.LightGreen)
            jevois.drawLine(outimg, int(cu[0][0,0]), int(cu[0][0,1]), int(cu[4][0,0]), int(cu[4][0,1]),
                            1, jevois.YUYV.LightGreen)
            jevois.drawLine(outimg, int(cu[1][0,0]), int(cu[1][0,1]), int(cu[5][0,0]), int(cu[5][0,1]),
                            1, jevois.YUYV.LightGreen)
            jevois.drawLine(outimg, int(cu[2][0,0]), int(cu[2][0,1]), int(cu[6][0,0]), int(cu[6][0,1]),
                            1, jevois.YUYV.LightGreen)
            jevois.drawLine(outimg, int(cu[3][0,0]), int(cu[3][0,1]), int(cu[7][0,0]), int(cu[7][0,1]),
                            1, jevois.YUYV.LightGreen)

            i += 1
            
    # ###################################################################################################
    ## Process function with no USB output
    def processNoUSB(self, inframe):
        # Get the next camera image (may block until it is captured) as OpenCV BGR:
        imgbgr = inframe.getCvBGR()
        h, w, chans = imgbgr.shape
        
        # Start measuring image processing time:
        self.timer.start()

        # Get a list of quadrilateral convex hulls for all good objects:
        hlist = self.detect(imgbgr)

        # Load camera calibration if needed:
        if not hasattr(self, 'camMatrix'): self.loadCameraCalibration(w, h)

        # Map to 6D (inverse perspective):
        (rvecs, tvecs) = self.estimatePose(hlist)

        # Send all serial messages:
        self.sendAllSerial(w, h, hlist, rvecs, tvecs)

        # Log frames/s info (will go to serlog serial port, default is None):
        self.timer.stop()

    # ###################################################################################################
    ## Process function with USB output
    def process(self, inframe, outframe):
        # Get the next camera image (may block until it is captured). To avoid wasting much time assembling a composite
        # output image with multiple panels by concatenating numpy arrays, in this module we use raw YUYV images and
        # fast paste and draw operations provided by JeVois on those images:
        inimg = inframe.get()

        # Start measuring image processing time:
        self.timer.start()
        
        # Convert input image to BGR24:
        imgbgr = jevois.convertToCvBGR(inimg)
        h, w, chans = imgbgr.shape

        # Get pre-allocated but blank output image which we will send over USB:
        outimg = outframe.get()
        outimg.require("output", w * 2, h + 12, jevois.V4L2_PIX_FMT_YUYV)
        jevois.paste(inimg, outimg, 0, 0)
        jevois.drawFilledRect(outimg, 0, h, outimg.width, outimg.height-h, jevois.YUYV.Black)
        
        # Let camera know we are done using the input image:
        inframe.done()
        
        # Get a list of quadrilateral convex hulls for all good objects:
        hlist = self.detect(imgbgr, outimg)

        # Load camera calibration if needed:
        if not hasattr(self, 'camMatrix'): self.loadCameraCalibration(w, h)

        # Map to 6D (inverse perspective):
        (rvecs, tvecs) = self.estimatePose(hlist)

        # Send all serial messages:
        self.sendAllSerial(w, h, hlist, rvecs, tvecs)

        # Draw all detections in 3D:
        self.drawDetections(outimg, hlist, rvecs, tvecs)

        # Write frames/s info from our timer into the edge map (NOTE: does not account for output conversion time):
        fps = self.timer.stop()
        jevois.writeText(outimg, fps, 3, h-10, jevois.YUYV.White, jevois.Font.Font6x10)
    
        # We are done with the output, ready to send it to host over USB:
        outframe.send()

        
######################################################################################################################
#
# JeVois Smart Embedded Machine Vision Toolkit - Copyright (C) 2017 by Laurent Itti, the University of Southern
# California (USC), and iLab at USC. See http://iLab.usc.edu and http://jevois.org for information about this project.
#
# This file is part of the JeVois Smart Embedded Machine Vision Toolkit.  This program is free software; you can
# redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software
# Foundation, version 2.  This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY;
# without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public
# License for more details.  You should have received a copy of the GNU General Public License along with this program;
# if not, write to the Free Software Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301, USA.
#
# Contact information: Laurent Itti - 3641 Watt Way, HNB-07A - Los Angeles, CA 90089-2520 - USA.
# Tel: +1 213 740 3527 - itti@pollux.usc.edu - http://iLab.usc.edu - http://jevois.org
######################################################################################################################
        
