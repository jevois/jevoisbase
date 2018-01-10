import libjevois as jevois
import cv2
import numpy as np
import math # for cos, sin, etc

## Simple example of object detection using ORB keypoints followed by 6D pose estimation in Python
#
# This module implements an object detector using ORB keypoints using OpenCV in Python. Its main goal is to also
# demonstrate full 6D pose recovery of the detected object, in Python. See \jvmod{ObjectDetect} for more info about
# object detection using keypoints. This module is available with \jvversion{1.6.3} and later.
#
# The algorithm consists of 5 phases:
# - detect keypoint locations, typically corners or other distinctive texture elements or markings;
# - compute keypoint descriptors, which are summary representations of the image neighborhood around each keypoint;
# - match descriptors from current image to descriptors previously extracted from training images;
# - if enough matches are found between the current image and a given training image, and they are of good enough
#   quality, compute the homography (geometric transformation) between keypoint locations in that training image and
#   locations of the matching keypoints in the current image. If it is well conditioned (i.e., a 3D viewpoint change
#   could well explain how the keypoints moved between the training and current images), declare that a match was
#   found, and draw a green rectangle around the detected object.
# - 6D pose estimation (3D translation + 3D rotation) given a known physical size for the object.
#
# For more information about ORB in OpenCV, see, e.g., https://docs.opencv.org/3.4.0/df/dd2/tutorial_py_surf_intro.html
#
# This module is provided for inspiration. It has no pretension of actually solving the FIRST Robotics vision problem
# in a complete and reliable way. It is released in the hope that FRC teams will try it out and get inspired to
# develop something much better for their own robot.
#
# Using this module
# -----------------
#
# Simply add an image of the object you want to detect as
# <b>JEVOIS:/modules/JeVois/PythonObject6D/images/reference.png</b> on your JeVois microSD card. It will be processed
# when the module starts. The name of recognized object returned by this module are simply the file name of the picture
# you have added in that directory (reference.png for now). No additional training procedure is needed. Then edit the
# values of \p self.owm and \p self.ohm to the width ahd height, in meters, of the actual physical object in your
# picture.
#
# \todo Add support for multiple images and online training as in \jvmod{ObjectDetect}
#
# @author Laurent Itti
# 
# @displayname FIRST Python
# @videomapping YUYV 320 262 15.0 YUYV 320 240 15.0 JeVois PythonObject6D
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
class PythonObject6D:
    # ###################################################################################################
    ## Constructor
    def __init__(self):
        # Full file name of the training image:
        self.fname = "/jevois/modules/JeVois/PythonObject6D/images/reference.png"
        
        # Measure your object (in meters) and set its size here:
        self.owm = 0.187 # width in meters
        self.ohm = 0.082 # height in meters

        # Other parameters:
        self.distth = 40.0 # Descriptor distance threshold (lower is stricter for exact matches)
        
        # Instantiate a JeVois Timer to measure our processing framerate:
        self.timer = jevois.Timer("PythonObject6D", 100, jevois.LOG_INFO)
        
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
    ## Detect objects using keypoints
    def detect(self, imggray, outimg = None):
        h, w = imggray.shape
        hlist = []
        
        # Create a keypoint detector if needed:
        if not hasattr(self, 'detector'):
            self.detector = cv2.ORB_create()

        # Load training image and detect keypoints on it if needed:
        if not hasattr(self, 'refkp'):
            refimg = cv2.imread(self.fname, 0)
            self.refkp, self.refdes = self.detector.detectAndCompute(refimg, None)

            # Also store corners of reference image for homography mapping:
            refh, refw = refimg.shape
            self.refcorners = np.float32([ [0.0, 0.0], [0.0, refh], [refw, refh], [refw, 0.0] ]).reshape(-1,1,2)
            jevois.LINFO("Extracted {} keypoints and descriptors from {}".format(len(self.refkp), self.fname))
            
        # Compute keypoints and descriptors:
        kp, des = self.detector.detectAndCompute(imggray, None)
        str = "{} keypoints".format(len(kp))

        # Create a matcher if needed:
        if not hasattr(self, 'matcher'):
            self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck = True)

        # Compute matches between reference image and camera image, then sort them by distance:
        matches = self.matcher.match(des, self.refdes)
        matches = sorted(matches, key = lambda x:x.distance)
        str += ", {} matches".format(len(matches))

        # Keep only good matches:
        lastidx = 0
        for m in matches:
            if m.distance < self.distth: lastidx += 1
            else: break
        matches = matches[0:lastidx]
        str += ", {} good".format(len(matches))

        # If we have enough matches, compute homography:
        corners = []
        if len(matches) >= 10:
            obj = []
            scene = []

            # Localize the object (see JeVois C++ class ObjectMatcher for details):
            for m in matches:
                obj.append(self.refkp[m.trainIdx].pt)
                scene.append(kp[m.queryIdx].pt)

            # compute the homography
            hmg, mask = cv2.findHomography(np.array(obj), np.array(scene), cv2.RANSAC, 5.0)

            # Check homography conditioning using SVD:
            #sv = cv2.SVD.compute(hmg, cv2.SVD.NO_UV)

            # Project the reference image corners to the camera image:
            corners = cv2.perspectiveTransform(self.refcorners, hmg)
            
        # Display any results requested by the users:
        if outimg is not None and outimg.valid():
            if len(corners) == 4:
                jevois.drawLine(outimg, int(corners[0][0,0] + 0.5), int(corners[0][0,1] + 0.5),
                                int(corners[1][0,0] + 0.5), int(corners[1][0,1] + 0.5),
                                2, jevois.YUYV.LightPink)
                jevois.drawLine(outimg, int(corners[1][0,0] + 0.5), int(corners[1][0,1] + 0.5),
                                int(corners[2][0,0] + 0.5), int(corners[2][0,1] + 0.5),
                                2, jevois.YUYV.LightPink)
                jevois.drawLine(outimg, int(corners[2][0,0] + 0.5), int(corners[2][0,1] + 0.5),
                                int(corners[3][0,0] + 0.5), int(corners[3][0,1] + 0.5),
                                2, jevois.YUYV.LightPink)
                jevois.drawLine(outimg, int(corners[3][0,0] + 0.5), int(corners[3][0,1] + 0.5),
                                int(corners[0][0,0] + 0.5), int(corners[0][0,1] + 0.5),
                                2, jevois.YUYV.LightPink)
            jevois.writeText(outimg, str, 3, h+4, jevois.YUYV.White, jevois.Font.Font6x10)

        # Return object corners if we did indeed detect the object:
        hlist = []
        if len(corners) == 4: hlist.append(corners)
        
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

            jevois.sendSerial("D3 {} {} {} {} {} {} {} {} {} {} OBJ6D".
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
        # Get the next camera image (may block until it is captured) as OpenCV GRAY:
        imggray = inframe.getCvGray()
        h, w = imggray.shape
        
        # Start measuring image processing time:
        self.timer.start()

        # Get a list of quadrilateral convex hulls for all good objects:
        hlist = self.detect(imggray)

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
        
        # Convert input image to GRAY:
        imggray = jevois.convertToCvGray(inimg)
        h, w = imggray.shape

        # Get pre-allocated but blank output image which we will send over USB:
        outimg = outframe.get()
        outimg.require("output", w, h + 22, jevois.V4L2_PIX_FMT_YUYV)
        jevois.paste(inimg, outimg, 0, 0)
        jevois.drawFilledRect(outimg, 0, h, outimg.width, outimg.height-h, jevois.YUYV.Black)
        
        # Let camera know we are done using the input image:
        inframe.done()
        
        # Get a list of quadrilateral convex hulls for all good objects:
        hlist = self.detect(imggray, outimg)

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
        
