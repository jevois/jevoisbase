import libjevois as jevois
import cv2
import numpy as np
from pylibdmtx.pylibdmtx import decode

## Decoding of DataMatrix (DMTX) 2D barcodes
#
# This module finds and decodes DataMatrix 2D barcodes.
#
# It uses libdmtx as a backend, and the pylibdmtx python wrapper.
#
# On host, you will need: sudo apt install libdmtx-dev; pip install pylibdmtx
#
# @author Laurent Itti
# 
# @videomapping YUYV 320 280 30.0 YUYV 320 240 30.0 JeVois PyDMTX
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
class PyDMTX:
    # ###################################################################################################
    ## Constructor
    def __init__(self):
        # Instantiate a JeVois Timer to measure our processing framerate:
        self.timer = jevois.Timer("dmtx", 100, jevois.LOG_INFO)
        
    # ###################################################################################################
    ## Process function with USB output
    def process(self, inframe, outframe):
        # Get the next camera image (may block until it is captured) as OpenCV BGR:
        inimg = inframe.getCvBGR()
        
        # Start measuring image processing time (NOTE: does not account for input conversion time):
        self.timer.start()

        # Create dark-gray (value 80) image for the bottom panel, 40 pixels tall:
        msgbox = np.zeros((40, inimg.shape[1], 3), dtype = np.uint8) + 80

        # Find and decode any DataMatrix symbols:
        dec = decode(inimg)

        # Draw the results in the input image (which we will copy to output):
        y = 13
        for d in dec:
            cv2.rectangle(inimg, (d.rect.left, d.rect.top), (d.rect.left+d.rect.width-1, d.rect.top+d.rect.height-1),
                          (255,0,0), 2)
            cv2.putText(msgbox, d.data.decode("utf-8"), (3, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255))
            y = y + 12
         
        # Create our output image as a copy of the input plus our message box:
        outimg = np.vstack((inimg, msgbox))

        # Write a title:
        cv2.putText(outimg, "JeVois Python DataMatrix", (3, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255))
        
        # Write frames/s info from our timer into the edge map (NOTE: does not account for output conversion time):
        fps = self.timer.stop()
        cv2.putText(outimg, fps, (3, inimg.shape[0] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255))

        # Convert our OpenCv output image to video output format and send to host over USB:
        outframe.sendCv(outimg)
