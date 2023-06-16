import pyjevois
if pyjevois.pro: import libjevoispro as jevois
else: import libjevois as jevois

import numpy as np
import cv2
        
## Python DNN post-processor for filtered color image
#
# Renders a filtered color image (with retinex-adjusted colors) on top of the original image, in between two
# user-draggable bars. Logic for the user bars here is converted from our ColorFiltering C++ module.
#
# URetinex-Net aims at recovering colors from very low light images. Hence, point your camera to a very dark area (e.g.,
# under your desk) to see the image enhancement provided by URetinex-Net.
#
# One of the goals of this post-processor is to demosntrate correct handling of coordinate transforms between display
# image, processing image, input tensor.
#
# @author Laurent Itti
# 
# @email itti\@usc.edu
# @address University of Southern California, HNB-07A, 3641 Watt Way, Los Angeles, CA 90089-2520, USA
# @copyright Copyright (C) 2023 by Laurent Itti, iLab and the University of Southern California
# @mainurl http://jevois.org
# @supporturl http://jevois.org/doc
# @otherurl http://iLab.usc.edu
# @license GPL v3
# @distribution Unrestricted
# @restrictions None
# @ingroup pydnn
class PyPostURetinex:
    # ###################################################################################################
    ## Constructor
    def __init__(self):
        self.rgba_map = None
        self.dragleft = False
        self.dragright = False
        self.left = -1.0e20
        self.right = -1.0e20
        self.pp = None

    # ###################################################################################################
    ## Get network outputs
    def process(self, outs, preproc):
        if len(outs) != 1: jevois.LERROR(f"Received {len(outs)} network outputs -- USING FIRST ONE")

        # Save RGBA depth map for later display:
        self.rgba_map = cv2.cvtColor(np.squeeze(outs[0] * 255).clip(0, 255).astype('uint8').transpose(1, 2, 0),
                                     cv2.COLOR_RGB2RGBA)

        # Compute overlay corner coords within the input image, for use in report():
        self.tlx, self.tly, self.cw, self.ch = preproc.getUnscaledCropRect(0)
        
        # We will need to do some coordinate conversion in report(), so keep a handle to the preproc:
        self.pp = preproc

    # ###################################################################################################
    ## Report the latest results obtained by process() by drawing them
    def report(self, outimg, helper, overlay, idle):

        col = 0xffffff7f # ARGB color of the vertical lines and square handles
        siz = 20 # size of the square handles, in image pixels
        
        # The main thing here is to properly handle coordinates: For example
        #
        # - Display typically is 1920x1080 @ 0,0, or it could also be 4K
        # - Captured video (full resolution stream for display) is typically 1920x1080 but could also be,
        #   e.g., 640x480 with scaling and translation to show up centered and as big as possible on the display
        # - Captured video for DNN processing typically is 1024x512 @ 0,0 -> use helper.i2d() to translate/scale
        #   from image to display, or helper.d2i() from display to image
        # - Input tensor (blob) for retinex processing typically is a rescaled and possibly letterboxed version
        #   of the processing image, e.g., to 320x180. Use preproc.b2i() or preproc.i2b()
        # - Mouse coordinates are in screen coordinates. Here, our left and right drag handles will be in processing
        #   image coordinates, since helper.drawLine(), etc will internally call i2d() as needed.
        #
        # Yes, the logic below is not trivial, you need to follow it carefully. It works in a broad range of cases:
        #
        # - start the DNN module with dual-stream capture of 1920x1080 (for display) + 1024x512 (for processing),
        #   and the overlay and drawn handles should display correctly.
        # - flip the 'letterbox' parameter of the pre-processor and check that graphics are still ok.
        # - Then try the DNN module in 640x480, which uses a single capture stream, and its display is centered
        #   and scaled on the screen. Again flip the 'letterbox' parameter of the pre-processor and graphics should
        #   still look good.
        
        if helper is not None and overlay and self.rgba_map is not None and self.pp is not None:
            # Processing image dims:
            iw, ih = self.pp.imagesize()

            # Initialize the handles at 1/4 and 3/4 of image width on first video frame after module is loaded:
            if self.left < -0.9e20:
                self.left = 0.25 * iw
                self.right = 0.75 * iw
                
            # Make sure the handles do not overlap and/or get out of the image bounds:
            if self.left > self.right - siz:
                if self.dragright: self.left = self.right - siz
                else: self.right = self.left + siz
                
            self.left = max(siz, min(iw - siz * 2, self.left))
            self.right = max(self.left + siz, min(iw - siz, self.right))

            # Mask and draw the overlay. To achieve this, we convert the whole result image to RGBA and then assign a
            # zero alpha channel to all pixels to the left of the 'left' bound and to the right of the 'right' bound.
            # First we need to convert from image coords to blob coords:
            blob_left, blob_top = self.pp.i2b(self.left, 0.0, 0)
            blob_right, blob_bot = self.pp.i2b(self.right, ih, 0)
                                         
            ovl = self.rgba_map
            ovl[:, :int(blob_left), 3] = 0  # make left side transparent
            ovl[:, int(blob_right):, 3] = 0 # make right side transparent

            # Convert box coords from input image to display ("c" is the displayed camera image):
            tl = helper.i2d(self.tlx, self.tly, "c")
            wh = helper.i2ds(self.cw, self.ch, "c")

            # Draw as a semi-transparent overlay. OpenGL will do scaling/stretching/blending as needed:
            helper.drawImage("r", ovl, True, int(tl.x), int(tl.y), int(wh.x), int(wh.y), False, True)

            # Draw drag handles:
            ovtop = self.tly # top of the overlay image (including possible preproc letterboxing)
            ovbot = self.tly + self.ch # bottom of overlay
            ovmid = 0.5 * (ovtop + ovbot) # vertical midpoint for our handles
            
            helper.drawLine(self.left, ovtop, self.left, ovbot, col)
            helper.drawRect(self.left - siz, ovmid - siz/2, self.left, ovmid + siz/2, col, True)
            helper.drawLine(self.right, ovtop, self.right, ovbot, col)
            helper.drawRect(self.right, ovmid - siz/2, self.right + siz, ovmid + siz/2, col, True)

            # Adjust the left and right handles if they get clicked and dragged:
            mp = helper.getMousePos()        # in screen coordinates
            ip = helper.d2i(mp.x, mp.y, "c") # in image coordinates
            
            if helper.isMouseClicked(0):
                # Are we clicking on the left or right handle?
                if ip.x > self.left-siz and ip.x < self.left and ip.y > (ih - siz)/2 and ip.y < (ih + siz)/2:
                    self.dragleft = True
                    
                if ip.x > self.right and ip.x < self.right+siz and ip.y > (ih - siz)/2 and ip.y < (ih + siz)/2:
                    self.dragright = True

            if helper.isMouseDragging(0):
                if self.dragleft: self.left = ip.x + 0.5 * siz
                if self.dragright: self.right = ip.x - 0.5 * siz
                # We will enforce validity of left and right on next frame, before we draw

            if helper.isMouseReleased(0):
                self.dragleft = False
                self.dragright = False
