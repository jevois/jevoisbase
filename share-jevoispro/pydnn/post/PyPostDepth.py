import pyjevois
if pyjevois.pro: import libjevoispro as jevois
else: import libjevois as jevois

import numpy as np
        
## Python DNN post-processor for depth map
#
# Renders a depth map as a semi-transparent overlay over the input frames.
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
class PyPostDepth:
    # ###################################################################################################
    ## Constructor
    def __init__(self):
        self.depthmap = None

    # ###################################################################################################
    ## JeVois parameters initialization
    def init(self):
        pc = jevois.ParameterCategory("DNN Post-Processing Options", "")
        
        self.alpha = jevois.Parameter(self, 'alpha', 'byte',
                        "Alpha value for depth overlay",
                        200, pc)

        self.dfac = jevois.Parameter(self, 'dfac', 'float',
                        "Depth conversion factor to bring values to [0..255]",
                        50, pc)

    # ###################################################################################################
    ## Get network outputs
    def process(self, outs, preproc):
        if len(outs) != 1: jevois.LERROR(f"Received {len(outs)} network outputs -- USING FIRST ONE")
        dmap = (np.squeeze(outs[0]) * self.dfac.get()).clip(0, 255).astype('uint8')

        # Compute overlay corner coords within the input image, for use in report():
        self.tlx, self.tly, self.cw, self.ch = preproc.getUnscaledCropRect(0)

        # Save RGBA depth map for later display:
        alphamap = np.full_like(dmap, self.alpha.get())
        self.depthmap = np.dstack( (dmap, dmap, dmap, alphamap) ) # RGBA
            
    # ###################################################################################################
    ## Report the latest results obtained by process() by drawing them
    def report(self, outimg, helper, overlay, idle):
        
        if helper is not None and overlay and self.depthmap is not None:
            # Convert box coords from input image to display ("c" is the displayed camera image):
            tl = helper.i2d(self.tlx, self.tly, "c")
            wh = helper.i2ds(self.cw, self.ch, "c")

            # Draw as a semi-transparent overlay. OpenGL will do scaling/stretching/blending as needed:
            helper.drawImage("depthmap", self.depthmap, True, int(tl.x), int(tl.y), int(wh.x), int(wh.y), False, True)
