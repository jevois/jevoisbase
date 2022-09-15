import pyjevois
if pyjevois.pro: import libjevoispro as jevois
else: import libjevois as jevois

import numpy as np
import cv2

## Simple DNN pre-processor written in python
#
# This version is mainly for tutorial purposes and does not support as many features as the C++ PreProcessorBlob
#
# Compare this code to the C++ PreProcessorBlob (which has more functionality than here):
# - Abstract base: https://github.com/jevois/jevois/blob/master/include/jevois/DNN/PreProcessor.H
# - Header: https://github.com/jevois/jevois/blob/master/include/jevois/DNN/PreProcessorBlob.H
# - Implementation: https://github.com/jevois/jevois/blob/master/src/jevois/DNN/PreProcessorBlob.C
#
# @author Laurent Itti
# 
# @email itti\@usc.edu
# @address University of Southern California, HNB-07A, 3641 Watt Way, Los Angeles, CA 90089-2520, USA
# @copyright Copyright (C) 2022 by Laurent Itti, iLab and the University of Southern California
# @mainurl http://jevois.org
# @supporturl http://jevois.org/doc
# @otherurl http://iLab.usc.edu
# @license GPL v3
# @distribution Unrestricted
# @restrictions None
# @ingroup pydnn
class PyPreBlob:
    # ###################################################################################################
    ## [Optional] Constructor
    def __init__(self):
        self.results = [] # list of strings containing our latest results to report
        
    # ###################################################################################################
    ## [Optional] JeVois parameters initialization
    def init(self):
        pc = jevois.ParameterCategory("DNN Pre-Processing Options", "")
        
        self.scale = jevois.Parameter(self, 'scale', 'float',
                    "Value scaling factor applied to input pixels, or 0.0 to extract a " +
                    "UINT8 blob, typically for use with quantized networks",
                    2.0/255.0, pc)
        
        self.mean = jevois.Parameter(self, 'mean', 'fscalar',
                    "Mean value subtracted from input image",
                    (127.5, 127.5, 127.5), pc)
        
    # ###################################################################################################
    ## [Required] Main processing function: extract one or more 4D blobs from an image
    ## img is the input image from the camera sensor
    ## swaprb is true if we should swap red/blue channels (based on camera grab order vs. network desired order)
    ## attrs is a list of string blob specifiers, one per desired blob (see below for format and parsing)
    ## We return a tuple with 2 lists: list of blobs, and list of crop rectangles that were used to extract the blobs.
    ## The Network will use the blobs, and a PostProcessor may use the crops to, e.g., rescale object detection boxes
    ## back to the original image.
    def process(self, img, swaprb, attrs):
        # Clear any old results:
        self.results.clear()

        # We want to return a tuple containing two lists: list of 4D blobs (numpy arrays), and list of crop rectangles
        blobs = []
        crops = []

        # Create one blob per received size attribute:
        for a in attrs:
            # attr format is: [NCHW:|NHWC:|NA:|AUTO:]Type:[NxCxHxW|NxHxWxC|...][:QNT[:fl|:scale:zero]]
            alist = a.split(':')

            # In this simple example, we will use cv2.blobFromImage() which returns NCHW blobs. Hence require that:
            if alist[0] != 'NCHW': raise ValueError("Only NCHW blobs supported. Try the C++ preprocessor for more.")
            typ = alist[1]
            dims = alist[2]

            # We only support uint8 or float32 in this example:
            if typ == '8U': typ = cv2.CV_8U
            elif typ == '32F': typ = cv2.CV_32F
            else: raise ValueError("Only 8U or 32F blob types supported. Try the C++ preprocessor for more.")

            # Split the dims from NxCxHxW string:
            dlist = dims.split('x')
            if len(dlist) != 4: raise ValueError("Only 4D blobs supported. Try the C++ preprocessor for more.")
            
            # We just want the width and height:
            siz = (int(dlist[3]), int(dlist[2]))
            
            # Ok, hand it over to blobFromImage(). To keep it simple here, we do not do cropping, so that we can easily
            # recover the crop rectangles from the original image:
            b = cv2.dnn.blobFromImage(img, self.scale.get(), siz, self.mean.get(), swaprb, crop = False, ddepth = typ)
            blobs.append(b)

            # We did not crop, so the recangles are just the original image. We send rectangles as tuples of format:
            # ( (x,y), (w, h) )
            crops.append( ( (0, 0), (img.shape[1], img.shape[0] ) ) )

        return (blobs, crops)
    
    # ###################################################################################################
    ## [Optional] Report the latest results obtained by process() by drawing them
    ## outimg is None or a RawImage to draw into when in Legacy mode (drawing to an image send to USB)
    ## helper is None or a GUIhelper to do OpenGL drawings when in JeVois-Pro mode
    ## overlay is True if users wishes to see overlay text
    ## idle is true if keyboard/mouse have been idle for a while, which typically would reduce what is displayed.
    ##
    ## Note that report() is called on every frame even though the network may run slower or take some time to load and
    ## initialize, thus you should be prepared for report() being called even before process() has ever been called
    ## (i.e., create some class member variables to hold the reported results, initialize them to some defaults in your
    ## constructor, report their current values here, and update their values in process()).
    def report(self, outimg, helper, overlay, idle):
        # Nothing to report here. Note that the base C++ PreProcessor class will do some reporting for us.
        # You could delete that whole method, kept here for tutorial purposes.
        pass
    
