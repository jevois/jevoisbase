import pyjevois
if pyjevois.pro: import libjevoispro as jevois
else: import libjevois as jevois

import numpy as np
import cv2

## Simple YOLO DNN post-processor written in python
#
# Compare this code to the C++ PostProcessorDetect (which has more functionality than here):
# - Abstract base: https://github.com/jevois/jevois/blob/master/include/jevois/DNN/PostProcessor.H
# - Header: https://github.com/jevois/jevois/blob/master/include/jevois/DNN/PostProcessorDetect.H
# - Implementation: https://github.com/jevois/jevois/blob/master/src/jevois/DNN/PostProcessorDetect.C
#
# Instead of re-inventing the wheel, this code uses the YOLO post-processor that we have implemented in C++,
# as that C++ code is quite complex and multi-threaded for speed.
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
class PyPostYolo:
    # ###################################################################################################
    ## [Optional] Constructor
    def __init__(self):
        # results of process(), held here for use by report():
        self.classIds = []
        self.confidences = []
        self.boxes = []

        # map from class index to class name:
        self.classmap = None
        
        # Add a C++ YOLO post-processor. This instantiates a C++ PostProcessorDetectYOLOforPython.
        # That one already creates parameters for anchors, scalexy, and sigmoid via an underlying
        # C++ PostProcessorDetectYOLO
        self.yolopp = jevois.PyPostYOLO()
        
    # ###################################################################################################
    ## [Optional] JeVois parameters initialization
    def init(self):
        pc = jevois.ParameterCategory("DNN Post-Processing Options", "")
        
        self.classoffset = jevois.Parameter(self, 'classoffset', 'int',
                        "Offset to apply to class indices",
                        0, pc)

        self.classes = jevois.Parameter(self, 'classes', 'str',
                        "Path to text file with names of object classes",
                        '', pc)
        self.classes.setCallback(self.loadClasses)

        self.detecttype = jevois.Parameter(self, 'detecttype', 'str', 
                                           "Type of detection output format -- only RAWYOLO supported in Python",
                                           'RAWYOLO', pc)
        self.detecttype.setCallback(self.setDetectType)
        
        self.nms = jevois.Parameter(self, 'nms', 'float',
                        "Non-maximum suppression intersection-over-union threshold in percent",
                        45.0, pc)
        
        self.maxnbox = jevois.Parameter(self, 'maxnbox', 'uint',
                        "Max number of top-scoring boxes to report (for YOLO flavors, "
                        "this is the max for each scale)",
                        500, pc);
        
        self.cthresh = jevois.Parameter(self, 'cthresh', 'float',
                        "Classification threshold, in percent confidence",
                        20.0, pc)
        
        self.dthresh = jevois.Parameter(self, 'dthresh', 'float',
                        "Detection box threshold (in percent confidence) above which "
                        "predictions will be reported. Not all networks use a separate box threshold, "
                        "many only use one threshold confidence threshold (cthresh parameter). The YOLO "
                        "family is an example that uses both box and classification confidences",
                        15.0, pc)

        self.sigmoid = jevois.Parameter(self, 'sigmoid', 'bool',
                                        "When true, apply sigmoid to class confidence scores",
                                        False, pc)
       
        # note: compared to the C++ code, only RAWYOLO detecttype is supported here.
        
    # ###################################################################################################
    ## [Optional] Freeze some parameters that should not be changed at runtime.
    # The JeVois core will call this with doit being either True or False
    def freeze(self, doit):
        self.classes.freeze(doit)
        self.detecttype.freeze(doit)
        self.yolopp.freeze(doit)

    # ###################################################################################################
    ## [Optional] Parameter callback: Load class names when 'classes' parameter value is changed by model zoo
    def loadClasses(self, filename):
        if filename:
            jevois.LINFO(f"Loading {filename}...")
            f = open(pyjevois.share + '/' + filename, 'rt') # will throw if file not found
            self.classmap = f.read().rstrip('\n').split('\n')

    # ###################################################################################################
    ## [Optional] Parameter callback: set type of object detector
    def setDetectType(self, dt):
        if dt != 'RAWYOLO':
            jevois.LFATAL(f"Invalid detecttype {dt} -- only RAWYOLO is supported in Python")

    # ###################################################################################################
    ## [Required] Main processing function: parse network output blobs and store resulting labels and scores locally.
    # outs is a list of numpy arrays for the network's outputs.
    # preproc is a handle to the pre-processor that was used, useful to recover transforms from original image
    # to cropped/resized network inputs.
    def process(self, outs, preproc):
        if (len(outs) == 0): jevois.LFATAL("No outputs received, we need at least one.");
        
        # Clear any old results:
        self.classIds.clear()
        self.confidences.clear()
        self.boxes.clear()
        
        # To send serial messages, it may be useful to know the input image size:
        self.imagew, self.imageh = preproc.imagesize()

        # To draw boxes, we will need to:
        # - scale from [0..1]x[0..1] to blobw x blobh
        # - scale and center from blobw x blobh to input image w x h, provided by PreProcessor::b2i()
        # - when using the GUI, we further scale and translate to OpenGL display coordinates using GUIhelper::i2d()
        # Here we assume that the first blob sets the input size.
        bw, bh = preproc.blobsize(0)

        # Process the newly received network outputs:
        # Note: boxes returned are (x, y, w, h), which is what NMSboxes() below wants:
        classids, confs, boxes = self.yolopp.yolo(outs,
                                                  len(self.classmap),
                                                  self.dthresh.get() * 0.01,
                                                  self.cthresh.get() * 0.01,
                                                  bw, bh,
                                                  self.classoffset.get(),
                                                  self.maxnbox.get(),
                                                  self.sigmoid.get())
        
        # Cleanup overlapping boxes:
        indices = cv2.dnn.NMSBoxes(boxes, confs, self.cthresh.get() * 0.01, self.nms.get() * 0.01)

        # Now clamp boxes to be within blob, and adjust the boxes from blob size to input image size:
        for i in indices:
            x, y, w, h = boxes[i]

            # Clamp box coords to within network's input blob, and convert box to (x1, y1, x2, y2):
            x1 = min(bw - 1, max(0, x))
            x2 = min(bw - 1, max(0, x + w))
            y1 = min(bh - 1, max(0, y))
            y2 = min(bh - 1, max(0, y + h))

            # Scale box from input blob to input image:
            x1, y1 = preproc.b2i(x1, y1, 0)
            x2, y2 = preproc.b2i(x2, y2, 0)
            
            self.boxes.append( [x1, y1, x2, y2] )

            # Note: further scaling and translation to OpenGL display coords is handled internally by GUI helper
            # using GUIhelper::i2d() when we call helper.drawRect(), etc below in report()
        
        self.classIds = [classids[i] for i in indices]
        self.confidences = [confs[i] for i in indices]
        
    # ###################################################################################################
    ## Helper to get class name and confidence as a clean string, and a color that varies with class name
    def getLabel(self, id, conf):
        if self.classmap is None or id < 0 or id >= len(self.classmap): categ = 'unknown'
        else: categ = self.classmap[id]
        
        color = jevois.stringToRGBA(categ, 255)
        
        return ( ("%s: %.2f" % (categ, conf * 100.0)), color & 0xffffffff)
    
    # ###################################################################################################
    ## [Optional] Report the latest results obtained by process() by drawing them
    # outimg is None or a RawImage to draw into when in Legacy mode (drawing to an image sent to USB)
    # helper is None or a GUIhelper to do OpenGL drawings when in JeVois-Pro GUI mode
    # overlay is True if user wishes to see overlay text
    # idle is true if keyboard/mouse have been idle for a while, which typically would reduce what is displayed
    #
    # Note that report() is called on every frame even though the network may run slower or take some time to load and
    # initialize, thus you should be prepared for report() being called even before process() has ever been called
    # (i.e., create some class member variables to hold the reported results, initialize them to some defaults in your
    # constructor, report their current values here, and update their values in process()).
    def report(self, outimg, helper, overlay, idle):
        
        # Legacy JeVois mode: Write results into YUYV RawImage to send over USB:
        if outimg is not None:
            if overlay:
                for i in range(len(self.classIds)):
                    label, color = self.getLabel(self.classIds[i], self.confidences[i])
                    x1, y1, x2, y2 = self.boxes[i]
                    jevois.drawRect(outimg, x1, y1, x2 - x1, y2 - y1, 2, jevois.YUYV.LightGreen)
                    jevois.writeText(outimg, label, x1 + 6, y1 + 2, jevois.YUYV.LightGreen, jevois.Font.Font10x20)

        # JeVois-Pro mode: Write the results as OpenGL overlay boxes and text on top of the video:
        if helper is not None:
            if overlay:
                for i in range(len(self.classIds)):
                    label, color = self.getLabel(self.classIds[i], self.confidences[i])
                    x1, y1, x2, y2 = self.boxes[i]
                    helper.drawRect(x1, y1, x2, y2, color & 0xffffffff, True)
                    helper.drawText(x1 + 3, y1 + 3, label, color & 0xffffffff)

        # Could here send serial messages, or do some other processing over classIds, confidences and boxes
        
