import pyjevois
if pyjevois.pro: import libjevoispro as jevois
else: import libjevois as jevois

import numpy as np
import cv2
        
## Python DNN post-processor for DAMO YOLO
#
# Adapted from https://github.com/PINTO0309/PINTO_model_zoo/blob/main/334_DAMO-YOLO/demo/demo_DAMO-YOLO_onnx.py and
# https://github.com/tinyvision/DAMO-YOLO/blob/master/tools/demo.py
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
class PyPostDAMOyolo:
    # ###################################################################################################
    ## Constructor
    def __init__(self):
        # results of process(), held here for use by report():
        self.classIds = []
        self.confidences = []
        self.boxes = []

        # map from class index to class name:
        self.classmap = None

    # ###################################################################################################
    ## JeVois parameters initialization. These can be set by users in the GUI or JeVois model zoo file
    def init(self):
        pc = jevois.ParameterCategory("DNN Post-Processing Options", "")
        
        self.classes = jevois.Parameter(self, 'classes', 'str',
                       "Path to text file with names of object classes",
                       '', pc)
        self.classes.setCallback(self.loadClasses)

        self.nms = jevois.Parameter(self, 'nms', 'float',
                   "Non-maximum suppression intersection-over-union threshold in percent",
                   45.0, pc)
        
        self.maxnbox = jevois.Parameter(self, 'maxnbox', 'uint',
                       "Max number of top-scoring boxes to report",
                       500, pc);
        
        self.cthresh = jevois.Parameter(self, 'cthresh', 'float',
                       "Detection/classification score threshold, in percent confidence",
                       20.0, pc)

    # ###################################################################################################
    ## Freeze some parameters that should not be changed at runtime
    def freeze(self, doit):
        self.classes.freeze(doit)

    # ###################################################################################################
    ## Parameter callback: Load class names when 'classes' parameter value is changed by model zoo
    def loadClasses(self, filename):
        if filename:
            jevois.LINFO(f"Loading {filename}...")
            f = open(pyjevois.share + '/' + filename, 'rt') # will throw if file not found
            self.classmap = f.read().rstrip('\n').split('\n')

    # ###################################################################################################
    ## Multiclass non-maximum suppression as implemented by PINTO0309
    def multiclass_nms(self, bboxes, scores, score_th, nms_th, max_num):
        num_classes = scores.shape[1]
        bboxes = np.broadcast_to(bboxes[:, None], (bboxes.shape[0], num_classes, 4), )
        valid_mask = scores > score_th
        bboxes = bboxes[valid_mask]
        scores = scores[valid_mask]

        np_labels = valid_mask.nonzero()[1]

        indices = cv2.dnn.NMSBoxes(bboxes.tolist(), scores.tolist(), score_th, nms_th)
        # Note to Pinto0309: I believe NMSBoxes expects boxes as (x, y, w, h) but our boxes here are (x1, y1, x2, y2)
        
        if max_num > 0:
            indices = indices[:max_num]

        if len(indices) > 0:
            bboxes = bboxes[indices]
            scores = scores[indices]
            np_labels = np_labels[indices]
            return bboxes, scores, np_labels
        else:
            return np.array([]), np.array([]), np.array([])

    # ###################################################################################################
    ## Post-processing as implemented by PINTO0309
    def postprocess(self, scores, bboxes, score_th, nms_th):
        batch_size = bboxes.shape[0]
        for i in range(batch_size):
            if not bboxes[i].shape[0]: continue
            bboxes, scores, class_ids = self.multiclass_nms(bboxes[i], scores[i], score_th, nms_th, self.maxnbox.get())

        return bboxes, scores, class_ids
        # Note: if batch size > 1 this would only return the last results; anyway, batch size always 1 on JeVois
        
    # ###################################################################################################
    ## Get network outputs
    # outs is a list of numpy arrays for the network's outputs.
    # preproc is a handle to the pre-processor that was used, useful to recover transforms from original image
    # to cropped/resized network inputs.
    def process(self, outs, preproc):
        if (len(outs) != 2): jevois.LFATAL("Need 2 outputs: scores, bboxes")

        # To draw boxes, we will need to:
        # - scale from [0..1]x[0..1] to blobw x blobh
        # - scale and center from blobw x blobh to input image w x h, provided by PreProcessor::b2i()
        # - when using the GUI, we further scale and translate to OpenGL display coordinates using GUIhelper::i2d()
        # Here we assume that the first blob sets the input size.
        bw, bh = preproc.blobsize(0)

        boxes, self.confidences, self.classIds = self.postprocess(outs[0], outs[1], self.cthresh.get() * 0.01,
                                                                  self.nms.get() * 0.01)
        
        # Now clamp boxes to be within blob, and adjust the boxes from blob size to input image size:
        self.boxes.clear()
        
        for b in boxes:
            x1, y1, x2, y2 = b

            # Clamp box coords to within network's input blob:
            x1 = float(min(bw - 1, max(0, x1)))
            x2 = float(min(bw - 1, max(0, x2)))
            y1 = float(min(bh - 1, max(0, y1)))
            y2 = float(min(bh - 1, max(0, y2)))

            # Scale box from input blob to input image:
            x1, y1 = preproc.b2i(x1, y1, 0)
            x2, y2 = preproc.b2i(x2, y2, 0)
            
            self.boxes.append( [x1, y1, x2, y2] )

            # Note: further scaling and translation to OpenGL display coords is handled internally by GUI helper
            # using GUIhelper::i2d() when we call helper.drawRect(), etc below in report()
  
    # ###################################################################################################
    ## Helper to get class name and confidence as a clean string, and a color that varies with class name
    def getLabel(self, id, conf):
        if self.classmap is None or id < 0 or id >= len(self.classmap): categ = 'unknown'
        else: categ = self.classmap[id]
        
        color = jevois.stringToRGBA(categ, 255)
        
        return ( ("%s: %.2f" % (categ, conf * 100.0)), color & 0xffffffff)
    
    # ###################################################################################################
    ## Report the latest results obtained by process() by drawing them
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
        
