import pyjevois
if pyjevois.pro: import libjevoispro as jevois
else: import libjevois as jevois

import numpy as np
import cv2

## Python DNN post-processor for YOLOv8-Seg
#
# Adapted from https://github.com/ultralytics/ultralytics/blob/main/examples/YOLOv8-Segmentation-ONNXRuntime-Python/main.py
#
# This network produces two outputs:
# - 1x116x8400: standard YOLOv8 output (detected boxes and class confidences)
# - 1x32x160x160: segmentation masks
#
# Here, we combine them to produce a final display.
#
# @author Laurent Itti
# 
# @email itti\@usc.edu
# @address University of Southern California, HNB-07A, 3641 Watt Way, Los Angeles, CA 90089-2520, USA
# @copyright Copyright (C) 2024 by Laurent Itti, iLab and the University of Southern California
# @mainurl http://jevois.org
# @supporturl http://jevois.org/doc
# @otherurl http://iLab.usc.edu
# @license GPL v3
# @distribution Unrestricted
# @restrictions None
# @ingroup pydnn
class PyPostYOLOv8seg:
    # ###################################################################################################
    ## Constructor
    def __init__(self):
        # results of process(), held here for use by report():
        self.boxes = []
        self.segments = []
        self.masks = []

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

        self.conf = jevois.Parameter(self, 'conf', 'float',
                                     "Confidence threshold",
                                     0.4, pc)
        
        self.iou = jevois.Parameter(self, 'iou', 'float',
                                    "Intersection-over-union (IOU) threshold",
                                    0.45, pc)
        
        self.smoothing = jevois.Parameter(self, 'smoothing', 'float',
                                          "Amount of smoothing applied to contours, higher is smoother",
                                          0.5, pc)
        
        self.fillmask = jevois.Parameter(self, 'fillmask', 'bool',
                                         "Whether to draw semi-transparent filled masks (requires approximating " +
                                         "shapes by their convex hull)",
                                         False, pc)
        
        self.drawboxes = jevois.Parameter(self, 'drawboxes', 'bool',
                                          "Whether to draw boxes and text labels around detected objects",
                                          True, pc)
   
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
    ## Get network outputs
    # outs is a list of numpy arrays for the network's outputs.
    # preproc is a handle to the pre-processor that was used, useful to recover transforms from original image
    # to cropped/resized network inputs.
    def process(self, outs, preproc):
        if (len(outs) != 2): jevois.LFATAL("Need 2 outputs: boxes, masks")
        self.boxes, self.segments, self.masks = self.postprocess(outs, preproc, preproc.blobsize(0))

    # ###################################################################################################
    ## Helper to get class name and confidence as a clean string, and a color that varies with class name
    def getLabel(self, id, score):
        if self.classmap is None or id < 0 or id >= len(self.classmap): categ = 'unknown'
        else: categ = self.classmap[id]
        
        color = jevois.stringToRGBA(categ, 255)
        
        return ( ("%s: %.2f" % (categ, score * 100.0)), color & 0xffffffff)

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
            jevois.LFATAL("Sorry, legacy mode not supported by PyPostYOLOv8seg.py")

        # JeVois-Pro mode: Write the results as OpenGL overlay boxes and text on top of the video:
        if helper is not None and overlay:
            for i in range(len(self.segments)):
                x1, y1, x2, y2, score, cla = self.boxes[i]
                label, color = self.getLabel(int(cla), score);
                
                # Draw shape outline, and possibly filled mask using OpenGL, which requires convex contours:
                approx = cv2.approxPolyDP(self.segments[i], self.smoothing.get(), True)
                if self.fillmask.get():
                    convexhull = cv2.convexHull(approx)
                    helper.drawPoly(convexhull, color, True)
                else:
                    helper.drawPoly(approx, color, False)
                    
                # Draw box, unfilled:
                if self.drawboxes.get():
                    helper.drawRect(x1, y1, x2, y2, color & 0xffffffff, False)
                    helper.drawText(x1 + 3, y1 + 3, label, color & 0xffffffff)

    # ###################################################################################################
    def postprocess(self, preds, preproc, blobsize, nm = 32):
        x, protos = preds[0], preds[1]  # Two outputs: predictions and protos

        # Transpose the first output: (Batch_size, xywh_conf_cls_nm, Num_anchors) ->
        # (Batch_size, Num_anchors, xywh_conf_cls_nm)
        x = np.einsum("bcn->bnc", x)

        # Predictions filtering by conf-threshold
        x = x[np.amax(x[..., 4:-nm], axis=-1) > self.conf.get()]

        # Create a new matrix which merge these(box, score, cls, nm) into one
        # For more details about `numpy.c_()`: https://numpy.org/doc/1.26/reference/generated/numpy.c_.html
        x = np.c_[x[..., :4], np.amax(x[..., 4:-nm], axis=-1), np.argmax(x[..., 4:-nm], axis=-1), x[..., -nm:]]

        # NMS filtering
        x = x[cv2.dnn.NMSBoxes(x[:, :4], x[:, 4], self.conf.get(), self.iou.get())]

        # Decode and return
        if len(x) > 0:
            # Bounding boxes format change: cxcywh -> xyxy
            x[..., [0, 1]] -= x[..., [2, 3]] / 2
            x[..., [2, 3]] += x[..., [0, 1]]

            # Bounding boxes boundary clamp
            x[..., [0, 2]] = x[:, [0, 2]].clip(0, blobsize[1])
            x[..., [1, 3]] = x[:, [1, 3]].clip(0, blobsize[0])
            
            # Process masks
            masks = self.process_mask(protos[0], x[:, 6:], x[:, :4], blobsize)

            # Masks -> Segments(contours)
            segments = self.masks2segments(masks)

            # Scale all boxes from input blob to input image:
            for i in range(x.shape[0]):
                x[i, [0, 1]] = preproc.b2i(x[i, 0], x[i, 1], 0)
                x[i, [2, 3]] = preproc.b2i(x[i, 2], x[i, 3], 0)

            # Scale all segments from input blob to input image:
            for s in range(len(segments)):
                for i in range(segments[s].shape[0]):
                    x1, y1 = segments[s][i, [0, 1]]
                    x1, y1 = preproc.b2i(float(x1), float(y1), 0)
                    segments[s][i, [0, 1]] = [int(x1), int(y1)]

            return x[..., :6], segments, masks  # boxes, segments, masks
        else:
            return [], [], []
                            
    # ###################################################################################################
    @staticmethod
    def masks2segments(masks):
        """
        It takes a list of masks(n,h,w) and returns a list of segments(n,xy) (Borrowed from
        https://github.com/ultralytics/ultralytics/blob/465df3024f44fa97d4fad9986530d5a13cdabdca/ultralytics/utils/ops.py#L750)

        Args:
            masks (numpy.ndarray): the output of the model, which is a tensor of shape (batch_size, 160, 160).

        Returns:
            segments (List): list of segment masks.
        """
        segments = []
        for x in masks.astype("uint8"):
            c = cv2.findContours(x, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]  # or CHAIN_APPROX_NONE
            if c:
                c = np.array(c[np.array([len(x) for x in c]).argmax()]).reshape(-1, 2)
            else:
                c = np.zeros((0, 2))  # no segments found
            segments.append(c.astype("float32"))
        return segments

    # ###################################################################################################
    @staticmethod
    def crop_mask(masks, boxes):
        """
        It takes a mask and a bounding box, and returns a mask that is cropped to the bounding box. (Borrowed from
        https://github.com/ultralytics/ultralytics/blob/465df3024f44fa97d4fad9986530d5a13cdabdca/ultralytics/utils/ops.py#L599)

        Args:
            masks (Numpy.ndarray): [n, h, w] tensor of masks.
            boxes (Numpy.ndarray): [n, 4] tensor of bbox coordinates in relative point form.

        Returns:
            (Numpy.ndarray): The masks are being cropped to the bounding box.
        """
        n, h, w = masks.shape
        x1, y1, x2, y2 = np.split(boxes[:, :, None], 4, 1)
        r = np.arange(w, dtype=x1.dtype)[None, None, :]
        c = np.arange(h, dtype=x1.dtype)[None, :, None]
        return masks * ((r >= x1) * (r < x2) * (c >= y1) * (c < y2))

    # ###################################################################################################
    def process_mask(self, protos, masks_in, bboxes, im0_shape):
        """
        Takes the output of the mask head, and applies the mask to the bounding boxes.
        This produces masks of higher quality but is slower.
        (Borrowed from https://github.com/ultralytics/ultralytics/blob/465df3024f44fa97d4fad9986530d5a13cdabdca/
        ultralytics/utils/ops.py#L618)

        Args:
            protos (numpy.ndarray): [mask_dim, mask_h, mask_w].
            masks_in (numpy.ndarray): [n, mask_dim], n is number of masks after nms.
            bboxes (numpy.ndarray): bboxes re-scaled to original image shape.
            im0_shape (tuple): the size of the input image (h,w,c).

        Returns:
            (numpy.ndarray): The upsampled masks.
        """
        c, mh, mw = protos.shape
        masks = np.matmul(masks_in, protos.reshape((c, -1))).reshape((-1, mh, mw)).transpose(1, 2, 0)  # HWN
        masks = np.ascontiguousarray(masks)
        masks = self.scale_mask(masks, im0_shape)  # re-scale mask from P3 shape to original input image shape
        masks = np.einsum("HWN -> NHW", masks)  # HWN -> NHW
        masks = self.crop_mask(masks, bboxes)
        return np.greater(masks, 0.5)

    # ###################################################################################################
    @staticmethod
    def scale_mask(masks, im0_shape, ratio_pad=None):
        """
        Takes a mask, and resizes it to the original image size. (Borrowed from
        https://github.com/ultralytics/ultralytics/blob/465df3024f44fa97d4fad9986530d5a13cdabdca/ultralytics/utils/ops.py#L305)

        Args:
            masks (np.ndarray): resized and padded masks/images, [h, w, num]/[h, w, 3].
            im0_shape (tuple): the original image shape.
            ratio_pad (tuple): the ratio of the padding to the original image.

        Returns:
            masks (np.ndarray): The masks that are being returned.
        """
        im1_shape = masks.shape[:2]
        if ratio_pad is None:  # calculate from im0_shape
            gain = min(im1_shape[0] / im0_shape[0], im1_shape[1] / im0_shape[1])  # gain  = old / new
            pad = (im1_shape[1] - im0_shape[1] * gain) / 2, (im1_shape[0] - im0_shape[0] * gain) / 2  # wh padding
        else:
            pad = ratio_pad[1]

        # Calculate tlbr of mask
        top, left = int(round(pad[1] - 0.1)), int(round(pad[0] - 0.1))  # y, x
        bottom, right = int(round(im1_shape[0] - pad[1] + 0.1)), int(round(im1_shape[1] - pad[0] + 0.1))
        if len(masks.shape) < 2:
            raise ValueError(f'"len of masks shape" should be 2 or 3, but got {len(masks.shape)}')
        masks = masks[top:bottom, left:right]
        masks = cv2.resize(
            masks, (im0_shape[1], im0_shape[0]), interpolation=cv2.INTER_LINEAR
        )  # INTER_CUBIC would be better
        if len(masks.shape) == 2:
            masks = masks[:, :, None]
        return masks

