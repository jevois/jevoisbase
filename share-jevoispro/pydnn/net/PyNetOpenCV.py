import pyjevois
if pyjevois.pro: import libjevoispro as jevois
else: import libjevois as jevois

import numpy as np
import cv2

## Simple DNN network invoked from in python
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
class PyNetOpenCV:
    # ###################################################################################################
    ## [Optional] Constructor
    def __init__(self):
        self.net = None # we do not have a network yet, we will load it after parameters have been set
        self.gflops = None # need a network and an image to compute number of operations in net
        
    # ###################################################################################################
    ## [Optional] JeVois parameters initialization
    def init(self):
        pc = jevois.ParameterCategory("DNN Network Options", "")

        self.dataroot = jevois.Parameter(self, 'dataroot', 'str',
                        "Root directory to use when config or model parameters are relative paths.",
                        pyjevois.share, pc) # pyjevois.share contains '/jevois[pro]/share'
        
        self.intensors = jevois.Parameter(self, 'intensors', 'str',
                        "Specification of input tensors",
                        '', pc)
        
        self.outtensors = jevois.Parameter(self, 'outtensors', 'str',
                        "Specification of output tensors (optional)",
                        '', pc)
        
        self.config = jevois.Parameter(self, 'config', 'str',
                        "Path to a text file that contains network configuration. " +
                        "Can have extension .prototxt (Caffe), .pbtxt (TensorFlow), or .cfg (Darknet). " +
                        "If path is relative, it will be prefixed by dataroot.",
                        '', pc);
        
        self.model = jevois.Parameter(self, 'model', 'str',
                        "Path to a binary file of model contains trained weights. " +
                        "Can have extension .caffemodel (Caffe), .pb (TensorFlow), .t7 or .net (Torch), " +
                        ".tflite (TensorFlow Lite), or .weights (Darknet). If path is relative, it will be " +
                        "prefixed by dataroot.",
                        "", pc);
        
    # ###################################################################################################
    ## [Optional] Freeze some parameters that should not be changed at runtime
    def freeze(self, doit):
        self.dataroot.freeze(doit)
        self.intensors.freeze(doit)
        self.outtensors.freeze(doit)
        self.config.freeze(doit)
        self.model.freeze(doit)

    # ###################################################################################################
    ## [Required] Load the network from disk
    def load(self):
        self.net = cv2.dnn.readNet(self.dataroot.get() + '/' + self.model.get(),
                                   self.dataroot.get() + '/' + self.config.get())
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        
    # ###################################################################################################
    ## [Required] Main processing function: process input blobs through network and return output blobs
    ## blobs is a list of numpy arrays for the network's outputs
    ## Should return a tuple with (list of output blobs, lits of info strings), where the info strings
    ## could contain some information about the network
    def process(self, blobs):
        if self.net is None: raise RuntimeError("Cannot process because no loaded network")
        if len(blobs) != 1: raise ValueError("Only one input blob is supported")

        # Run the network:
        self.net.setInput(blobs[0])
        outs = self.net.forward()
        
        # Some simple info strings that will be shown along with preproc/postproc info:
        if self.gflops is None: self.gflops = int(self.net.getFLOPS(blobs[0].shape) * 1.0e-9)
        
        info = [
            "* Network",
            "{}GFLOPS".format(self.gflops),
        ]

        # Return outs and info:
        return (outs, info)
