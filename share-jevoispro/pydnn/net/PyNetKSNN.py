import pyjevois
if pyjevois.pro: import libjevoispro as jevois
else: import libjevois as jevois

import numpy as np
import cv2
from ksnn.api import KSNN
from ksnn.types import *

## Simple DNN network running on NPU and invoked from the Khadas KSNN library in python
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
class PyNetKSNN:
    # ###################################################################################################
    ## [Optional] Constructor
    def __init__(self):
        self.net = None
        
    # ###################################################################################################
    ## [Optional] JeVois parameters initialization
    def init(self):
        pc = jevois.ParameterCategory("DNN Network Options", "")

        self.dataroot = jevois.Parameter(self, 'dataroot', 'str',
                        "Root directory to use when config or model parameters are relative paths.",
                        pyjevois.share, pc) # pyjevois.share contains '/jevois[pro]/share'
        
        self.model = jevois.Parameter(self, 'model', 'str',
                        "Path to a binary file of model contains trained weights with .onnx extension. " +
                        "If path is relative, it will be prefixed by dataroot.",
                        "", pc);

        self.library = jevois.Parameter(self, 'library', 'str',
                        "Path to a library .so file of model support code, obtained during conversion to NPU. " +
                        "If path is relative, it will be prefixed by dataroot.",
                        "", pc);

        # Note: input shapes are stored in the network already; however, here we require them as a parameter that the
        # pre-processor will be able to query to create the input tensors. In the future, we may add some callback
        # function here to allow the pre-processor to get the input shapes directly from the ONNX net
        self.intensors = jevois.Parameter(self, 'intensors', 'str',
                        "Specification of input tensors",
                        '', pc)

    # ###################################################################################################
    ## [Optional] Freeze some parameters that should not be changed at runtime
    def freeze(self, doit):
        self.dataroot.freeze(doit)
        self.model.freeze(doit)
        self.intensors.freeze(doit)

    # ###################################################################################################
    ## [Required] Load the network from disk
    def load(self):
        self.net = KSNN('VIM3')
        self.net.nn_init(model = self.dataroot.get() + '/' + self.model.get(),
                         library = self.dataroot.get() + '/' + self.library.get(), level = 0)
        jevois.LINFO("Model ready.")
        
    # ###################################################################################################
    ## [Required] Main processing function: process input blobs through network and return output blobs
    ## blobs is a list of numpy arrays for the network's outputs
    ## Should return a tuple with (list of output blobs, list of info strings), where the info strings
    ## could contain some information about the network
    def process(self, blobs):
        if self.net is None:
            raise RuntimeError("Cannot process because no loaded network")

        if len(blobs) != len(self.inputs):
            raise ValueError(f"{len(blobs)} inputs received but network wants {len(self.inputs)}")

        # Create a dictionary to associate one blob to each network input:
        ins = { }
        for i in range(len(blobs)): ins[self.inputs[i].name] = blobs[i]
        
        # Run the network:
        outs = self.net.nn_inference(blobs, platform = 'ONNX', input_tensor = len(blobs), reorder = '0 1 2',
                                     output_tensor = numouts,
                                     output_format = output_format.OUT_FORMAT_FLOAT32)

        # Some simple info strings that will be shown along with preproc/postproc info:
        info = [ "* Network", "Forward pass OK" ]
        
        # Return outs and info:
        return (outs, info)
