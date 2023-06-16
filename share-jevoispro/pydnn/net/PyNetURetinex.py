import pyjevois
if pyjevois.pro: import libjevoispro as jevois
else: import libjevois as jevois

import numpy as np
import cv2
import onnxruntime as rt

## Simple DNN network invoked from ONNX-Runtime in python for URetinex-Net
#
# This network expects a fixed 1x1 tensor for a parameter, in addition to the image input
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
class PyNetURetinex:
    # ###################################################################################################
    ## [Optional] Constructor
    def __init__(self):
        self.session = None # we do not have a network yet, we will load it after parameters have been set
        
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

        # Note: input shapes are stored in the network already; however, here we require them as a parameter that the
        # pre-processor will be able to query to create the input tensors. In the future, we may add some callback
        # function here to allow the pre-processor to get the input shapes directly from the ONNX net
        self.intensors = jevois.Parameter(self, 'intensors', 'str',
                        "Specification of input tensors",
                        '', pc)

        self.exposure = jevois.Parameter(self, 'exposure', 'float',
                                         "Exposure ratio for URetinex-Net, with typical values 3 to 5",
                                         5.0, pc)
        
    # ###################################################################################################
    ## [Optional] Freeze some parameters that should not be changed at runtime
    def freeze(self, doit):
        self.dataroot.freeze(doit)
        self.model.freeze(doit)
        self.intensors.freeze(doit)

    # ###################################################################################################
    ## [Required] Load the network from disk
    def load(self):
        self.session = rt.InferenceSession(self.dataroot.get() + '/' + self.model.get(),
                                           providers = rt.get_available_providers())

        # Get a list of input specs to be used during inference:
        self.inputs = self.session.get_inputs()
        
    # ###################################################################################################
    ## [Required] Main processing function: process input blobs through network and return output blobs
    ## blobs is a list of numpy arrays for the network's outputs
    ## Should return a tuple with (list of output blobs, list of info strings), where the info strings
    ## could contain some information about the network
    def process(self, blobs):
        if self.session is None: raise RuntimeError("Cannot process because no loaded network")
        if len(blobs) != 1: raise ValueError(f"{len(blobs)} inputs received but network wants 1")

        # Get exposure ratio parameter and create a tensor from it. Then create a dictionary to associate one blob to
        # each network input:
        expo = self.exposure.get()
        
        ins = { self.inputs[0].name: blobs[0],
                self.inputs[1].name: np.asarray(expo, dtype=np.float32).reshape(self.inputs[1].shape) }
        
        # Run the network:
        outs = self.session.run(None, ins)

        # Some simple info strings that will be shown along with preproc/postproc info:
        info = [ "* Network", "Forward pass OK", f"Exposure ratio: {expo}" ]
        
        # Return outs and info:
        return (outs, info)
