import pyjevois
if pyjevois.pro: import libjevoispro as jevois
else: import libjevois as jevois

import numpy as np
        
## Stub DNN post-processor written in python
#
# When importing a network for which you are not sure about the exact output tensors and how to interpret them,
# you can use this stub first to see what outputs are produced by your network. You can then copy this stub to a
# new file name and edit it to actually decode the outputs of your network.
#
# You would create an entry like this in your network's YAML file:
#
# MyNet:
#    [...]
#    postproc: Python
#    pypost: "pydnn/post/PyPostStub.py"
#    [...]
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
class PyPostStub:    # When copying to a new file, REMEMBER to change class name to match file name
    # ###################################################################################################
    ## [Optional] Constructor
    def __init__(self):
        self.results = [] # use a persistent variable to store results in process() and then draw them in report()
        
    # ###################################################################################################
    ## [Optional] JeVois parameters initialization
    def init(self):
        # Could define parameters here that will be accessible to users in the GUI
        #pc = jevois.ParameterCategory("DNN Post-Processing Options", "")
        
        #self.classes = jevois.Parameter(self, 'classes', 'str',
        #                "Path to text file with names of object classes",
        #                '', pc)
        #self.classes.setCallback(self.loadClasses)

        # See other python post-processors for examples.
        pass
    
    # ###################################################################################################
    ## [Optional] Freeze some parameters that should not be changed at runtime.
    ## The JeVois core will call this with doit being either True or False
    def freeze(self, doit):
        #self.classes.freeze(doit)
        pass
        
    # ###################################################################################################
    ## [Optional] Parameter callback: Load class names  when 'classes' parameter value is changed,
    ## when a pipeline is selected from the model zoo
    #def loadClasses(self, filename):
    #    if filename:
    #        jevois.LINFO(f"Loading {filename}...")
    #        f = open(pyjevois.share + '/' + filename, 'rt') # will throw if file not found
    #        self.classmap = f.read().rstrip('\n').split('\n')

    # ###################################################################################################
    ## [Required] Main processing function: parse network output blobs and store resulting labels and scores locally.
    ## outs is a list of numpy arrays for the network's outputs.
    ## preproc is a handle to the pre-processor that was used, useful to recover transforms from original image
    ## to cropped/resized network inputs (not used here).
    def process(self, outs, preproc):
        # Clear any old results:
        self.results.clear()
        
        # Process the newly received network outputs:
        for o in outs:
            # Remove all dimensions that are 1, e.g., from 1x1x1000 to 1000
            o = np.squeeze(o)
            
            # [INSERT YOUR OWN CODE HERE]
            # Get any reportable results and store them in self.results or similar class variables,
            # those will then be rendered on screen and to serial ports in report()
            
    # ###################################################################################################
    ## [Optional] Report the latest results obtained by process() by drawing them
    ## outimg is None or a RawImage to draw into when in Legacy mode (drawing to an image sent to USB)
    ## helper is None or a GUIhelper to do OpenGL drawings when in JeVois-Pro GUI mode
    ## overlay is True if user wishes to see overlay text
    ## idle is true if keyboard/mouse have been idle for a while, which typically would reduce what is displayed
    ##
    ## Note that report() is called on every frame even though the network may run slower or take some time to load and
    ## initialize, thus you should be prepared for report() being called even before process() has ever been called
    ## (i.e., create some class member variables to hold the reported results, initialize them to some defaults in your
    ## constructor, report their current values here, and update their values in process()).
    def report(self, outimg, helper, overlay, idle):
        # Legacy JeVois mode: Write results into YUYV RawImage to send over USB:
        if outimg is not None:
            if overlay:
                jevois.writeText(outimg, 'Legacy mode', 50, 100, jevois.YUYV.White, jevois.Font.Font6x10)

        # JeVois-Pro mode: Write the results as OpenGL overlay text on top of the video:
        if helper is not None:
            if overlay:
                # itext writes one line of text and keeps track of incrementing the ordinate:
                helper.itext('Pro mode', 0, -1)
