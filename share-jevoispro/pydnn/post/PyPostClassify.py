import pyjevois
if pyjevois.pro: import libjevoispro as jevois
else: import libjevois as jevois

import numpy as np
        
## Simple classification DNN post-processor written in python
#
# Compare this code to the C++ PostProcessorClassify (which has more functionality than here):
# - Abstract base: https://github.com/jevois/jevois/blob/master/include/jevois/DNN/PostProcessor.H
# - Header: https://github.com/jevois/jevois/blob/master/include/jevois/DNN/PostProcessorClassify.H
# - Implementation: https://github.com/jevois/jevois/blob/master/src/jevois/DNN/PostProcessorClassify.C
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
class PyPostClassify:
    # ###################################################################################################
    ## [Optional] Constructor
    def __init__(self):
        self.results = [] # list of strings containing our latest top predictions and scores
        self.classmap = None # map from class index to class name
        
    # ###################################################################################################
    ## [Optional] JeVois parameters initialization
    def init(self):
        pc = jevois.ParameterCategory("DNN Post-Processing Options", "")
        
        self.classes = jevois.Parameter(self, 'classes', 'str',
                        "Path to text file with names of object classes",
                        '', pc)
        self.classes.setCallback(self.loadClasses)
        
        self.cthresh = jevois.Parameter(self, 'cthresh', 'float',
                        "Classification threshold, in percent confidence",
                        20.0, pc)
        
        self.top = jevois.Parameter(self, 'top', 'uint',
                        "Max number of predictions above cthresh to report",
                        5, pc)
        
        self.classoffset = jevois.Parameter(self, 'classoffset', 'int',
                        "Offset to apply to class indices",
                        0, pc)
        
    # ###################################################################################################
    ## [Optional] Freeze some parameters that should not be changed at runtime.
    ## The JeVois core will call this with doit being either True or False
    def freeze(self, doit):
        self.classes.freeze(doit)

    # ###################################################################################################
    ## [Optional] Parameter callback: Load class names  when 'classes' parameter value is changed,
    ## when a pipeline is selected from the model zoo
    def loadClasses(self, filename):
        if filename:
            jevois.LINFO(f"Loading {filename}...")
            f = open(pyjevois.share + '/' + filename, 'rt') # will throw if file not found
            self.classmap = f.read().rstrip('\n').split('\n')

    # ###################################################################################################
    ## [Required] Main processing function: parse network output blobs and store resulting labels and scores locally.
    ## outs is a list of numpy arrays for the network's outputs.
    ## preproc is a handle to the pre-processor that was used, useful to recover transforms from original image
    ## to cropped/resized network inputs (not used here).
    def process(self, outs, preproc):
        # Clear any old results:
        self.results.clear()
        
        # Getting the value of a JeVois Parameter is somewhat costly, so cache it if we will be using that value
        # repeatedly in an inner loop:
        fudge = self.classoffset.get()
        topn = self.top.get()
        
        # Process the newly received network outputs:
        for o in outs:
            # Remove all dimensions that are 1, e.g., from 1x1x1000 to 1000
            o = np.squeeze(o)
            
            # Sort descending by scores and get the top scores:
            topNidxs = np.argsort(o)[::-1][:topn]

            # Store results for later display in report():
            cth = self.cthresh.get() * 0.01

            if self.classmap:
                # Show class name and confidence:
                for (i, idx) in enumerate(topNidxs):
                    if o[idx] > cth:
                        classidx = idx + fudge
                        if classidx >= 0 and classidx < len(self.classmap):
                            self.results.append('%s: %.2f' % (self.classmap[classidx], o[idx]*100))
                        else:
                            self.results.append('Unknown: %.2f' % o[idx])
            else:
                # Show class number and confidence:
                for (i, idx) in enumerate(topNidxs):
                    if o[idx] > cth:
                        self.results.append('%d: %.2f' % (idx, o[idx]*100))

            # Make sure we always display topn lines of results for each output blob:
            while len(self.results) < topn * len(outs):
                self.results.append('-')
    
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
                x, y = 220, 16 # mirror C++ PostProcessorClassify::report()
                jevois.writeText(outimg, 'Top-%d above %.2f%%' % (self.top.get(), self.cthresh.get()),
                                 x, y, jevois.YUYV.White, jevois.Font.Font6x10)
                y += 15
                for r in self.results:
                    jevois.writeText(outimg, r, x, y, jevois.YUYV.White, jevois.Font.Font6x10)
                    y += 11

        # JeVois-Pro mode: Write the results as OpenGL overlay text on top of the video:
        if helper is not None:
            if overlay:
                for r in self.results:
                    # itext writes one line of text and keeps track of incrementing the ordinate:
                    helper.itext(r, 0, -1)
