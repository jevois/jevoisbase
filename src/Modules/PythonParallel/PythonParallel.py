import pyjevois
if pyjevois.pro: import libjevoispro as jevois
else: import libjevois as jevois
import cv2
import numpy as np
import multiprocessing as mp

# ###################################################################################################
## Image processing function, several instances will run in parallel
# It is just defined as a free function here to emphasize the fact that class member data will not be shared anyway
# across the workers.  NOTE: Do not attempt to use jevois.sendSerial() or any other functions of module jevois here, it
# will not work because computefunc() is running in a completely different process than jevois-daemon is. Just return
# any strings you wish to send out, and let your process() or processNoUSB() function do the sendSerial() instead.
def computefunc(inimggray, th1, th2):
    return cv2.Canny(inimggray, threshold1 = th1, threshold2 = th2, apertureSize = 3, L2gradient = False)

## Simple example of parallel image processing using OpenCV in Python on JeVois
#
# This module by default simply converts the input image to a grayscale OpenCV image, and then applies the Canny edge
# detection algorithm, 4 times running in parallel with 4 different edge coarseness settings. The resulting image is
# simply the 4 horizontally stacked results from the 4 parallel runs. Try to edit it to do something else!
#
# True multi-threaded processing is not supported by Python (the python \b threading module does not allow concurrent
# execution of several threads of python code). Parallel processing is somewhat feasible using the \b mutiprocessing
# python module, which is a process-based multiprocessing approach. Note that there are significant costs to
# parallelizing code over multiple processes, the main one being that data needs to be transferred back and forth
# between processes, using pipes, sockets, or other mechanisms. For machine vision, this is a significant problem as the
# amount of data (streaming video) that needs to be packaged, transferred, and unpacked is high. C++ is the preferred
# way of developping multi-threaded JeVois modules, where std::async() makes multi-threaded programming easy.
#
# \fixme <b>You should consider this module highly experimental and buggy!</b> This module is currently not working well
# when running with USB output. There is some internal issue with using the Python \b multiprocessing module in
# JeVois. Just creating a python process pool interferes with our USB video output driver, even if we simply create the
# pool and destroy it immediately without using it at all. Once the python process pool has been created, any subsequent
# attempt to change video format will fail with a video buffer allocation error. This module may still be useful for
# robotics applications where parallel python processing is needed but no video output to USB is necessary).
#
# \fixme This module is hence not enabled by default. You need to edit <b>JEVOIS:/config/videomappings.cfg</b>, and
# uncomment the line with \jvmod{PythonParallel} in it, to enable it.
#
# \fixme Conflated with this problem is the fact that guvcview sometimes, when it starts, turns streaming on, then grabs
# only 5 frames, then stream off, then set same video format again, and then stream on again. We are not sure why
# guvcview is doing this, however, this breaks this module since the second streamon fails as it is unable to allocate
# video buffers.
#
# Using this module
# -----------------
#
# One way we have been able to use this module with USB video outputs is: start `guvcview -f yuyv -x 352x288` (launches
# \jvmod{PythonSandbox}), then use the pull-down menu to select 1280x240 (switches to \jvmod{PythonParallel})
#
# This module is best used with no USB video outputs. Connect to JeVois over serial and issue:
# \verbatim
# setpar serout USB # to get text results through serial-over-USB, or use Hard if you want results on the 4-pin serial
# setpar serlog USB # to get log messages through serial-over-USB, or use Hard if you want them on the 4-pin serial
# setmapping2 YUYV 320 240 25.0 JeVois PythonParallel
# streamon
# \endverbatim
#
# As noted above, once you have loaded this module, any later attempts to change format again will fail.
#
# Creating your own module
# ------------------------
#
# See http://jevois.org/tutorials for tutorials on getting started with programming JeVois in Python without having
# to install any development software on your host computer.
#
# @author Laurent Itti
# 
# @videomapping YUYV 1280 240 25.0 YUYV 320 240 25.0 JeVois PythonParallel # CAUTION: has major issues
# @videomapping NONE 0 0 0.0 YUYV 320 240 25.0 JeVois PythonParallel
# @email itti\@usc.edu
# @address University of Southern California, HNB-07A, 3641 Watt Way, Los Angeles, CA 90089-2520, USA
# @copyright Copyright (C) 2018 by Laurent Itti, iLab and the University of Southern California
# @mainurl http://jevois.org
# @supporturl http://jevois.org/doc
# @otherurl http://iLab.usc.edu
# @license GPL v3
# @distribution Unrestricted
# @restrictions None
# @ingroup modules
class PythonParallel:
    # NOTE: Do not use a constructor in python multiprocessing JeVois modules, as it would be executed several times, in
    # each spawned worker process. Just do everything in the process() and/or processNoUSB() functions instead. Since
    # python allows runtime creation of new class data members, you can simply check whether they already exist, and, if
    # not, create them as new data members of the JeVois module.
    
    # ###################################################################################################
    ## Process function with no USB output
    def processNoUSB(self, inframe):
        # Create a parallel processing pool and a timer, if needed (on first frame only):
        if not hasattr(self, 'pool'):
            # create a multiprocessing pool, not specifying the number of processes, to use the number of cores:
            self.pool = mp.Pool()
            # Instantiate a JeVois Timer to measure our processing framerate:
            self.timer = jevois.Timer("PythonParallel", 100, jevois.LOG_INFO)
        
        # Get the next camera image (may block until it is captured) and convert it to OpenCV GRAY:
        inimggray = inframe.getCvGRAY()
            
        # Start measuring image processing time (NOTE: does not account for input conversion time):
        self.timer.start()
            
        # Detect edges using the Canny algorithm from OpenCV, launching 4 instances in parallel:
        futures = [ self.pool.apply_async(computefunc, args = (inimggray, 10*x, 20*x, )) for x in range(1,5) ]

        # Collect the results, handling any exception thrown by the workers. Here, we make sure we get() all the results
        # first, then rethrow the last exception received, if any, so that we do ensure that all results will be
        # collected before we bail out on an exception:
        results = []
        error = 0
        for ii in range(4):
            try: results.append(futures[ii].get(timeout = 10))
            except Exception as e: error = e
        if error: raise error
            
        # In real modules, we would do something with the results... Here, just report their size:
        str = ""
        for ii in range(4):
            h, w = results[ii].shape
            str += "Canny {}: {}x{}    ".format(ii, w, h)

        # Send a message to serout:
        jevois.sendSerial(str)

        # Report frames/s info to serlog:
        self.timer.stop()

    
    # ###################################################################################################
    ## Process function with USB output
    def process(self, inframe, outframe):
        # Create a parallel processing pool and a timer, if needed (on first frame only):
        if not hasattr(self, 'pool'):
            # create a multiprocessing pool, not specifying the number of processes, to use the number of cores:
            self.pool = mp.Pool()
            # Instantiate a JeVois Timer to measure our processing framerate:
            self.timer = jevois.Timer("PythonParallel", 100, jevois.LOG_INFO)
        
        # Get the next camera image (may block until it is captured) and convert it to OpenCV GRAY:
        inimggray = inframe.getCvGRAY()
            
        # Start measuring image processing time (NOTE: does not account for input conversion time):
        self.timer.start()
            
        # Detect edges using the Canny algorithm from OpenCV, launching 4 instances in parallel:
        futures = [ self.pool.apply_async(computefunc, args = (inimggray, 10*x, 20*x, )) for x in range(1,5) ]

        # Collect the results, handling any exception thrown by the workers. Here, we make sure we get() all the results
        # first, then rethrow the last exception received, if any, so that we do ensure that all results will be
        # collected before we bail out on an exception:
        results = []
        error = 0
        for ii in range(4):
            try: results.append(futures[ii].get(timeout = 10))
            except Exception as e: error = e
        if error: raise error
            
        # Aggregate the worker result images into a single output image:
        outimggray = np.hstack(results)

        # Write frames/s info from our timer into the edge map (NOTE: does not account for output conversion time):
        fps = self.timer.stop()
        height, width = outimggray.shape
        cv2.putText(outimggray, fps, (3, height - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 255, 1, cv2.LINE_AA)
            
        # Convert our GRAY output image to video output format and send to host over USB:
        outframe.sendCvGRAY(outimggray)

    # ###################################################################################################
    ## Required multiprocessing pool cleanup to avoid hanging on module unload
    # JeVois engine calls uninit(), if present, before destroying our module. FIXME: python multiprocessing still messes
    # up the system deeply, we become unable to allocate mmap'd UVC buffers after this module has been loaded.
    def uninit(self):
        # Close and join the worker pool if any, so we don't leave lingering processes:
        if hasattr(self, 'pool'):
            self.pool.close()
            self.pool.terminate()
            del self.pool

    # ###################################################################################################
    ## Parse a serial command forwarded to us by the JeVois Engine, return a string
    #def parseSerial(self, str):
    #    return "ERR: Unsupported command"
            
    # ###################################################################################################
    ## Return a string that describes the custom commands we support, for the JeVois help message
    #def supportedCommands(self):
    #    return ""

        

