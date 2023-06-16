import pyjevois
if pyjevois.pro: import libjevoispro as jevois
else: import libjevois as jevois

## Simple test of programming JeVois modules in Python
#
# This module by default simply draws a cricle and a text message onto the grabbed video frames.
#
# Feel free to edit it and try something else. Note that this module does not import OpenCV, see the PythonOpenCV for a
# minimal JeVois module written in Python that uses OpenCV.
#
# See http://jevois.org/tutorials for tutorials on getting started with programming JeVois in Python without having
# to install any development software on your host computer.
#
# @author Laurent Itti
# 
# @videomapping YUYV 640 480 15.0 YUYV 640 480 15.0 JeVois PythonTest
# @email itti\@usc.edu
# @address University of Southern California, HNB-07A, 3641 Watt Way, Los Angeles, CA 90089-2520, USA
# @copyright Copyright (C) 2017 by Laurent Itti, iLab and the University of Southern California
# @mainurl http://jevois.org
# @supporturl http://jevois.org/doc
# @otherurl http://iLab.usc.edu
# @license GPL v3
# @distribution Unrestricted
# @restrictions None
# @ingroup modules
class PythonTest:
    # ###################################################################################################
    ## Constructor
    def __init__(self):
        jevois.LINFO("PythonTest Constructor")
        self.frame = 0 # a simple frame counter used to demonstrate sendSerial()
        self.timer = jevois.Timer("pytest", 100, jevois.LOG_INFO)

    # ###################################################################################################
    ## JeVois optional extra init once the instance is fully constructed
    def init(self):
        jevois.LINFO("PythonTest JeVois init")
        
        # Examples of adding user-tunable parameters to the module. Users can modify these using the JeVois console,
        # over a serial connection, using JeVois-Inventor, or using the JeVois-Pro GUI:
        pc = jevois.ParameterCategory("PythonTest Parameters", "")
        self.cx = jevois.Parameter(self, 'cx', 'int', "Circle horizontal center, in pixels", 320, pc)
        self.cy = jevois.Parameter(self, 'cy', 'int', "Circle vertical center, in pixels", 240, pc)
        self.radius = jevois.Parameter(self, 'radius', 'byte', "Circle radius, in pixels", 50, pc)
        self.thickness = jevois.Parameter(self, 'thickness', 'byte', "Circle thickness, in pixels", 2, pc)
        
    # ###################################################################################################
    ## Process function with no USB output
    def processNoUSB(self, inframe):
        jevois.LFATAL("process no usb not implemented")

    # ###################################################################################################
    ## Process function with USB output
    def process(self, inframe, outframe):
        # Get the next camera image (may block until it is captured):
        inimg = inframe.get()
        if self.frame == 0:
            jevois.LINFO("Input image is {} {}x{}".format(jevois.fccstr(inimg.fmt), inimg.width, inimg.height))

        # Get the next available USB output image:
        outimg = outframe.get()
        if self.frame == 0:
            jevois.LINFO("Output image is {} {}x{}".format(jevois.fccstr(outimg.fmt), outimg.width, outimg.height))

        # Example of getting pixel data from the input and copying to the output:
        jevois.paste(inimg, outimg, 0, 0)

        # We are done with the input image:
        inframe.done()

        # Example of in-place processing:
        jevois.hFlipYUYV(outimg)
        
        # Example of simple drawings and of accessing parameter values (users can change them via console, JeVois
        # Inventor, or JeVois-Pro GUI):
        jevois.writeText(outimg, "Hi from Python!", 20, 20, jevois.YUYV.White, jevois.Font.Font10x20)

        jevois.drawCircle(outimg, self.cx.get(), self.cy.get(), self.radius.get(), self.thickness.get(),
                          jevois.YUYV.White)

        # We are done with the output, ready to send it to host over USB:
        outframe.send()

        # Send a string over serial (e.g., to an Arduino). Remember to tell the JeVois Engine to display those messages,
        # as they are turned off by default. For example: 'setpar serout All' in the JeVois console:
        if self.frame % 100 == 0:
            jevois.sendSerial("DONE frame {}".format(self.frame));
        self.frame += 1

    # ###################################################################################################
    ## Process function with GUI output on JeVois-Pro
    def processGUI(self, inframe, helper):
        # Start a new display frame, gets its size and also whether mouse/keyboard are idle:
        idle, winw, winh = helper.startFrame()

        # Draw full-resolution input frame from camera:
        x, y, w, h = helper.drawInputFrame("c", inframe, False, False)
        
        # Get the next camera image (may block until it is captured):
        #inimg = inframe.getCvBGRp()
        
        # Start measuring image processing time (NOTE: does not account for input conversion time):
        self.timer.start()

        # Some drawings:
        helper.drawCircle(self.cx.get(), self.cy.get(), self.radius.get(), 0xffffffff, True)
        
        # Write frames/s info from our timer:
        fps = self.timer.stop()
        helper.iinfo(inframe, fps, winw, winh);

        # End of frame:
        helper.endFrame()

    # ###################################################################################################
    ## Parse a serial command forwarded to us by the JeVois Engine, return a string
    def parseSerial(self, str):
        jevois.LINFO("parseserial received command [{}]".format(str))
        if str == "hello":
            return self.hello()
        return "ERR Unsupported command"
    
    # ###################################################################################################
    ## Return a string that describes the custom commands we support, for the JeVois help message
    def supportedCommands(self):
        # use \n separator if your module supports several commands
        return "hello - print hello using python"

    # ###################################################################################################
    ## Internal method that can get invoked by users from the JeVois console as a custom command
    def hello(self):
        return "Hello from python!"
        
    # ###################################################################################################
    ## JeVois optional extra uninit before the instance is destroyed
    def uninit(self):
        jevois.LINFO("PythonTest JeVois uninit")
    
