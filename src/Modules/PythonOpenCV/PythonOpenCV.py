import libjevois as jevois
import cv2
import numpy as np

class PythonOpenCV:
    ####################################################################################################
    # Constructor
    ####################################################################################################
    def __init__(self):
        print("Constructor")
        self.timer = jevois.Timer("canny", 100, jevois.LOG_INFO)
        
    ####################################################################################################
    # Process function with no USB output
    ####################################################################################################
    def process(self, inframe):
        print("process no usb")

    ####################################################################################################
    # Process function with USB output
    ####################################################################################################
    def process(self, inframe, outframe):
        self.timer.start()

        # Get the next camera image (may block until it is captured):
        inimg = inframe.get()
        #jevois.LINFO("Input image is {} {}x{}".format(jevois.fccstr(inimg.fmt), inimg.width, inimg.height))

        # Convert the input image to OpenCV grayscale:
        inimggray = jevois.convertToCvGray(inimg);

        # We are done with the input image:
        inframe.done()

        # Get the next available USB output image:
        outimg = outframe.get()
        #jevois.LINFO("Output image is {} {}x{}".format(jevois.fccstr(outimg.fmt), outimg.width, outimg.height))

        # Require that output image has same dims as input and is grayscale:
        outimg.require("output", inimg.width, inimg.height, jevois.V4L2_PIX_FMT_GREY);
        
        # Detect edges using the Canny algorithm from OpenCV and stuff the results into our outimgcv, and hence directly
        # into outimg since the pixel data is shared:
        edges = cv2.Canny(inimggray, 100, 200, apertureSize = 3)

        # Copy the edge map to the output image buffer to send over USB:
        jevois.convertCvGRAYtoRawImage(edges, outimg, 100)

        # Write frames/s info from our timer:
        fps = self.timer.stop();
        jevois.writeText(outimg, fps, 3, outimg.height-13, jevois.YUYV.White, jevois.Font.Font6x10)
        
        # We are done with the output, ready to send it to host over USB:
        outframe.send()

    ####################################################################################################
    # Parse a serial command forwarded to us by the JeVois Engine, return a string
    ####################################################################################################
    def parseSerial(self, str):
        return "ERR: Unsupported command"
    
    ####################################################################################################
    # Return a string that describes the custom commands we support, for the JeVois help message
    ####################################################################################################
    def supportedCommands(self):
        return ""
