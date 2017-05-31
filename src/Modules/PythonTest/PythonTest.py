import libjevois as jevois

class PythonTest:
    ####################################################################################################
    # Constructor
    ####################################################################################################
    def __init__(self):
        print("Constructor")
        print(dir(jevois))

    ####################################################################################################
    # Process function with no USB output
    ####################################################################################################
    def process(self, inframe):
        print("process no usb")

    ####################################################################################################
    # Process function with USB output
    ####################################################################################################
    def process(self, inframe, outframe):
        print("process with usb")

        # Get the next camera image (may block until it is captured):
        inimg = inframe.get()
        print("Input image is {} {}x{}".format(jevois.fccstr(inimg.fmt), inimg.width, inimg.height))

        # Get the next available USB output image:
        outimg = outframe.get()
        print("Output image is {} {}x{}".format(jevois.fccstr(outimg.fmt), outimg.width, outimg.height))

        # Example of getting pixel data from the input and copying to the output:
        jevois.paste(inimg, outimg, 0, 0)

        # We are done with the input image:
        inframe.done()

        # Example of in-place processing:
        jevois.hFlipYUYV(outimg)

        # We are done with the output, ready to send it to host over USB:
        outframe.send()

    ####################################################################################################
    # Parse a serial command forwarded to us by the JeVois Engine, return a string
    ####################################################################################################
    def parseSerial(self, str):
        print("parseserial received command [{}]".format(str))
        if str == "hello":
            return self.hello()
        return "ERR: Unsupported command"
    
    ####################################################################################################
    # Return a string that describes the custom commands we support, for the JeVois help message
    ####################################################################################################
    def supportedCommands(self):
        # use \n seperator if your module supports several commands
        return "hello - print hello using python"

    ####################################################################################################
    # Internal method that gets invoked as a custom command
    ####################################################################################################
    def hello(self):
        return("Hello from python!")
        
