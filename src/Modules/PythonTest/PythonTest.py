import libjevois as jevois

class PythonTest:
    # Constructor
    def __init__(self):
        print("Constructor")
        print(dir(jevois))

    # Process function with no USB output
    def process1(self, inframe):
        print("process no usb")

    # Process function with USB output
    def process2(self, inframe, outframe):
        print("process with usb")
        inimg = inframe.get()
        outimg = outframe.get()
        jevois.paste(inimg, outimg, 0, 0)
        inframe.done()
        outframe.send()

    # test hello function
    def hello(self):
        print("hello there from python")

