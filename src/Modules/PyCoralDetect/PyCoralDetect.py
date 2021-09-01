import pyjevois
if pyjevois.pro: import libjevoispro as jevois
else: import libjevois as jevois
import cv2 as cv
import numpy as np
from PIL import Image
from PIL import ImageDraw
from pycoral.adapters import common
from pycoral.adapters import detect
from pycoral.utils.dataset import read_label_file
from pycoral.utils.edgetpu import make_interpreter
import time

## Object detection using Coral Edge TPU
#
# More pre-trained models are available at https://coral.ai/models/
#
#
# @author Laurent Itti
# 
# @videomapping YUYV 320 264 30.0 YUYV 320 240 30.0 JeVois PyClassificationDNN
# @email itti@usc.edu
# @address 880 W 1st St Suite 807, Los Angeles CA 90012, USA
# @copyright Copyright (C) 2020 by Laurent Itti
# @mainurl http://jevois.org
# @supporturl http://jevois.org
# @otherurl http://jevois.org
# @license GPL v3
# @distribution Unrestricted
# @restrictions None
# @ingroup modules
class PyCoralDetect:
    # ####################################################################################################
    ## Constructor
    def __init__(self):
        self.threshold = 0.4 # Confidence threshold (0..1), higher for stricter confidence.
        self.rgb = True      # True if model expects RGB inputs, otherwise it expects BGR
        
        # Select one of the models:
        self.model = 'MobileDetSSD' # expects 320x320

        # You should not have to edit anything beyond this point.
        if (self.model == 'MobileDetSSD'):
            classnames = 'coco_labels.txt'
            modelname = 'ssdlite_mobiledet_coco_qat_postprocess_edgetpu.tflite'

        # Load names of classes:
        sdir = pyjevois.share + '/coral/detection/'
        self.labels = read_label_file(sdir + classnames)

        # Load network:
        self.interpreter = make_interpreter(sdir + modelname)
        #self.interpreter = make_interpreter(*modelname.split('@'))
        self.interpreter.allocate_tensors()
        self.timer = jevois.Timer('Coral classification', 10, jevois.LOG_DEBUG)

    # ####################################################################################################
    def stringToRGBA(self, str):
        col = 0x80808080
        alpha = 0xff
        for c in str: col = ord(c) + ((col << 5) - col)
        col = (col & 0xffffff) | (alpha << 24)
        return col
    
    # ####################################################################################################
    def draw_objects(self, draw, objs, labels):
        """Draws the bounding box and label for each object."""
        for obj in objs:
            bbox = obj.bbox
            draw.rectangle([(bbox.xmin, bbox.ymin), (bbox.xmax, bbox.ymax)], outline='red')
            draw.text((bbox.xmin+10, bbox.ymin+10), '%s: %.2f' % (labels.get(obj.id, obj.id), obj.score), fill='red')
    
    # ####################################################################################################
    ## JeVois main processing function
    def process(self, inframe, outframe):
        frame = inframe.getCvRGB() if self.rgb else inframe.getCvBGR()
        self.timer.start()
        
        h = frame.shape[0]
        w = frame.shape[1]

        # Set the input:
        image = Image.fromarray(frame);
        _, scale = common.set_resized_input(self.interpreter, image.size,
                                            lambda size: image.resize(size, Image.ANTIALIAS))
  
        # Run the model
        start = time.perf_counter()
        self.interpreter.invoke()
        inference_time = time.perf_counter() - start
        
        # Get detections with high enough scores:
        objs = detect.get_objects(self.interpreter, self.threshold, scale)

        # Draw the detections:
        image = image.convert('RGB')
        self.draw_objects(ImageDraw.Draw(image), objs, self.labels)

        # Back to OpenCV:
        frame = np.array(image) 
        
        # Output to serial:
        #for obj in objs:
        #    print(self.labels.get(obj.id, obj.id))
        #    print('  id:    ', obj.id)
        #    print('  score: ', obj.score)
        #    print('  bbox:  ', obj.bbox)

        # Put efficiency information:
        cv.putText(frame, 'JeVois Coral Detection - ' + self.model, (3, 15),
                   cv.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv.LINE_AA)

        fps = self.timer.stop()
        label = fps + ', %dms' % (inference_time * 1000.0)
        cv.putText(frame, label, (3, h-5), cv.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv.LINE_AA)
        
        # Send output frame to host:
        if self.rgb: outframe.sendCvRGB(frame)
        else: outframe.sendCv(frame)

    # ####################################################################################################
    ## Process function with GUI output
    def processGUI(self, inframe, helper):
        # Start a new display frame, gets its size and also whether mouse/keyboard are idle:
        idle, winw, winh = helper.startFrame()

        # Draw full-resolution input frame from camera:
        x, y, w, h = helper.drawInputFrame("c", inframe, False, False)
        
        # Get the next camera image at processing resolution (may block until it is captured):
        frame = inframe.getCvRGBp() if self.rgb else inframe.getCvBGRp()

        # Start measuring image processing time:
        self.timer.start()

        # Set the input:
        image = Image.fromarray(frame);
        _, scale = common.set_resized_input(self.interpreter, image.size,
                                            lambda size: image.resize(size, Image.ANTIALIAS))
        # Run the model
        start = time.perf_counter()
        self.interpreter.invoke()
        inference_time = time.perf_counter() - start
        
        # Get detections with high enough scores:
        objs = detect.get_objects(self.interpreter, self.threshold, scale)

        # Draw the detections:
        for obj in objs:
            bbox = obj.bbox
            label = self.labels.get(obj.id, obj.id)
            col = self.stringToRGBA(label)
            helper.drawRect(bbox.xmin, bbox.ymin, bbox.xmax, bbox.ymax, col, True)
            helper.drawText(bbox.xmin+2, bbox.ymin+1, '%s: %.2f' % (label, obj.score), col)

        # Output to serial:
        #for obj in objs:
        #    print(self.labels.get(obj.id, obj.id))
        #    print('  id:    ', obj.id)
        #    print('  score: ', obj.score)
        #    print('  bbox:  ', obj.bbox)
        
        # Write frames/s info from our timer:
        fps = self.timer.stop()
        helper.iinfo(inframe, fps, winw, winh);
        helper.itext('JeVois-Pro Python Coral Detection - %s - %dms/inference' %
                     (self.model, inference_time * 1000.0), 0, -1)

        # End of frame:
        helper.endFrame()
