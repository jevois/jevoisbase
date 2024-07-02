import pyjevois
if pyjevois.pro: import libjevoispro as jevois
else: import libjevois as jevois
import cv2 as cv
import numpy as np
from PIL import Image
from pycoral.utils import edgetpu
from pycoral.adapters import common
from pycoral.adapters import segment
import time

## Semantic segmentation using Coral Edge TPU
#
# More pre-trained models are available at https://coral.ai/models/
#
#
# @author Laurent Itti
# 
# @videomapping YUYV 320 264 30.0 YUYV 320 240 30.0 JeVois PyCoralSegment
# @videomapping JVUI 0 0 30.0 CropScale=RGB24@512x288:YUYV 1920 1080 30.0 JeVois PyCoralSegment
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
class PyCoralSegment:
    # ####################################################################################################
    ## Constructor
    def __init__(self):
        if jevois.getNumInstalledTPUs() == 0:
            jevois.LFATAL("A Google Coral EdgeTPU is required for this module (PCIe M.2 2230 A+E or USB)")
            
        self.rgb = True        # True if model expects RGB inputs, otherwise it expects BGR
        self.keepaspect = True # Keep aspect ratio using zero padding
        alpha = 128            # Transparency alpha values for processGUI, higher is less transparent
        tidx = 0               # Class index of transparent background
        
        # Select one of the models:
        self.model = 'UNet128' # expects 128x128
        #self.model = 'MobileNetV2DeepLabV3' # expects 513x513

        # You should not have to edit anything beyond this point.
        if (self.model == 'MobileNetV2DeepLabV3'):
            modelname = 'deeplabv3_mnv2_dm05_pascal_quant_edgetpu.tflite'
        elif (self.model == 'UNet128'):
            modelname = 'keras_post_training_unet_mv2_128_quant_edgetpu.tflite'
            tidx = 1
            
        # Load network:
        sdir = pyjevois.share + '/coral/segmentation/'
        self.interpreter = edgetpu.make_interpreter(sdir + modelname)
        #self.interpreter = edgetpu.make_interpreter(*modelname.split('@'))
        self.interpreter.allocate_tensors()
        self.timer = jevois.Timer('Coral segmentation', 10, jevois.LOG_DEBUG)
        self.cmapRGB = self.create_pascal_label_colormap()
        self.cmapRGBA = self.create_pascal_label_colormapRGBA(alpha, tidx)
                      
    # ####################################################################################################
    def create_pascal_label_colormap(self):
        """Creates a label colormap used in PASCAL VOC segmentation benchmark.
        Returns:
        A Colormap for visualizing segmentation results.
        """
        colormap = np.zeros((256, 3), dtype=int)
        indices = np.arange(256, dtype=int)

        for shift in reversed(range(8)):
            for channel in range(3):
                colormap[:, channel] |= ((indices >> channel) & 1) << shift
                indices >>= 3

        return colormap.astype(np.uint8)
    
    # ####################################################################################################
    def create_pascal_label_colormapRGBA(self, alpha, tidx):
        """Creates a label colormap used in PASCAL VOC segmentation benchmark.
        Returns:
        A Colormap for visualizing segmentation results.
        """
        colormap = np.zeros((256, 4), dtype=int)
        indices = np.arange(256, dtype=int)

        for shift in reversed(range(8)):
            for channel in range(3):
                colormap[:, channel] |= ((indices >> channel) & 1) << shift
                indices >>= 3
                
        colormap[:, 3] = alpha
        colormap[tidx, 3] = 0 # force fully transparent for entry tidx
        return colormap.astype(np.uint8)
        
    # ####################################################################################################
    ## JeVois main processing function
    def process(self, inframe, outframe):
        frame = inframe.getCvRGB() if self.rgb else inframe.getCvBGR()
        self.timer.start()
        
        h = frame.shape[0]
        w = frame.shape[1]

        # Set the input:
        width, height = common.input_size(self.interpreter)
        img = Image.fromarray(frame);
        if self.keepaspect:
            resized_img, _ = common.set_resized_input(self.interpreter, img.size,
                                                      lambda size: img.resize(size, Image.LANCSOZ))
        else:
            resized_img = img.resize((width, height), Image.LANCSOZ)
            common.set_input(self.interpreter, resized_img)
  
        # Run the model
        start = time.perf_counter()
        self.interpreter.invoke()
        inference_time = time.perf_counter() - start
        
        # Draw segmentation results:
        result = segment.get_output(self.interpreter)
        if len(result.shape) == 3: result = np.argmax(result, axis=-1)

        # If keep_aspect_ratio, we need to remove the padding area.
        new_width, new_height = resized_img.size
        result = result[:new_height, :new_width]
        mask_img = Image.fromarray(self.cmapRGB[result])

        # Concat resized input image and processed segmentation results.
        output_img = Image.new('RGB', (2 * img.width, img.height))
        output_img.paste(img, (0, 0))
        output_img.paste(mask_img.resize(img.size), (img.width, 0))

        # Back to opencv:
        outcv = np.array(output_img)
        
        # Put efficiency information.
        cv.putText(outcv, 'JeVois Coral Segmentation - ' + self.model, (3, 15),
                   cv.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv.LINE_AA)

        fps = self.timer.stop()
        label = fps + ', %dms' % (inference_time * 1000.0)
        cv.putText(outcv, label, (3, h-5), cv.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv.LINE_AA)
        
        # Send output frame to host:
        if self.rgb: outframe.sendCvRGB(outcv)
        else: outframe.sendCv(outcv)
    
    # ###################################################################################################
    ## Process function with GUI output
    def processGUI(self, inframe, helper):
        # Start a new display frame, gets its size and also whether mouse/keyboard are idle:
        idle, winw, winh = helper.startFrame()

        # Draw full-resolution input frame from camera:
        x, y, w, h = helper.drawInputFrame("c", inframe, False, False)
        
        # Get the next camera image at processing resolution (may block until it is captured):
        frame = inframe.getCvRGBp() if self.rgb else inframe.getCvBGRp()
        iw, ih = frame.shape[1], frame.shape[0]
        
        # Start measuring image processing time:
        self.timer.start()

        # Set the input:
        width, height = common.input_size(self.interpreter)
        img = Image.fromarray(frame);
        if self.keepaspect:
            resized_img, _ = common.set_resized_input(self.interpreter, img.size,
                                                      lambda size: img.resize(size, Image.LANCSOZ))
        else:
            resized_img = img.resize((width, height), Image.LANCSOZ)
            common.set_input(self.interpreter, resized_img)
  
        # Run the model:
        start = time.perf_counter()
        self.interpreter.invoke()
        inference_time = time.perf_counter() - start
        
        # Draw segmentation results:
        result = segment.get_output(self.interpreter)
        if len(result.shape) == 3: result = np.argmax(result, axis=-1)

        # If keep_aspect_ratio, we need to remove the padding area:
        new_width, new_height = resized_img.size
        result = result[:new_height, :new_width]
        mask = self.cmapRGBA[result]

        # Draw the mask on top of our image, OpenGL will do the alpha blending:
        helper.drawImage("m", mask, self.rgb, False, True)

        # Put efficiency information:
        helper.itext('JeVois-Pro Python Coral Segmentation - %s - %dms/inference' %
                     (self.model, inference_time * 1000.0))
        
        # Write frames/s info from our timer:
        fps = self.timer.stop()
        helper.iinfo(inframe, fps, winw, winh);

        # End of frame:
        helper.endFrame()
   
