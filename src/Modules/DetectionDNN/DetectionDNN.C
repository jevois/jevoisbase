// ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//
// JeVois Smart Embedded Machine Vision Toolkit - Copyright (C) 2016 by Laurent Itti, the University of Southern
// California (USC), and iLab at USC. See http://iLab.usc.edu and http://jevois.org for information about this project.
//
// This file is part of the JeVois Smart Embedded Machine Vision Toolkit.  This program is free software; you can
// redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software
// Foundation, version 2.  This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY;
// without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public
// License for more details.  You should have received a copy of the GNU General Public License along with this program;
// if not, write to the Free Software Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301, USA.
//
// Contact information: Laurent Itti - 3641 Watt Way, HNB-07A - Los Angeles, CA 90089-2520 - USA.
// Tel: +1 213 740 3527 - itti@pollux.usc.edu - http://iLab.usc.edu - http://jevois.org
// ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/*! \file */

#include <jevois/Core/Module.H>
#include <jevois/Debug/Timer.H>
#include <jevois/Image/RawImageOps.H>
#include <opencv2/core/core.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <fstream>

// icon from opencv

static jevois::ParameterCategory const ParamCateg("Darknet YOLO Options");

//! Parameter \relates DetectionDNN
JEVOIS_DECLARE_PARAMETER(classnames, std::string, "Path to a text file with names of classes to label detected objects",
                         "/jevois/share/opencv-dnn/detection/opencv_face_detector.classes", ParamCateg);

//! Parameter \relates DetectionDNN
JEVOIS_DECLARE_PARAMETER(configname, std::string, "Path to a text file that contains network configuration. "
                         "Can have extension .prototxt (Caffe), .pbtxt (TensorFlow), or .cfg (Darknet).",
                         "/jevois/share/opencv-dnn/detection/opencv_face_detector.prototxt", ParamCateg);

//! Parameter \relates DetectionDNN
JEVOIS_DECLARE_PARAMETER(modelname, std::string, "Path to a binary file of model contains trained weights. "
                         "Can have extension .caffemodel (Caffe), .pb (TensorFlow), .t7 or .net (Torch), "
                         "or .weights (Darknet).",
                         "/jevois/share/opencv-dnn/detection/opencv_face_detector.caffemodel", ParamCateg);

//! Parameter \relates DetectionDNN
JEVOIS_DECLARE_PARAMETER(netin, cv::Size, "Width and height (in pixels) of the neural network input layer, or [0 0] "
                         "to make it match camera frame size. NOTE: for YOLO v3 sizes must be multiples of 32.",
                         cv::Size(160, 120), ParamCateg);

//! Parameter \relates DetectionDNN
JEVOIS_DECLARE_PARAMETER(thresh, float, "Detection threshold in percent confidence",
                         50.0F, jevois::Range<float>(0.0F, 100.0F), ParamCateg);

//! Parameter \relates DetectionDNN
JEVOIS_DECLARE_PARAMETER(nms, float, "Non-maximum suppression intersection-over-union threshold in percent",
                         45.0F, jevois::Range<float>(0.0F, 100.0F), ParamCateg);

//! Parameter \relates DetectionDNN
JEVOIS_DECLARE_PARAMETER(rgb, bool, "When true, model works with RGB input images instead BGR ones",
                         true, ParamCateg);

//! Parameter \relates DetectionDNN
JEVOIS_DECLARE_PARAMETER(scale, float, "Value scaling factor applied to input pixels",
                         2.0F / 255.0F, ParamCateg);

//! Parameter \relates DetectionDNN
JEVOIS_DECLARE_PARAMETER(mean, cv::Scalar, "Mean BGR value subtracted from input image",
                         cv::Scalar(127.5F, 127.5F, 127.5F), ParamCateg);

//! Detect and recognize multiple objects in scenes using OpenCV Deep Neural Nets (DNN)
/*! This module runs an object detection deep neural network using the OpenCV DNN library. Detection networks analyze a
    whole scene and produce a number of bounding boxes around detected objects, together with identity labels and
    confidence scores for each detected box.

    This module runs the selected deep neural network and shows all detections obtained.

    Note that by default this module runs the OpenCV Face Detector DNN which can detect human faces.

    Included with the standard JeVois distribution are the following networks:

    - OpenCV Face Detector, Caffe model
    - MobileNet + SSD trained on Pascal VOC (20 object classes), Caffe model
    - MobileNet + SSD trained on Coco (80 object classes), TensorFlow model
    - MobileNet v2 + SSD trained on Coco (80 object classes), TensorFlow model
    - Darknet Tiny YOLO v3 trained on Coco (80 object classes), Darknet model
    - Darknet Tiny YOLO v2 trained on Pascal VOC (20 object classes), Darknet model

    See the module's \b params.cfg file to switch network. Object categories are as follows:
    
    - The 80 COCO object categories are: person, bicycle, car, motorbike, aeroplane, bus, train, truck, boat, traffic,
      fire, stop, parking, bench, bird, cat, dog, horse, sheep, cow, elephant, bear, zebra, giraffe, backpack, umbrella,
      handbag, tie, suitcase, frisbee, skis, snowboard, sports, kite, baseball, baseball, skateboard, surfboard, tennis,
      bottle, wine, cup, fork, knife, spoon, bowl, banana, apple, sandwich, orange, broccoli, carrot, hot, pizza, donut,
      cake, chair, sofa, pottedplant, bed, diningtable, toilet, tvmonitor, laptop, mouse, remote, keyboard, cell,
      microwave, oven, toaster, sink, refrigerator, book, clock, vase, scissors, teddy, hair, toothbrush.

    - The 20 Pascal-VOC object categories are: aeroplane, bicycle, bird, boat, bottle, bus, car, cat, chair, cow,
      diningtable, dog, horse, motorbike, person, pottedplant, sheep, sofa, train, tvmonitor.

    Sometimes it will make mistakes! The performance of yolov3-tiny is about 33.1% correct (mean average precision) on
    the COCO test set. The OpenCV Face Detector is quite fast and robust!

    Speed and network size
    ----------------------

    The parameter \p netin allows you to rescale the neural network to the specified size. Beware that this will only
    work if the network used is fully convolutional (as is the case with the default networks listed above). This not
    only allows you to adjust processing speed (and, conversely, accuracy), but also to better match the network to the
    input images (e.g., the default size for tiny-yolo is 416x416, and, thus, passing it a input image of size 640x480
    will result in first scaling that input to 416x312, then letterboxing it by adding gray borders on top and bottom so
    that the final input to the network is 416x416). This letterboxing can be completely avoided by just resizing the
    network to 320x240.

    Here are expected processing speeds for the OpenCV Face Detector:
    - when netin = [320 240], processes 320x240 inputs, about 650ms/image (1.5 frames/s)
    - when netin = [160 120], processes 160x120 inputs, about 190ms/image (5.0 frames/s)

    Serial messages
    ---------------

    When detections are found which are above threshold, one message will be sent for each detected
    object (i.e., for each box that gets drawn when USB outputs are used), using a standardized 2D message:
    + Serial message type: \b 2D
    + `id`: the category of the recognized object, followed by ':' and the confidence score in percent
    + `x`, `y`, or vertices: standardized 2D coordinates of object center or corners
    + `w`, `h`: standardized object size
    + `extra`: any number of additional category:score pairs which had an above-threshold score for that box
    
    See \ref UserSerialStyle for more on standardized serial messages, and \ref coordhelpers for more info on
    standardized coordinates.

    This code is heavily inspired from:
    https://github.com/opencv/opencv/blob/master/samples/dnn/object_detection.cpp

    @author Laurent Itti

    @displayname Detection DNN
    @videomapping NONE 0 0 0.0 YUYV 640 480 15.0 JeVois DetectionDNN
    @videomapping YUYV 640 498 15.0 YUYV 640 480 15.0 JeVois DetectionDNN
    @email itti\@usc.edu
    @address University of Southern California, HNB-07A, 3641 Watt Way, Los Angeles, CA 90089-2520, USA
    @copyright Copyright (C) 2018 by Laurent Itti, iLab and the University of Southern California
    @mainurl http://jevois.org
    @supporturl http://jevois.org/doc
    @otherurl http://iLab.usc.edu
    @license GPL v3
    @distribution Unrestricted
    @restrictions None
    \ingroup modules */
class DetectionDNN : public jevois::StdModule,
                     public jevois::Parameter<classnames, configname, modelname, netin, thresh, nms, rgb, scale, mean>
{
  public: 
    // ####################################################################################################
    //! Constructor
    // ####################################################################################################
    DetectionDNN(std::string const & instance) : jevois::StdModule(instance)
    { }

    // ####################################################################################################
    //! Virtual destructor for safe inheritance
    // ####################################################################################################
    virtual ~DetectionDNN()
    { }

    // ####################################################################################################
    //! Initialization
    // ####################################################################################################
    virtual void postInit() override
    {
      // Load the class names:
      std::ifstream ifs(classnames::get());
      if (ifs.is_open() == false) LFATAL("Class names file " << classnames::get() << " not found");
      std::string line;
      while (std::getline(ifs, line)) itsClasses.push_back(line);

      // Create and load the network:
      itsNet = cv::dnn::readNet(modelname::get(), configname::get());
      itsNet.setPreferableBackend(cv::dnn::DNN_BACKEND_DEFAULT);
      itsNet.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);

      // Get names of the network's output layers:
      itsOutLayers = itsNet.getUnconnectedOutLayers();
      std::vector<cv::String> layersNames = itsNet.getLayerNames();
      itsOutNames.resize(itsOutLayers.size());
      for (size_t i = 0; i < itsOutLayers.size(); ++i) itsOutNames[i] = layersNames[itsOutLayers[i] - 1];
      itsOutLayerType = itsNet.getLayer(itsOutLayers[0])->type;
    }

    // ####################################################################################################
    //! Un-initialization
    // ####################################################################################################
    virtual void postUninit() override
    { }

    // ####################################################################################################
    //! Post-processing to extract boxes from network outputs
    // ####################################################################################################
    void postprocess(cv::Mat const & frame, std::vector<cv::Mat> const & outs, jevois::RawImage * outframe = nullptr)
    {
      float const confThreshold = thresh::get() * 0.01F;
      float const nmsThreshold = nms::get() * 0.01F;

      std::vector<int> classIds;
      std::vector<float> confidences;
      std::vector<cv::Rect> boxes;
      if (itsNet.getLayer(0)->outputNameToIndex("im_info") != -1)  // Faster-RCNN or R-FCN
      {
        // Network produces output blob with a shape 1x1xNx7 where N is a number of detections and an every detection is
        // a vector of values [batchId, classId, confidence, left, top, right, bottom]
        if (outs.size() != 1) LFATAL("Malformed output layers");
        float* data = (float*)outs[0].data;
        for (size_t i = 0; i < outs[0].total(); i += 7)
        {
          float confidence = data[i + 2];
          if (confidence > confThreshold)
          {
            int left = (int)data[i + 3];
            int top = (int)data[i + 4];
            int right = (int)data[i + 5];
            int bottom = (int)data[i + 6];
            int width = right - left + 1;
            int height = bottom - top + 1;
            classIds.push_back((int)(data[i + 1]) - 1);  // Skip 0th background class id.
            boxes.push_back(cv::Rect(left, top, width, height));
            confidences.push_back(confidence);
          }
        }
      }
      else if (itsOutLayerType == "DetectionOutput")
      {
        // Network produces output blob with a shape 1x1xNx7 where N is a number of detections and an every detection is
        // a vector of values [batchId, classId, confidence, left, top, right, bottom]
        if (outs.size() != 1) LFATAL("Malformed output layers");
        float* data = (float*)outs[0].data;
        for (size_t i = 0; i < outs[0].total(); i += 7)
        {
          float confidence = data[i + 2];
          if (confidence > confThreshold)
          {
            int left = (int)(data[i + 3] * frame.cols);
            int top = (int)(data[i + 4] * frame.rows);
            int right = (int)(data[i + 5] * frame.cols);
            int bottom = (int)(data[i + 6] * frame.rows);
            int width = right - left + 1;
            int height = bottom - top + 1;
            classIds.push_back((int)(data[i + 1]) - 1);  // Skip 0th background class id.
            boxes.push_back(cv::Rect(left, top, width, height));
            confidences.push_back(confidence);
          }
        }
      }
      else if (itsOutLayerType == "Region")
      {
        for (size_t i = 0; i < outs.size(); ++i)
        {
          // Network produces output blob with a shape NxC where N is a number of detected objects and C is a number of
          // classes + 4 where the first 4 numbers are [center_x, center_y, width, height]
          float* data = (float*)outs[i].data;
          for (int j = 0; j < outs[i].rows; ++j, data += outs[i].cols)
          {
            cv::Mat scores = outs[i].row(j).colRange(5, outs[i].cols);
            cv::Point classIdPoint;
            double confidence;
            minMaxLoc(scores, 0, &confidence, 0, &classIdPoint);
            if (confidence > confThreshold)
            {
              int centerX = (int)(data[0] * frame.cols);
              int centerY = (int)(data[1] * frame.rows);
              int width = (int)(data[2] * frame.cols);
              int height = (int)(data[3] * frame.rows);
              int left = centerX - width / 2;
              int top = centerY - height / 2;
              
              classIds.push_back(classIdPoint.x);
              confidences.push_back((float)confidence);
              boxes.push_back(cv::Rect(left, top, width, height));
            }
          }
        }
      }
      else LFATAL("Unknown output layer type: " << itsOutLayerType);

      // Cleanup overlapping boxes:
      std::vector<int> indices;
      cv::dnn::NMSBoxes(boxes, confidences, confThreshold, nmsThreshold, indices);

      // Send serial messages and draw boxes:
      for (size_t i = 0; i < indices.size(); ++i)
      {
        int idx = indices[i];
        cv::Rect const & box = boxes[idx];
        std::vector<jevois::ObjReco> data;
        float const conf = confidences[idx] * 100.0F;
        std::string name;
        if (classIds[idx] < itsClasses.size()) name = itsClasses[classIds[idx]]; else name = "Oooops";
        data.push_back({ conf, name });
        
        std::string label = jevois::sformat("%s: %.2f", name.c_str(), conf);

        if (outframe)
        {
          jevois::rawimage::drawRect(*outframe, box.x, box.y, box.width, box.height, 2, jevois::yuyv::LightGreen);
          jevois::rawimage::writeText(*outframe, label, box.x + 6, box.y + 2, jevois::yuyv::LightGreen,
                                      jevois::rawimage::Font10x20);
        }

        sendSerialObjDetImg2D(frame.cols, frame.rows, box.x, box.y, box.width, box.height, data);
      }
    }
    
    // ####################################################################################################
    //! Processing function, no video output
    // ####################################################################################################
    virtual void process(jevois::InputFrame && inframe) override
    {
      // Wait for next available camera image:
      jevois::RawImage const inimg = inframe.get();
      unsigned int const w = inimg.width, h = inimg.height;

      // Convert input image to BGR for predictions:
      cv::Mat cvimg = jevois::rawimage::convertToCvBGR(inimg);

      // Let camera know we are done processing the input image:
      inframe.done();

      // Extract blob that will be sent to network:
      cv::Mat blob;
      cv::dnn::blobFromImage(cvimg, blob, scale::get(), netin::get(), mean::get(), rgb::get(), false);
      
      // Launch the predictions:
      itsNet.setInput(blob);
      std::vector<cv::Mat> outs; itsNet.forward(outs, itsOutNames);

      // Post-process the outputs and send serial messages:
      postprocess(cvimg, outs);
    }
    
    // ####################################################################################################
    //! Processing function with video output to USB
    // ####################################################################################################
    virtual void process(jevois::InputFrame && inframe, jevois::OutputFrame && outframe) override
    {
      static jevois::Timer timer("processing", 10, LOG_DEBUG);

      // Wait for next available camera image:
      jevois::RawImage const inimg = inframe.get();

      timer.start();
      
      // We only handle one specific pixel format, and any image size in this module:
      unsigned int const w = inimg.width, h = inimg.height;
      inimg.require("input", w, h, V4L2_PIX_FMT_YUYV);

      // While we process it, start a thread to wait for out frame and paste the input into it:
      jevois::RawImage outimg;
      auto paste_fut = std::async(std::launch::async, [&]() {
          outimg = outframe.get();
          outimg.require("output", w, h + 18, inimg.fmt);
          
          // Paste the current input image:
          jevois::rawimage::paste(inimg, outimg, 0, 0);
          jevois::rawimage::writeText(outimg, "JeVois ObjectDetection DNN", 3, 3, jevois::yuyv::White);
          jevois::rawimage::drawFilledRect(outimg, 0, h, w, outimg.height - h, jevois::yuyv::Black);
        });

      // Convert input image to BGR for predictions:
      cv::Mat cvimg = jevois::rawimage::convertToCvBGR(inimg);

      // Extract blob that will be sent to network:
      cv::Mat blob;
      cv::dnn::blobFromImage(cvimg, blob, scale::get(), netin::get(), mean::get(), rgb::get(), false);
      
      // Let camera know we are done processing the input image:
      inframe.done();

      // Launch the predictions:
      itsNet.setInput(blob);
      std::vector<cv::Mat> outs; itsNet.forward(outs, itsOutNames);

      // Wait for paste to finish up:
      paste_fut.get();

      // Post-process the outputs, draw them. and send serial messages:
      postprocess(cvimg, outs, &outimg);

      // Display efficiency information:
      std::vector<double> layersTimes;
      double freq = cv::getTickFrequency() / 1000;
      double t = itsNet.getPerfProfile(layersTimes) / freq;
      std::string label = jevois::sformat("Inference time: %.2f ms", t);
      jevois::rawimage::writeText(outimg, label, 3, h + 3, jevois::yuyv::White);
      
      // Show processing fps:
      std::string const & fpscpu = timer.stop();
      jevois::rawimage::writeText(outimg, fpscpu, 3, h - 13, jevois::yuyv::White);
      
      // Send the output image with our processing results to the host over USB:
      outframe.send();
    }

    // ####################################################################################################
  protected:
    std::vector<std::string> itsClasses;
    cv::dnn::Net itsNet;
    std::vector<cv::String> itsOutNames;
    std::vector<int> itsOutLayers;
    std::string itsOutLayerType;
};

// Allow the module to be loaded as a shared object (.so) file:
JEVOIS_REGISTER_MODULE(DetectionDNN);
