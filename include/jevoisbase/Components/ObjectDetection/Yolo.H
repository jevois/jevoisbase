// ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//
// JeVois Smart Embedded Machine Vision Toolkit - Copyright (C) 2017 by Laurent Itti, the University of Southern
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

#pragma once

#include <jevois/Component/Component.H>
#include <jevois/Image/RawImageOps.H>
#include <atomic>

#include <nnpack.h>

extern "C" {
#include <darknet.h>
}

namespace jevois { class StdModule; }

namespace yolo
{
  static jevois::ParameterCategory const ParamCateg("Darknet YOLO Options");

  //! Parameter \relates Yolo
  JEVOIS_DECLARE_PARAMETER(dataroot, std::string, "Root path for data, config, and weight files. If empty, use "
                           "the module's path.",
                           JEVOIS_SHARE_PATH "/darknet/yolo", ParamCateg);

  
  //! Parameter \relates Yolo
  JEVOIS_DECLARE_PARAMETER(datacfg, std::string, "Data configuration file (if relative, relative to dataroot)",
                           "cfg/voc.data", ParamCateg);

  //! Parameter \relates Yolo
  JEVOIS_DECLARE_PARAMETER(cfgfile, std::string, "Network configuration file (if relative, relative to dataroot)",
                           "cfg/tiny-yolo-voc.cfg", ParamCateg);

  //! Parameter \relates Yolo
  JEVOIS_DECLARE_PARAMETER(weightfile, std::string, "Network weights file (if relative, relative to dataroot)",
                           "weights/tiny-yolo-voc.weights", ParamCateg);

  //! Parameter \relates Yolo
  JEVOIS_DECLARE_PARAMETER(namefile, std::string, "Category names file, or empty to fetch it from the network "
                           "config file (if relative, relative to dataroot)",
                           "", ParamCateg);

  //! Parameter \relates Yolo
  JEVOIS_DECLARE_PARAMETER(nms, float, "Non-maximum suppression in percent confidence",
                           40.0F, jevois::Range<float>(0.0F, 100.0F), ParamCateg);

  //! Parameter \relates Yolo
  JEVOIS_DECLARE_PARAMETER(thresh, float, "Detection threshold in percent confidence",
                           24.0F, jevois::Range<float>(0.0F, 100.0F), ParamCateg);

  //! Parameter \relates Yolo
  JEVOIS_DECLARE_PARAMETER(hierthresh, float, "Hierarchical detection threshold in percent confidence",
                           50.0F, jevois::Range<float>(0.0F, 100.0F), ParamCateg);

  //! Parameter \relates Yolo
  JEVOIS_DECLARE_PARAMETER(threads, int, "Number of parallel computation threads",
                           6, jevois::Range<int>(1, 1024), ParamCateg);
}

//! Detect multiple objects in scenes using the Darknet YOLO deep neural network
/*! Darknet is a popular neural network framework, and YOLO is a very interesting network that detects all objects in a
    scene in one pass. This component detects all instances of any of the objects it knows about (determined by the
    network structure, labels, dataset used for training, and weights obtained) in the image that is given to is.

    See https://pjreddie.com/darknet/yolo/

    Darknet is a great, bare-metal deep learning and deep neural network framework. It is great for embedded systems
    like the small JeVois camera because it has a very small footprint and fewer dependencies than other deep neural
    network frameworks like Tensorflow, MXNet, Theano, Keras, PyTorch, etc. In addition, the port of Darknet to JeVois
    includes acceleration using the ARM NEON multimedia instructions through the popular NNPACK neural network
    acceleration package.

    \ingroup Components */
class Yolo : public jevois::Component,
             jevois::Parameter<yolo::dataroot, yolo::datacfg, yolo::cfgfile, yolo::weightfile, yolo::namefile,
                               yolo::nms, yolo::thresh, yolo::hierthresh, yolo::threads>
{
  public: 
    //! Constructor
    Yolo(std::string const & instance);

    //! Initialize, configure and load the network in a thread
    /*! Any call to process() will simply throw until the network is loaded and ready */
    void postInit() override;

    //! Virtual destructor for safe inheritance
    virtual ~Yolo();

    //! Un-initialize and free resources
    void postUninit() override;

    //! Processing function, results are stored internally in the underlying Darknet network object
    /*! This version expects an OpenCV RGB byte image which will be converted to float RGB planar, and which may be
        letterboxed if necessary to fit network input dims. Returns the prediction time (neural net forward pass) in
        milliseconds. Throws std::logic_error if the network is still loading and not ready. */
    float predict(cv::Mat const & cvimg);
    
    //! Processing function, results are stored internally in the underlying Darknet network object
    /*! This version expects a Darknet image input, RGB float planar normalized to [0..1], with same dims as network
        input dims. Returns the prediction time (neural net forward pass) in milliseconds. Throws std::logic_error if
        the network is still loading and not ready. */
    float predict(image & im);

    //! Compute the boxes
    /*! You must have called predict() first for this to not violently crash. */
    void computeBoxes(int inw, int inh);

    //! Draw the detections
    /*! You must have called computeBoxes() first for this to not violently crash. */
    void drawDetections(jevois::RawImage & outimg, int inw, int inh, int xoff, int yoff);

    //! Send serial messages about detections
    /*! You must have called computeBoxes() first for this to not violently crash. The module given should be the owner
        of this component, we will use it to actually send each serial message using some variant of
        jevois::Module::sendSerial(). */
    void sendSerial(jevois::StdModule * mod, int inw, int inh, unsigned long frame);

    //! Resize the network's input image dims
    /*! This will prepare the network to receive inputs of the specified size. It is optional and will be called
        automatically by predict() if the given image size does not match the current network input size. Note that this
        only works with fully convolutional networks. Note that the number of channels cannot be changed at this
        time. Throws std::logic_error if the network is still loading and not ready. */
    void resizeInDims(int w, int h);
    
    //! Get input width, height, channels
    /*! Throws std::logic_error if the network is still loading and not ready. */
    void getInDims(int & w, int & h, int & c) const;

    // We leave these in the open in case one wants to access the probs, names, etc but just be careful with them
    network net;
    char ** names = nullptr;
    box * boxes = nullptr;
    float ** probs = nullptr;
    int classes;
    
  protected:
    std::future<void> itsReadyFut;
    std::atomic<bool> itsReady;
 };
