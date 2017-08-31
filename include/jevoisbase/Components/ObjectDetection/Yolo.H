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

#include <nnpack.h>

extern "C" {
#include <darknet.h>
}

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
  JEVOIS_DECLARE_PARAMETER(nms, float, "Non-maximum suppression",
                           0.4F, ParamCateg);

  //! Parameter \relates Yolo
  JEVOIS_DECLARE_PARAMETER(thresh, float, "Detection threshold",
                           0.24F, ParamCateg);

  //! Parameter \relates Yolo
  JEVOIS_DECLARE_PARAMETER(hierthresh, float, "Hierarchical detection threshold",
                           0.5F, ParamCateg);
}

//! Detect multiple objects in scenes using the Darknet YOLO deep neural network
/*! 
    \ingroup Components */
class Yolo : public jevois::Component,
             jevois::Parameter<yolo::dataroot, yolo::datacfg, yolo::cfgfile, yolo::weightfile, yolo::namefile,
                               yolo::nms, yolo::thresh, yolo::hierthresh>
{
  public: 
    //! Inherited constructor ok
    using jevois::Component::Component;

    //! Initialize, configure and load the network in a thread
    /*! Any call to process() will simply throw until the network is loaded and ready */
    void postInit() override;

    //! Virtual destructor for safe inheritance
    virtual ~Yolo();

    //! Un-initialize and free resources
    void postUninit() override;

    //! Processing function, results are stored internally in the underlying Darknet network object
    /*! This version expects an OpenCV RGB byte image which will be converted to float RGB planar, and which may be
        letterboxed if necessary to fit network input dims. */
    void predict(cv::Mat const & cvimg);
    
    //! Processing function, results are stored internally in the underlying Darknet network object
    /*! This version expects a Darknet image input, RGB float planar normalized to [0..1], with same dims as network
        input dims. */
    void predict(image & im);

    void computeBoxes(jevois::RawImage & im);

    //void sendSerial();

    void drawDetections(jevois::RawImage & im);

  protected:
    network net;
    char ** names = nullptr;
    box * boxes = nullptr;
    float ** probs = nullptr;
 };
