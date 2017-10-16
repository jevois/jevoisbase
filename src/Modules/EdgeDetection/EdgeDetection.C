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
#include <jevois/Image/RawImageOps.H>

#include <linux/videodev2.h>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

// icon by Sergiu Bagrin in interface at flaticon

// Parameters for our module:
static jevois::ParameterCategory const ParamCateg("Edge Detection Options");

//! Parameter \relates EdgeDetection
JEVOIS_DECLARE_PARAMETER(thresh1, double, "First threshold for hysteresis", 50.0, ParamCateg);
//! Parameter \relates EdgeDetection
JEVOIS_DECLARE_PARAMETER(thresh2, double, "Second threshold for hysteresis", 150.0, ParamCateg);
//! Parameter \relates EdgeDetection
JEVOIS_DECLARE_PARAMETER(aperture, int, "Aperture size for the Sobel operator", 3, ParamCateg);
//! Parameter \relates EdgeDetection
JEVOIS_DECLARE_PARAMETER(l2grad, bool, "Use more accurate L2 gradient norm if true, L1 if false", false, ParamCateg);

//! Simple module to detect edges using the Canny algorithm from OpenCV
/*! Compute edges in an image using the Canny edge detection algorithm.

    This module is intended as a pre-processor, delivering edge maps to a host computer, which may then be in charge of
    further processing them, for example to detect objects of various shapes.
    
    You should be able to run this module at 60 frames/s with resolution 320x240 on the JeVois camera.

    @author Laurent Itti

    @videomapping GREY 640 480 29.0 YUYV 640 480 29.0 JeVois EdgeDetection
    @videomapping GREY 320 240 59.0 YUYV 320 240 59.0 JeVois EdgeDetection
    @videomapping GREY 176 144 119.0 YUYV 176 144 119.0 JeVois EdgeDetection
    @email itti\@usc.edu
    @address University of Southern California, HNB-07A, 3641 Watt Way, Los Angeles, CA 90089-2520, USA
    @copyright Copyright (C) 2016 by Laurent Itti, iLab and the University of Southern California
    @mainurl http://jevois.org
    @supporturl http://jevois.org/doc
    @otherurl http://iLab.usc.edu
    @license GPL v3
    @distribution Unrestricted
    @restrictions None
    \ingroup modules */
class EdgeDetection : public jevois::Module,
                      public jevois::Parameter<thresh1, thresh2, aperture, l2grad>
{
  public:
    //! Default base class constructor ok
    using jevois::Module::Module;

    //! Virtual destructor for safe inheritance
    virtual ~EdgeDetection() { }

    //! Processing function
    virtual void process(jevois::InputFrame && inframe, jevois::OutputFrame && outframe) override
    {
      // Wait for next available camera image:
      jevois::RawImage inimg = inframe.get();

      // Convert to OpenCV grayscale:
      cv::Mat grayimg = jevois::rawimage::convertToCvGray(inimg);
  
      // Let camera know we are done processing the input image:
      inframe.done();

      // Wait for an image from our gadget driver into which we will put our results. Require that it must have same
      // image size as the input image, and greyscale pixels:
      jevois::RawImage outimg = outframe.get();
      outimg.require("output", inimg.width, inimg.height, V4L2_PIX_FMT_GREY);

      // Compute Canny edges directly into the output image:
      cv::Mat edges = jevois::rawimage::cvImage(outimg); // Pixel data of "edges" shared with "outimg", no copy
      cv::Canny(grayimg, edges, thresh1::get(), thresh2::get(), aperture::get(), l2grad::get());
      
      // Send the output image with our processing results to the host over USB:
      outframe.send();
    }
};

// Allow the module to be loaded as a shared object (.so) file:
JEVOIS_REGISTER_MODULE(EdgeDetection);
