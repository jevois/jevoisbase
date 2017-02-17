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
#include <jevois/Debug/Log.H>
#include <jevois/Util/Utils.H>
#include <jevois/Image/RawImageOps.H>

#include <linux/videodev2.h>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <string.h>

// icon by Pixel Buddha in arrows at flaticon

static jevois::ParameterCategory const ParamCateg("Convert Options");

//! Parameter \relates Convert
JEVOIS_DECLARE_PARAMETER(quality, int, "Compression quality for MJPEG", 75, jevois::Range<int>(1, 100), ParamCateg);

//! Simple module to convert between any supported camera grab formats and USB output formats
/*! This module can convert from any supported camera sensor pixel format (YUYV, BAYER, RGB565) to any supported USB
    output pixel format (YUYV, GREY, MJPG, BAYER, RGB565, BGR24). Note that it only converts pixel type, and is not
    capable of resizing the image. Thus, input and output image dimensions must match.

    @author Laurent Itti

    @videomapping BAYER 640 480 26.8 YUYV 640 480 26.8 JeVois Convert
    @videomapping BGR24 640 480 26.8 YUYV 640 480 26.8 JeVois Convert
    @videomapping GREY 640 480 26.8 YUYV 640 480 26.8 JeVois Convert
    @videomapping RGB565 640 480 26.8 YUYV 640 480 26.8 JeVois Convert
    @videomapping MJPG 352 288 60.0 BAYER 352 288 60.0 JeVois Convert
    @videomapping MJPG 320 240 30.0 RGB565 320 240 30.0 JeVois Convert
    @videomapping MJPG 320 240 15.0 YUYV 320 240 15.0 JeVois Convert
    @videomapping YUYV 640 500 20.0 YUYV 640 480 20.0 JeVois DemoArUco
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
class Convert : public jevois::Module, public jevois::Parameter<quality>
{
  public:
    //! Default base class constructor ok
    using jevois::Module::Module;

    //! Virtual destructor for safe inheritance
    virtual ~Convert() { }

    //! Processing function
    virtual void process(jevois::InputFrame && inframe, jevois::OutputFrame && outframe) override
    {
      // Wait for next available camera image:
      jevois::RawImage inimg = inframe.get(true);
      unsigned int const w = inimg.width, h = inimg.height;

      // Convert it to BGR24:
      cv::Mat imgbgr = jevois::rawimage::convertToCvBGR(inimg);
  
      // Let camera know we are done processing the input image:
      inframe.done();
      
      // Wait for an image from our gadget driver into which we will put our results:
      jevois::RawImage outimg = outframe.get();

      // Require that output has same dims as input, allow any output format:
      outimg.require("output", w, h, outimg.fmt);

      // Convert from BGR to desired output format:
      jevois::rawimage::convertCvBGRtoRawImage(imgbgr, outimg, quality::get());
      
      // Send the output image with our processing results to the host over USB:
      outframe.send();
    }
};

// Allow the module to be loaded as a shared object (.so) file:
JEVOIS_REGISTER_MODULE(Convert);
