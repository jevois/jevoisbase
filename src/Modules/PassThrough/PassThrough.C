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

// icon by Catalin Fertu in cinema at flaticon

//! Simple module that just passes the captured camera frames through to USB host
/*! This module makes your JeVois smart camera operate like a regular "dumb" camera. It is intended mainly for use in
    programming tutorials, and to allow you to debug new machine vision modules that you test on your host computer,
    using the JeVois camera in PassThrough mode as input, to simulate what will happen when your code runs on the JeVois
    embedded processor.

    Any video mapping is possible here, as long as camera and USB pixel types match, and camera and USB image dimensions
    also match.

    @author Laurent Itti

    @videomapping YUYV 1280 1024 7.5 YUYV 1280 1024 7.5 JeVois PassThrough
    @videomapping YUYV 640 480 30.0 YUYV 640 480 30.0 JeVois SaveVideo
    @videomapping YUYV 640 480 19.6 YUYV 640 480 19.6 JeVois PassThrough
    @videomapping YUYV 640 480 12.0 YUYV 640 480 12.0 JeVois PassThrough
    @videomapping YUYV 640 480 8.3 YUYV 640 480 8.3 JeVois PassThrough
    @videomapping YUYV 640 480 7.5 YUYV 640 480 7.5 JeVois PassThrough
    @videomapping YUYV 640 480 5.5 YUYV 640 480 5.5 JeVois PassThrough
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
class PassThrough : public jevois::Module
{
  public:
    //! Default base class constructor ok
    using jevois::Module::Module;

    //! Virtual destructor for safe inheritance
    virtual ~PassThrough() { }

    //! Processing function
    virtual void process(jevois::InputFrame && inframe, jevois::OutputFrame && outframe) override
    {
      // Wait for next available camera image:
      jevois::RawImage const inimg = inframe.get(true);
      
      // Wait for an image from our gadget driver into which we will put our results:
      jevois::RawImage outimg = outframe.get();

      // Enforce that the input and output formats and image sizes match:
      outimg.require("output", inimg.width, inimg.height, inimg.fmt);
      
      // Just copy the pixel data over:
      memcpy(outimg.pixelsw<void>(), inimg.pixels<void>(), std::min(inimg.buf->length(), outimg.buf->length()));

      // Camera outputs RGB565 in big-endian, but most video grabbers expect little-endian:
      if (outimg.fmt == V4L2_PIX_FMT_RGB565) jevois::rawimage::byteSwap(outimg);
      
      // Let camera know we are done processing the input image:
      inframe.done(); // NOTE: optional here, inframe destructor would call it anyway

      // Send the output image with our processing results to the host over USB:
      outframe.send(); // NOTE: optional here, outframe destructor would call it anyway
    }
};

// Allow the module to be loaded as a shared object (.so) file:
JEVOIS_REGISTER_MODULE(PassThrough);
