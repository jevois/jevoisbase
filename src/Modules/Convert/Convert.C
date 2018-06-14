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
    output pixel format (YUYV, GREY, MJPG, BAYER, RGB565, BGR24). 

    See \ref PixelFormats for information about supported pixel formats.

    This module accepts any resolution supported by the JeVois camera sensor:
    
    - SXGA (1280 x 1024): up to 15 fps
    - VGA (640 x 480): up to 30 fps
    - CIF (352 x 288): up to 60 fps
    - QVGA (320 x 240): up to 60 fps
    - QCIF (176 x 144): up to 120 fps
    - QQVGA (160 x 120): up to 60 fps
    - QQCIF (88 x 72): up to 120 fps

    This module also automatically rescales the image if input and output sizes differ.

    Things to try
    -------------

    Edit <b>JEVOIS:/config/videomappings.cfg</b> on your MicroSD card (see \ref VideoMapping) and try to add some new
    convert mappings. Not all of the thousands of possible convert mappings have been included in the card to avoid
    having too many of these simple conversion mappings in the base software distribution. For example,

    \verbatim
    YUYV 176 144 115.0 BAYER 176 144 115.0 JeVois Convert
    \endverbatim
    
    will grab raw BAYER frames on the sensor, with resolution 176x144 at 115 frames/s, and will convert them to YUYV
    before sending them over the USB link. To test this mapping, select the corresponding resolution and framerate in
    your video viewing software (here, YUYV 176x144 \@ 115fps). Although the sensor can capture at up to 120fps at this
    resolution, here we used 115fps to avoid a conflict with a mapping using YUYV 176x144 \@ 120fps USB output and the
    SaveVideo module that is already in the default \b videomappings.cfg file.

    Note that this module may suffer from DMA coherency artifacts if the \p camturbo parameter of the jevois::Engine is
    turned on, which it is by default. The \p camturbo parameter relaxes some of the cache coherency constraints on the
    video buffers captured by the camera sensor, which allows the JeVois processor to access video pixel data from
    memory faster. But with modules that do not do much processing, sometimes this yields video artifacts, we presume
    because some of the video data from previous frames still is in the CPU cache and hence is not again fetched from
    main memory by the CPU. If you see short stripes of what appears to be wrong pixel colors in the video, try to
    disable \p camturbo by editing <b>JEVOIS:/config/params.cfg</b> on your MicroSD card and in there turning \p
    camturbo to false.


    @author Laurent Itti

    @videomapping BAYER 640 480 26.8 YUYV 640 480 26.8 JeVois Convert
    @videomapping BGR24 640 480 26.8 YUYV 640 480 26.8 JeVois Convert
    @videomapping GREY 640 480 26.8 YUYV 640 480 26.8 JeVois Convert
    @videomapping RGB565 640 480 26.8 YUYV 640 480 26.8 JeVois Convert
    @videomapping MJPG 352 288 60.0 BAYER 352 288 60.0 JeVois Convert
    @videomapping MJPG 320 240 30.0 RGB565 320 240 30.0 JeVois Convert
    @videomapping MJPG 320 240 15.0 YUYV 320 240 15.0 JeVois Convert
    @videomapping YUYV 640 480 20.0 YUYV 640 480 20.0 JeVois Convert
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
      // Wait for next available camera image, convert it to OpenCV BGR:
      cv::Mat imgbgr = inframe.getCvBGR();
      
      // Send the output image to the host over USB, after possible format conversion and size scaling:
      outframe.sendCv(imgbgr, quality::get());
    }
};

// Allow the module to be loaded as a shared object (.so) file:
JEVOIS_REGISTER_MODULE(Convert);
