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
#include <jevois/Types/Enum.H>
#include <jevois/Debug/Timer.H>

#include <jevoisbase/Components/FilterGPU/FilterGPU.H>

#include <jevois/Image/RawImageOps.H>
#include <linux/videodev2.h>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

static jevois::ParameterCategory const ParamCateg("DemoGPU Parameters");

// Note: we use NoEffect here as None seems to be defined a s anumeric constant somewhere, when compiling on host

//! Parameter \relates DemoGPU
JEVOIS_DEFINE_ENUM_CLASS(Effect, (NoEffect) (Blur) (Sobel) (Median) (Mult) (Thresh) (Dilate) (Erode) (Twirl) (Dewarp));

//! Parameter \relates DemoGPU
JEVOIS_DECLARE_PARAMETER_WITH_CALLBACK(effect, Effect, "GPU image processing effect to apply",
                                       Effect::Twirl, Effect_Values, ParamCateg);

// icon by by Freepik in technology at flaticon

//! Simple image filtering demo using OpenGL-ES shaders on the Mali-400MP2 GPU
/*! This class first copies the given input image into an OpenGL texture, then applies OpenGL-ES vertex and fragment
    shaders to achieve some image processing, and finally gets the resulting pixels back into an image.
    
    This code is inspired somewhat from the tutorial and code examples found on this page:
    http://robotblogging.blogspot.com/2013/10/gpu-accelerated-camera-processing-on.html

    But an important distinction is that we render to a framebuffer with an RGB565 renderbuffer attached, which
    accelerates processing and transfer of the results a lot.

    Dewarp algorithm contributed by JeVois user [Ali AlSaibie](https://github.com/alsaibie). It is to be used with a
    modified JeVois camera sensor that has a wide-angle lens. See http://jevois.org/qa/index.php?qa=153 for details.

    \youtube{rxsWR3I_LnM}

    Using this module
    -----------------

    Unfortunately, MacOS computers refuse to detect the JeVois smart camera as soon as it exposes one or more RGB565
    video modes. Thus, you cannot use this module with Macs in its native version (but see below), and the module is
    disabled by default. In addition, RGB565 does not seem to work in older \c guvcview either, on Ubuntu prior to
    17.04! Proceed as follows to enable and use this module on a Linux host:

    Edit <b>JEVOIS:/jevois/config/videomappings.cfg</b> and look for the line that mentions \jvmod{DemoGPU}. It is
    commented out, so just remove the leading \b # sign. The line should then look like this:

    \verbatim
    RGB565 320 240 22.0 YUYV 320 240 22.0 JeVois DemoGPU
    \endverbatim

    Restart JeVois and run this on your Linux host if older than Ubuntu 17.04:

    \verbatim
    sudo apt install ffmpeg
    ffplay /dev/video0 -pixel_format rgb565 -video_size 320x240
    \endverbatim

    or, with Ubuntu 17.04 and later:

    \verbatim
    guvcview -f RGBP -x 320x240
    \endverbatim

    This module can also work with YUYV output, which is then compatible with MacOS, JeVois Inventor, etc. But note
    that the conversion from RGB565 to YUYV consumes some CPU on the JeVois camera.

    @author Laurent Itti

    @displayname Demo GPU
    @videomapping RGB565 320 240 22.0 YUYV 320 240 22.0 JeVois DemoGPU # Native mode
    @videomapping YUYV 320 240 22.0 YUYV 320 240 22.0 JeVois DemoGPU # With CPU-based conversion from RGB565 to YUYV
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
class DemoGPU : public jevois::Module,
                public jevois::Parameter<effect>
{
  public:
    //! Constrctor
    DemoGPU(std::string const & instance) : jevois::Module(instance)
    {
      itsFilter = addSubComponent<FilterGPU>("gpu");
    }
    
    //! Virtual destructor for safe inheritance
    virtual ~DemoGPU() { }

    //! Processing function
    virtual void process(jevois::InputFrame && inframe, jevois::OutputFrame && outframe) override
    {
      static jevois::Timer timer("DemoGPU");
      
      // Wait for next available camera image:
      jevois::RawImage const inimg = inframe.get();

      timer.start();
      
      // Convert input image to RGBA:
      cv::Mat inimgcv = jevois::rawimage::convertToCvRGBA(inimg);

      // Let camera know we are done processing the input image:
      inframe.done();

      // Wait for an image from our gadget driver into which we will put our results:
      jevois::RawImage outimg = outframe.get();
      if (outimg.width != inimg.width || outimg.height != inimg.height)
        LFATAL("Input and output image sizes must match");
      unsigned int col; // color for our text
      
      if (outimg.fmt == V4L2_PIX_FMT_RGB565)
      {
        // Process input to output, going from YUYV to RGBA internally, to RGB565 rendering:
        cv::Mat outimgcv = jevois::rawimage::cvImage(outimg);
        itsFilter->process(inimgcv, outimgcv);
        col = jevois::rgb565::White;
      }
      else if (outimg.fmt == V4L2_PIX_FMT_YUYV)
      {
        // Process input into a temporary RGB565 buffer, then convert RGB565 to RGB to YUYV on CPU.
        // Clearly this uses a lot of CPU that would offset any gains of GPU computation. For demo purposes only.
        cv::Mat tmpcv(inimg.height, inimg.width, CV_8UC2);
        itsFilter->process(inimgcv, tmpcv);
        cv::Mat tmpcvbgr; cv::cvtColor(tmpcv, tmpcvbgr, cv::COLOR_BGR5652BGR); // note: swap red/blue
        jevois::rawimage::convertCvRGBtoRawImage(tmpcvbgr, outimg, 100); // note: swap red/blue
        col = jevois::yuyv::White;
      }
      else LFATAL("output format must be RGB565 or YUYV");
      
      // Show processing fps:
      std::string const & fpscpu = timer.stop();
      jevois::rawimage::writeText(outimg, fpscpu, 3, outimg.height - 13, col);
      std::ostringstream oss; oss << "JeVois DemoGPU - Effect: " << effect::get();
      jevois::rawimage::writeText(outimg, oss.str(), 3, 3, col);
        
      // Send the output image with our processing results to the host over USB:
      outframe.send();
    }


    // ####################################################################################################
    void onParamChange(effect const & JEVOIS_UNUSED_PARAM(param), Effect const & newval) override
    {
      switch (newval)
      {
      case Effect::NoEffect:
        itsFilter->setProgram("shaders/simplevertshader.glsl", "shaders/simplefragshader.glsl");
        break;

      case Effect::Blur:
        itsFilter->setProgram("shaders/simplevertshader.glsl", "shaders/blurfragshader.glsl");
        break;

      case Effect::Sobel:
        itsFilter->setProgram("shaders/simplevertshader.glsl", "shaders/sobelfragshader.glsl");
        break;

      case Effect::Median:
        itsFilter->setProgram("shaders/simplevertshader.glsl", "shaders/medianfragshader.glsl");
        break;

      case Effect::Mult:
        itsFilter->setProgram("shaders/simplevertshader.glsl", "shaders/multfragshader.glsl");
        break;

      case Effect::Thresh:
        itsFilter->setProgram("shaders/simplevertshader.glsl", "shaders/threshfragshader.glsl");
        break;

      case Effect::Dilate:
        itsFilter->setProgram("shaders/simplevertshader.glsl", "shaders/dilatefragshader.glsl");
        break;

      case Effect::Erode:
        itsFilter->setProgram("shaders/simplevertshader.glsl", "shaders/erodefragshader.glsl");
        break;

      case Effect::Twirl:
        itsFilter->setProgram("shaders/simplevertshader.glsl", "shaders/twirlfragshader.glsl");
        itsFilter->setProgramParam1f("twirlamount", 2.0F);
        break;

      // See fragment shader dewarpfragshader.glsl file for details
      case Effect::Dewarp:
        itsFilter->setProgram("shaders/dewarpvertshader.glsl", "shaders/dewarpfragshader.glsl");
        break;
      }

      // These parameters are used by the vertex shader and hence apply to most demo programs:
      itsFilter->setProgramParam2f("offset", -1.0F, -1.0F);
      itsFilter->setProgramParam2f("scale", 2.0F, 2.0F);

      // Crop Dewarped Image - change According to dewarped image
      itsFilter->setProgramParam2f("offsetd", -1.2F, -1.2F);
      itsFilter->setProgramParam2f("scaled", 2.4F, 2.4F);
    }
    
  private:
    std::shared_ptr<FilterGPU> itsFilter;
};

// Allow the module to be loaded as a shared object (.so) file:
JEVOIS_REGISTER_MODULE(DemoGPU);
