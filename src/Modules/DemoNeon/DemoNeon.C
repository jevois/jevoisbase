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
#include <jevois/Debug/Timer.H>

#include <linux/videodev2.h>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <string.h>

// Neon-related:
#include <Ne10/inc/NE10_imgproc.h>

// icon by by Madebyoliver in technology at flaticon

namespace
{
  // OpenCV's cvtColor() cannot convert from RGBA to YUYV. Found this code here and cleaned it up a bit:
  // http://study.marearts.com/2014/12/yuyv-to-rgb-and-rgb-to-yuyv-using.html
  class Parallel_process : public cv::ParallelLoopBody
  {
    private:
      cv::Mat const & inImg;
      unsigned char * outImg;
      int widhStep;
      int m_stride;
      
    public:
      Parallel_process(cv::Mat const & inputImgage,  unsigned char* outImage, size_t outw) :
          inImg(inputImgage), outImg(outImage)
      {
        widhStep = inputImgage.size().width * 4; // 4bpp for RGBA
        m_stride = outw * 2; // 2bpp for YUYV
      }

      virtual void operator()(const cv::Range & range) const
      {
        for (int i = range.start; i < range.end; ++i)
        {
          int const s1 = i * widhStep;
          
          for (int iw = 0; iw < inImg.size().width; iw += 2)
          {
            int const s2 = iw * 4; int mc = s1 + s2;
            float const R1 = inImg.data[mc + 0];
            float const G1 = inImg.data[mc + 1];
            float const B1 = inImg.data[mc + 2];
            // skip A
            float const R2 = inImg.data[mc + 4];
            float const G2 = inImg.data[mc + 5];
            float const B2 = inImg.data[mc + 6];
            // skip A
            
            int Y = (0.257F * R1) + (0.504F * G1) + (0.098F * B1) + 16;
            int U = -(0.148F * R1) - (0.291F * G1) + (0.439F * B1) + 128;
            int V = (0.439F * R1 ) - (0.368F * G1) - (0.071F * B1) + 128;
            int Y2 = (0.257F * R2) + (0.504F * G2) + (0.098F * B2) + 16;

            if (Y > 255) Y = 255; else if (Y < 0) Y = 0;
            if (U > 255) U = 255; else if (U < 0) U = 0;
            if (V > 255) V = 255; else if (V < 0) V = 0;
            if (Y2 > 255) Y2 = 255; else if (Y2 < 0) Y2 = 0;
            
            mc = i * m_stride + iw * 2;
            outImg[mc + 0] = Y; outImg[mc + 1] = U; outImg[mc + 2] = Y2; outImg[mc + 3] = V;
          }
        }
      }
  };
  
  void rgba2yuyv(cv::Mat const & src, unsigned char * dst, size_t dstw)
  { cv::parallel_for_(cv::Range(0, src.rows), Parallel_process(src, dst, dstw)); }

} // anonymous namespace

// Module parameters: allow user to play with filter kernel size
static jevois::ParameterCategory const ParamCateg("Neon Demo Options");

//! Parameter \relates DemoNeon
JEVOIS_DECLARE_PARAMETER(kernelw, unsigned int, "Kernel width (pixels)", 5, ParamCateg);

//! Parameter \relates DemoNeon
JEVOIS_DECLARE_PARAMETER(kernelh, unsigned int, "Kernel height (pixels)", 5, ParamCateg);

//! Simple demo of ARM Neon (SIMD) extensions, comparing a box filter (blur) between CPU and Neon
/*! NEON are specialized ARM processor instructions that can handle several operations at once, for example, 8 additions
    of 8 bytes with 8 other bytes. NEON is the counterpart for ARM architectures of SSE for Intel architectures.

    They are very useful for image processing. NEON instructions are supported both by the JeVois hardware platform and
    by the JeVois programming framework.

    In fact, one can directly call NEON instructions using C-like function calls and specialized C data types to
    represent small vectors of numbers (like 8 bytes).

    This demo uses a blur filter from the open-source NE10 library. It compares processing time to apply the same filter
    to the input video stream, either using conventional C code, or using NEON-accelerated code. The NEON-accelerated
    code is about 6x faster.

    For more examples of use of NEON on JeVois, see modules \jvmod{DarknetSingle}, \jvmod{DarknetYOLO}, and
    \jvmod{DarknetSaliency} which use NEON to accelerate the deep neural networks implemented in these modules.


    @author Laurent Itti

    @displayname Demo NEON
    @videomapping YUYV 960 240 30.0 YUYV 320 240 30.0 JeVois DemoNeon
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
class DemoNeon : public jevois::Module,
                 public jevois::Parameter<kernelw, kernelh>
{
  public:
    //! Default base class constructor ok
    using jevois::Module::Module;

    //! Virtual destructor for safe inheritance
    virtual ~DemoNeon() { }

    //! Processing function
    virtual void process(jevois::InputFrame && inframe, jevois::OutputFrame && outframe) override
    {
      static jevois::Timer cputim("CPU time");
      static jevois::Timer neontim("Neon time");

      // Wait for next available camera image:
      jevois::RawImage inimg = inframe.get();
      unsigned int const w = inimg.width, h = inimg.height;
      inimg.require("input", w, h, V4L2_PIX_FMT_YUYV); // any image size but require YUYV pixels

      // While we convert it, start a thread to wait for out frame and paste the input into it:
      jevois::RawImage outimg;
      auto paste_fut = std::async(std::launch::async, [&]() {
          outimg = outframe.get();
          outimg.require("output", w * 3, h, inimg.fmt);
          jevois::rawimage::paste(inimg, outimg, 0, 0);
          jevois::rawimage::writeText(outimg, "JeVois NEON Demo", 3, 3, jevois::yuyv::White);
        });
      
      // Convert input frame to RGBA:
      cv::Mat imgrgba = jevois::rawimage::convertToCvRGBA(inimg);

      // Wait for paste to finish up:
      paste_fut.get();

      // Let camera know we are done processing the input image:
      inframe.done();

      // First, apply blur filter using CPU:
      ne10_size_t src_size { w, h }, kernel_size { kernelw::get(), kernelh::get() };

      cv::Mat cpuresult(h, w, CV_8UC4);
      cputim.start();
      ne10_img_boxfilter_rgba8888_c(imgrgba.data, cpuresult.data, src_size, w * 4, w * 4, kernel_size);
      std::string const & cpufps = cputim.stop();
      
      // Then apply it using neon:
      cv::Mat neonresult(h, w, CV_8UC4);
      neontim.start();

#ifdef __ARM_NEON__
      // Neon version:
      ne10_img_boxfilter_rgba8888_neon(imgrgba.data, neonresult.data, src_size, w * 4, w * 4, kernel_size);
#else
      // On non-ARM/NEON host, revert to CPU version again:
      ne10_img_boxfilter_rgba8888_c(imgrgba.data, neonresult.data, src_size, w * 4, w * 4, kernel_size);
#endif

      std::string const & neonfps = neontim.stop();
      
      // Convert both results back to YUYV for display:
      rgba2yuyv(cpuresult, outimg.pixelsw<unsigned char>() + w * 2, w * 3);
      jevois::rawimage::writeText(outimg, "Box filter - CPU", w + 3, 3, jevois::yuyv::White);
      rgba2yuyv(neonresult, outimg.pixelsw<unsigned char>() + w * 4, w * 3);
      jevois::rawimage::writeText(outimg, "Box filter - NEON", w * 2 + 3, 3, jevois::yuyv::White);

      // Show processing fps:
      jevois::rawimage::writeText(outimg, cpufps, w + 3, h - 13, jevois::yuyv::White);
      jevois::rawimage::writeText(outimg, neonfps, w * 2 + 3, h - 13, jevois::yuyv::White);
   
      // Send the output image with our processing results to the host over USB:
      outframe.send();
    }
};

// Allow the module to be loaded as a shared object (.so) file:
JEVOIS_REGISTER_MODULE(DemoNeon);
