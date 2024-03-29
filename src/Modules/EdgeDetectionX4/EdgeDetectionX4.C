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
#include <jevois/Image/RawImageOps.H>

#include <linux/videodev2.h>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <future>

// icon by Sergiu Bagrin in interface at flaticon

static jevois::ParameterCategory const ParamCateg("Edge Detection Options");
//! Parameter \relates EdgeDetectionX4
JEVOIS_DECLARE_PARAMETER(thresh1, double, "First threshold for hysteresis", 20.0, ParamCateg);
//! Parameter \relates EdgeDetectionX4
JEVOIS_DECLARE_PARAMETER(thresh2, double, "Second threshold for hysteresis", 60.0, ParamCateg);
//! Parameter \relates EdgeDetectionX4
JEVOIS_DECLARE_PARAMETER(aperture, int, "Aperture size for the Sobel operator", 3, ParamCateg);
//! Parameter \relates EdgeDetectionX4
JEVOIS_DECLARE_PARAMETER(l2grad, bool, "Use more accurate L2 gradient norm if true, L1 if false", false, ParamCateg);
//! Parameter \relates EdgeDetectionX4
JEVOIS_DECLARE_PARAMETER(thresh1delta, double, "First threshold delta over threads", 50.0, ParamCateg);
//! Parameter \relates EdgeDetectionX4
JEVOIS_DECLARE_PARAMETER(thresh2delta, double, "Second threshold delta over threads", 50.0, ParamCateg);

//! Simple module to detect edges, running 4 filters in parallel with 4 different settings
/*! Compute 4 Canny edge detection filters with 4 different settings, in parallel.

    This module is useful as a pre-processor, feeding edge maps at 4 different levels of details to further processing
    that may happen on a host computer. The 4 different levels of detail can be leveraged to first detect the gross
    outlines of objects, and then focus on finer textures within these objects.

    This algorithm should easily run at 45 frames/s on the JeVois smart camera.

   
    @author Laurent Itti

    @displayname Edge Detection X4
    @videomapping GREY 320 960 45.0 YUYV 320 240 45.0 JeVois EdgeDetectionX4
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
class EdgeDetectionX4 : public jevois::Module,
                        public jevois::Parameter<thresh1, thresh2, aperture, l2grad, thresh1delta, thresh2delta>
{
  public:
    //! Default base class constructor ok
    using jevois::Module::Module;

    //! Virtual destructor for safe inheritance
    virtual ~EdgeDetectionX4() { }

    //! Processing function
    virtual void process(jevois::InputFrame && inframe, jevois::OutputFrame && outframe) override
    {
      // Wait for next available camera image:
      jevois::RawImage inimg = inframe.get();

      // Convert to grayscale:
      cv::Mat grayimg = jevois::rawimage::convertToCvGray(inimg);
  
      // Let camera know we are done processing the input image:
      inframe.done();

      // Wait for an image from our gadget driver into which we will put our results:
      jevois::RawImage outimg = outframe.get();
      outimg.require("output", inimg.width, inimg.height * 4, V4L2_PIX_FMT_GREY);

      // Launch 4 Canny filters in parallel. We launch 3 threads and will do the fourth in the current thread:
      std::vector<std::future<void> > fut;
      
      for (int i = 0; i < 3; ++i)
        fut.push_back(jevois::async([&](int i) {
              // Compute Canny edges directly into the output image, offset by i images down. The last argument of the
              // cv::Mat constructor below is the address of an already-allocated pixel buffer for the cv::Mat:
              cv::Mat edges(grayimg.rows, grayimg.cols, CV_8UC1, outimg.pixelsw<unsigned char>() + i * grayimg.total());

              cv::Canny(grayimg, edges, thresh1::get() + i * thresh1delta::get(),
                        thresh2::get() + i * thresh2delta::get(), aperture::get(), l2grad::get());
            }, i));

      // Fourth one (same code as above except for the async, and for i=3):
      cv::Mat edges(grayimg.rows, grayimg.cols, CV_8UC1, outimg.pixelsw<unsigned char>() + 3 * grayimg.total());
      cv::Canny(grayimg, edges, thresh1::get() + 3 * thresh1delta::get(),
                thresh2::get() + 3 * thresh2delta::get(), aperture::get(), l2grad::get());

      // The fourth one is done now, wait for all the threads to complete. Note: using async() is preferred to using
      // std::thread, as get() below will throw if any exception was thrown by a thread, as opposed to std::thread
      // violently terminating the program on exception. In case two or more threads threw, we can here avoid
      // termination by catching the exceptions one by one. Here we just ignore (since we are done anyway) but could
      // throw just once if any of the threads threw:
      for (auto & f : fut) try { f.get(); } catch (...) { jevois::warnAndIgnoreException(); }
      
      // Send the output image with our processing results to the host over USB:
      outframe.send();
    }
};

// Allow the module to be loaded as a shared object (.so) file:
JEVOIS_REGISTER_MODULE(EdgeDetectionX4);
