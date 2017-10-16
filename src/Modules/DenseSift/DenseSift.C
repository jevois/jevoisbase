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

#include <vlfeat/vl/dsift.h>

// Module parameters: allow user to play with step and binsize:
static jevois::ParameterCategory const ParamCateg("Dense Sift Options");

//! Parameter \relates DenseSift
JEVOIS_DECLARE_PARAMETER(step, unsigned int, "Keypoint step (pixels)", 11, ParamCateg);

//! Parameter \relates DenseSift
JEVOIS_DECLARE_PARAMETER(binsize, unsigned int, "Descriptor bin size", 8, ParamCateg);

// icon by Pixel Buddha in interface at flaticon

//! Simple demo of dense SIFT feature descriptors extraction
/*! Compute SIFT keypoint descriptors on a regular grid over the input image.

    This module is useful when using JeVois as a pre-processor, delivering a dense array of keypoint descriptors to a
    host computer, where the array is disguised as a grayscale video frame. Upon receiving the array of descriptors, the
    host computer can further process them. For example, the host computer may compute camera motion in space by
    matching descriptors across successive frames, or may attempt to detect and identify objects based on the
    descriptors.

    Beware that changing the values for the \p step and \p binsize parameters changes the output image size, so you need
    to adjust your video mappings accordingly. Hence, setting those parameters is best done once and for all in the
    module's optional \b params.cfg or \b script.cfg file.

    This module can either have a color YUYV output, which shows the original camera image, keypoint locations, and
    descriptor values; or a greyscale output, which is just the descriptor values.

    This algorithm is implemented using the VLfeat library. It is quite slow, maybe because this library is a bit old
    and appears to be single-threaded.


    @author Laurent Itti

    @displayname Dense SIFT
    @videomapping YUYV 288 120 5.0 YUYV 160 120 5.0 JeVois DenseSift
    @videomapping GREY 128 117 5.0 YUYV 160 120 5.0 JeVois DenseSift
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
class DenseSift : public jevois::Module,
                  public jevois::Parameter<step, binsize>
{
  public:
    //! Default base class constructor ok
    using jevois::Module::Module;

    //! Virtual destructor for safe inheritance
    virtual ~DenseSift() { }

    //! Processing function
    virtual void process(jevois::InputFrame && inframe, jevois::OutputFrame && outframe) override
    {
      static jevois::Timer timer("processing");

      // Wait for next available camera image:
      jevois::RawImage inimg = inframe.get();
      unsigned int const w = inimg.width, h = inimg.height;
      inimg.require("input", w, h, V4L2_PIX_FMT_YUYV); // any image size but require YUYV pixels
      bool demodisplay = false;
      
      timer.start();
      
      // Create the dense sift filter:
      VlDsiftFilter * vlds = vl_dsift_new_basic(w, h, step::get(), binsize::get());
      int const descsize = vl_dsift_get_descriptor_size(vlds);
      int const numkp = vl_dsift_get_keypoint_num(vlds);

      // Everything from here on is in a try-catch so we de-allocate vlds on exception:
      try
      {
        // While we convert it, start a thread to wait for out frame and paste the input into it:
        jevois::RawImage outimg;
        auto paste_fut = std::async(std::launch::async, [&]() {
            // Get next output video frame:
            outimg = outframe.get();

            // Do we want color (demo) or grey (raw data) output:
            switch (outimg.fmt)
            {
            case V4L2_PIX_FMT_YUYV:
              demodisplay = true;
              outimg.require("output", w + 128, h, V4L2_PIX_FMT_YUYV);
              jevois::rawimage::paste(inimg, outimg, 0, 0);
              jevois::rawimage::writeText(outimg, "JeVois Dense SIFT Demo", 3, 3, jevois::yuyv::White);
            
              // if the number of keypoints (based on step) is smaller than image height, blank out bottom:
              if (numkp < int(h))
                jevois::rawimage::drawFilledRect(outimg, w, numkp, descsize, h-numkp, jevois::yuyv::DarkGrey);
              break;
              
            case V4L2_PIX_FMT_GREY:
              demodisplay = false;
              outimg.require("output", descsize, numkp, V4L2_PIX_FMT_GREY);
              break;

            default: LFATAL("This module only supports YUYV or GREY output images");
            }
          });
        
        // Convert input frame to gray byte first:
        cv::Mat grayimgcv = jevois::rawimage::convertToCvGray(inimg);
        
        // Then we need it as floats for vlfeat:
        cv::Mat floatimgcv; grayimgcv.convertTo(floatimgcv, CV_32F, 1.0, 0.0);
        
        // Wait for paste to finish up:
        paste_fut.get();
        
        // Let camera know we are done processing the input image:
        inframe.done();
        
        // Process the float gray image:
        vl_dsift_process(vlds, reinterpret_cast<float const *>(floatimgcv.data));
        
        // Get the descriptors: size is descsize * numkp:
        float const * descriptors = vl_dsift_get_descriptors(vlds);
        
        // Convert them to byte using opencv. The conversion factor should be 255, but for demo display the descriptors
        // look mostly black, so we use a higher factor for demo display:
        cv::Mat dfimg(numkp, descsize, CV_32FC1, reinterpret_cast<void *>(const_cast<float *>(descriptors)));
        cv::Mat bdfimg; dfimg.convertTo(bdfimg, CV_8U, (demodisplay ? 512.0 : 255.0), 0.0);

        std::string const & fpscpu = timer.stop();

        if (demodisplay)
        {
          // Paste into our output image:
          jevois::rawimage::pasteGreyToYUYV(bdfimg, outimg, w, 0);
          jevois::rawimage::writeText(outimg, "SIFT", w + 3, 3, jevois::yuyv::LightGreen);

          // Draw the keypoint locations and scale:
          VlDsiftKeypoint const * keypoints = vl_dsift_get_keypoints(vlds);

          for (int i = 0; i < numkp; ++i)
          {
            VlDsiftKeypoint const & kp = keypoints[i];
            unsigned int s = (unsigned int)(kp.s / 150.0 + 1.499); if (s >20) s = 20;
            jevois::rawimage::drawDisk(outimg, int(kp.x + 0.499), int(kp.y + 0.499), s, jevois::yuyv::LightPink);
          }

          // Show processing fps:
          jevois::rawimage::writeText(outimg, fpscpu, 3, h - 13, jevois::yuyv::White);
        }
        else
        {
          // Just copy the byte-converted descriptors to the output image:
          memcpy(outimg.buf->data(), bdfimg.data, outimg.width * outimg.height);
        }
        
        // Send the output image with our processing results to the host over USB:
        outframe.send();
      }
      catch (...) { jevois::warnAndIgnoreException(); }

      // Nuke the dense sift computer:
      vl_dsift_delete(vlds);
    }
};

// Allow the module to be loaded as a shared object (.so) file:
JEVOIS_REGISTER_MODULE(DenseSift);
