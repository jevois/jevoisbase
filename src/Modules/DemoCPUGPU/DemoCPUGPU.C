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

#include <jevoisbase/Components/FilterGPU/FilterGPU.H>
#include <jevoisbase/Components/Saliency/Saliency.H>
#include <jevoisbase/Components/Tracking/Kalman2D.H>

#include <jevois/Image/RawImageOps.H>
#include <linux/videodev2.h>

#include <opencv2/imgproc/imgproc.hpp>
#include <future>

// icon by Freepik in computer at flaticon

//! Live saliency computation and image filtering using 4-core CPU and OpenGL-ES 2.0 shaders on the Mali-400MP2 GPU
/*! This module computes saliency and gist over our 4 CPU cores while also computing 4 different image filters over the
    GPU, finally combining all results into a single grayscale image:

    - saliency: multicore CPU-based detection of the most conspicuous (most attention-grabbing) object in the field of
      view.
    - GPU filter 1: Sobel edge detector
    - GPU filter 2: Median filter
    - GPU filter 3: Morphological erosion filter
    - GPU filter 4: Morphological dilation filter

    For an introduction to visual saliency, see http://ilab.usc.edu/bu/

    Also see \jvmod{DemoSaliency}, \jvmod{JeVoisIntro}, \jvmod{DarknetSaliency} for more about saliency.

    Video output
    ------------

    The video output is arranged vertically, with, from top to bottom:
    - Sobel filter results (same size as input image)
    - Median filter results (same size as input image)
    - Morphological erosion filter results (same size as input image)
    - Morphological dilation filter results (same size as input image)
    - Saliency results (those are very small): from left to right: saliency map, color map, intensity map, orientation
      map, flicker map, motion map. Map size is input size divided by 8 horizontally and vertically. Gist vector is
      appended to the right but note that at 160x120 it is truncated (see source code for details).

    Serial Messages
    ---------------

    This module can send standardized serial messages as described in \ref UserSerialStyle, where all coordinates and
    sizes are standardized using \ref coordhelpers. One message is issued on every video frame at the temporally
    filtered attended (most salient) location:

    - Serial message type: \b 2D
    - `id`: always \b sm (shorthand for saliency map)
    - `x`, `y`: standardized 2D coordinates of temporally-filtered most salient point
    - `w`, `h`: always 0, 0
    - `extra`: none (empty string)

    See \ref UserSerialStyle for more on standardized serial messages, and \ref coordhelpers for more info on
    standardized coordinates.

    @author Laurent Itti

    @displayname Demo CPU GPU
    @videomapping GREY 160 495 60.0 YUYV 160 120 60.0 JeVois DemoCPUGPU
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
class DemoCPUGPU : public jevois::StdModule
{
  public:
    //! Constructor
    DemoCPUGPU(std::string const & instance) : jevois::StdModule(instance)
    {
      itsFilter = addSubComponent<FilterGPU>("gpu");
      itsSaliency = addSubComponent<Saliency>("saliency");
      itsKF = addSubComponent<Kalman2D>("kalman");

      // Use some fairly large saliency and feature maps so we can see them:
      itsSaliency->centermin::set(1);
      itsSaliency->smscale::set(3);
    }
    
    //! Virtual destructor for safe inheritance
    virtual ~DemoCPUGPU() { }

    //! Set our GPU program after we are fully constructed and our Component path has been set
    void postInit() override
    {
      itsFilter->setProgram("shaders/simplevertshader.glsl", "shaders/combofragshader.glsl");
      itsFilter->setProgramParam2f("offset", -1.0F, -1.0F);
      itsFilter->setProgramParam2f("scale", 2.0F, 2.0F);
    }
    
    //! Processing function
    virtual void process(jevois::InputFrame && inframe, jevois::OutputFrame && outframe) override
    {
      // Wait for next available camera image:
      jevois::RawImage const inimg = inframe.get(); unsigned int const w = inimg.width, h = inimg.height;
      inimg.require("input", w, h, V4L2_PIX_FMT_YUYV); // accept any image size but require YUYV pixels

      // In this demo, the GPU completes earlier than the CPU. Hence, we are going to wait for the output frame in the
      // main thread (which runs the GPU code). We use a mutex to signal to the saliency thread when the output image is
      // available. Using a mutex and unique_lock here ensures that we will be exception-safe (which may be trickier
      // with a condition variable or such):
      jevois::RawImage outimg;
      std::unique_lock<std::mutex> lck(itsOutMtx); // mutex will be released by main thread when outimg is available
      
      // Launch the saliency computation in a thread:
      auto sal_fut = std::async(std::launch::async, [&](){
          // Compute saliency and gist:
          itsSaliency->process(inimg, true);

          // Find most salient point:
          int mx, my; intg32 msal; itsSaliency->getSaliencyMax(mx, my, msal);
      
          // Compute attended location in original frame coordinates:
          int const smlev = itsSaliency->smscale::get();
          int const smadj = smlev > 0 ? (1 << (smlev-1)) : 0; // half a saliency map pixel adjustment
          unsigned int const dmx = (mx << smlev) + smadj;
          unsigned int const dmy = (my << smlev) + smadj;
          unsigned int const mapw = itsSaliency->salmap.dims.w, maph = itsSaliency->salmap.dims.h;
          unsigned int const gistsize = itsSaliency->gist_size;
          unsigned int const gistw = w - 6 * mapw; // width for gist block
          unsigned int const gisth = (gistsize + gistw - 1) / gistw; // divide gist_size by w and round up
          
          // Filter over time the salient location coordinates:
          itsKF->set(dmx, dmy, w, h);
          float kfxraw, kfyraw; itsKF->get(kfxraw, kfyraw, 1.0F); // round to int for serial

          // Send kalman-filtered most-salient-point info to serial port (for arduino, etc):
          sendSerialStd2D(kfxraw, kfyraw, 0.0F, 0.0F, "sm");

          // Wait for output image to be available:
          std::lock_guard<std::mutex> _(itsOutMtx);

          // Paste saliency and feature maps:
          unsigned int offset = 0;
          pasteGrayMap(outimg, itsSaliency->salmap, offset, h*4, 20);
          pasteGrayMap(outimg, itsSaliency->color, offset, h*4, 18);
          pasteGrayMap(outimg, itsSaliency->intens, offset, h*4, 18);
          pasteGrayMap(outimg, itsSaliency->ori, offset, h*4, 18);
          pasteGrayMap(outimg, itsSaliency->flicker, offset, h*4, 18);
          pasteGrayMap(outimg, itsSaliency->motion, offset, h*4, 18);

          // Paste gist. Note: at 160x120 we will only end up pasting the first 600 gist entries. This code may fail at
          // different resolutions, beware:
          unsigned char * d = outimg.pixelsw<unsigned char>() + 4*w*h + 6*mapw;
          for (int i = 0; i < maph; ++i) memcpy(d + i*w, itsSaliency->gist + i*gistw, gistw);
        });

      // Convert input image to grayscale:
      cv::Mat grayimg = jevois::rawimage::convertToCvGray(inimg);

      // Once saliency is done using the input image, let camera know we are done with it:
      itsSaliency->waitUntilDoneWithInput();
      inframe.done();
 
      // Upload the greyscale image to GPU and apply 4 grayscale GPU filters, storing the 4 results into an RGBA image:
      cv::Mat gpuout(h, w, CV_8UC4);
      itsFilter->process(grayimg, gpuout);

      // Wait for an image from our gadget driver into which we will put our results:
      outimg = outframe.get();
      lck.unlock(); // let saliency thread proceed with using the output image
      outimg.require("output", w, h * 4 + (h >> itsSaliency->smscale::get()), V4L2_PIX_FMT_GREY);
      
      // Unpack the GPU output into 4 gray images, starting from top-left of the output image:
      jevois::rawimage::unpackCvRGBAtoGrayRawImage(gpuout, outimg);

      // Wait until saliency computation is complete:
      sal_fut.get();

      // Send the output image with our processing results to the host over USB:
      outframe.send();
    }

    // ####################################################################################################
    //! Paste a map and add its width to the dx offset
    /*! Beware this is for a gray outimg only. */
    void pasteGrayMap(jevois::RawImage & outimg, env_image const & fmap, unsigned int & dx, unsigned int dy,
                      unsigned int bitshift)
    {
      env_size_t const fw = fmap.dims.w, fh = fmap.dims.h;
      unsigned int const ow = outimg.width, oh = outimg.height;
      
      if (dy + fh > oh) LFATAL("Map would extend past output image bottom");
      if (fw + dx > ow) LFATAL("Map would extend past output image width");
      
      unsigned int const stride = ow - fw;
      
      intg32 * s = fmap.pixels; unsigned char * d = outimg.pixelsw<unsigned char>() + dx + dy * ow;
      
      for (unsigned int j = 0; j < fh; ++j)
      {
        for (unsigned int i = 0; i < fw; ++i)
        {
          intg32 v = (*s++) >> bitshift; if (v > 255) v = 255;
          *d++ = (unsigned char)(v);
        }
        d += stride;
      }
      
      // Add the map width to the dx offset:
      dx += fw;
    }

  private:
    std::shared_ptr<FilterGPU> itsFilter;
    std::shared_ptr<Saliency> itsSaliency;
    std::shared_ptr<Kalman2D> itsKF;
    std::mutex itsOutMtx;
};

// Allow the module to be loaded as a shared object (.so) file:
JEVOIS_REGISTER_MODULE(DemoCPUGPU);
