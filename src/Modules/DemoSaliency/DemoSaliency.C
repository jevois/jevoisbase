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
#include <jevois/Debug/Timer.H>
#include <jevois/Image/RawImageOps.H>
#include <jevois/Image/ColorConversion.h>
#include <jevoisbase/Components/Saliency/Saliency.H>
#include <jevoisbase/Components/Tracking/Kalman2D.H>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <future>
#include <linux/videodev2.h> // for v4l2 pixel types

// icon by Freepik in other at flaticon

//! Simple demo of the visual saliency algorithm of Itti et al., IEEE PAMI, 1998
/*! Visual saliency algorithm as described at http://ilab.usc.edu/bu/

    This algorithm finds the location in the camera's view that is the most attention-grabbing, conspicuous, or
    so-called salient. This location is marked on every video frame by the pink square. Salient locations detected on
    each frame are smoothed over time using a Kalman filter. The smoothed attention trajectory is shown with a green
    circle.

    For an introduction to visual saliency computation, see http://ilab.usc.edu/bu/

    Serial Messages
    ---------------

    This module can send standardized serial messages as described in \ref UserSerialStyle, where all coordinates and
    sizes are standardized using \ref coordhelpers. One message is issued on every video frame at the temporally
    smoothed attended (most salient) location (green circle in the video display):

    - Serial message type: \b 2D
    - `id`: always \b sm (shorthand for saliency map)
    - `x`, `y`: standardized 2D coordinates of temporally-filtered most salient point
    - `w`, `h`: standardized size of the pink square box around each attended point
    - `extra`: none (empty string)

    See \ref UserSerialStyle for more on standardized serial messages, and \ref coordhelpers for more info on
    standardized coordinates.


    @author Laurent Itti

    @videomapping YUYV 176 90 120.0 YUYV 88 72 120.0 JeVois DemoSaliency
    @videomapping YUYV 320 150 60.0 YUYV 160 120 60.0 JeVois DemoSaliency
    @videomapping YUYV 352 180 120.0 YUYV 176 144 120.0 JeVois DemoSaliency
    @videomapping YUYV 352 180 100.0 YUYV 176 144 100.0 JeVois DemoSaliency
    @videomapping YUYV 640 300 60.0 YUYV 320 240 60.0 JeVois DemoSaliency
    @videomapping YUYV 704 360 30.0 YUYV 352 288 30.0 JeVois DemoSaliency
    @videomapping YUYV 1280 600 15.0 YUYV 640 480 15.0 JeVois DemoSaliency
    @videomapping YUYV 320 260 30.0 YUYV 320 240 30.0 JeVois DemoArUco
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
class DemoSaliency : public jevois::StdModule
{
  public:
    //! Constructor
    DemoSaliency(std::string const & instance) : jevois::StdModule(instance), itsTimer("DemoSaliency")
    {
      itsSaliency = addSubComponent<Saliency>("saliency");
      itsKF = addSubComponent<Kalman2D>("kalman");
    }

    //! Virtual destructor for safe inheritance
    virtual ~DemoSaliency() { }

    //! Processing function
    virtual void process(jevois::InputFrame && inframe, jevois::OutputFrame && outframe) override
    {
      // Wait for next available camera image:
      jevois::RawImage inimg = inframe.get(); unsigned int const w = inimg.width, h = inimg.height;
      inimg.require("input", w, h, V4L2_PIX_FMT_YUYV); // accept any image size but require YUYV pixels
  
      itsTimer.start();

      // Check whether the input image size is small, in which case we will scale the maps up one notch for the purposes
      // of this demo:
      if (w < 170) { itsSaliency->centermin::set(1); itsSaliency->smscale::set(3); }
      else { itsSaliency->centermin::set(2); itsSaliency->smscale::set(4); }

      // Launch the saliency computation in a thread:
      auto sal_fut = std::async(std::launch::async, [&](){ itsSaliency->process(inimg, true); });
      
      // While computing, wait for an image from our gadget driver into which we will put our results:
      jevois::RawImage outimg = outframe.get();
      
      // Paste the original image to the top-left corner of the display:
      jevois::rawimage::paste(inimg, outimg, 0, 0);
      jevois::rawimage::writeText(outimg, "JeVois Saliency + Gist Demo", 3, 3, jevois::yuyv::White);

      // Once saliency is done using the input image, let camera know we are done with it:
      itsSaliency->waitUntilDoneWithInput();
      inframe.done();
      
      // Wait until saliency computation is complete:
      sal_fut.get();
      
      // Get some info from the saliency computation:
      int const smlev = itsSaliency->smscale::get();
      int const smfac = (1 << smlev);
      int const roihw = (smfac * 3) / 2; // roi half width and height
      int const mapdrawfac = smfac / 4; // factor by which we enlarge the feature maps for drawing
      int const mapdw = (w >> smlev) * mapdrawfac; // width of the drawn feature maps
      int const mapdh = (h >> smlev) * mapdrawfac; // height of the drawn feature maps

      // Enforce the correct output image size and format:
      outimg.require("output", w + (w & ~(smfac-1)), h + mapdh, V4L2_PIX_FMT_YUYV);

      // Find most salient point:
      int mx, my; intg32 msal; itsSaliency->getSaliencyMax(mx, my, msal);
      
      // Compute attended ROI (note: coords must be even to avoid flipping U/V when we later paste):
      unsigned int const dmx = (mx << smlev) + (smfac >> 2);
      unsigned int const dmy = (my << smlev) + (smfac >> 2);
      int rx = std::min(int(w) - roihw, std::max(roihw, int(dmx + 1 + smfac/4)));
      int ry = std::min(int(h) - roihw, std::max(roihw, int(dmy + 1 + smfac/4)));

      // Asynchronously launch a bunch of saliency drawings and filter the attended locations
      auto draw_fut =
        std::async(std::launch::async, [&]() {
            // Filter the attended locations:
            itsKF->set(dmx, dmy, w, h);
            float kfxraw, kfyraw, kfximg, kfyimg;
            itsKF->get(kfxraw, kfyraw, kfximg, kfyimg, inimg.width, inimg.height, 1.0F, 1.0F);

            // Draw a circle around the kalman-filtered attended location:
            jevois::rawimage::drawCircle(outimg, int(kfximg), int(kfyimg), 20, 1, jevois::yuyv::LightGreen);
      
            // Send kalman-filtered most-salient-point coords to serial port (for arduino, etc):
            sendSerialImg2D(inimg.width, inimg.height, kfximg, kfyimg, roihw * 2, roihw * 2, "sm");

            // Paste the saliency map:
            drawMap(outimg, &itsSaliency->salmap, w, 0, smfac, 20);
            jevois::rawimage::writeText(outimg, "Saliency Map", w*2 - 12*6-4, 3, jevois::yuyv::White);
          });

      // Paste the feature maps:
      unsigned int dx = 0; // drawing x offset for each feature map
      drawMap(outimg, &itsSaliency->color, dx, h, mapdrawfac, 18);
      jevois::rawimage::writeText(outimg, "Color", dx+3, h+3, jevois::yuyv::White);
      dx += mapdw;
      
      drawMap(outimg, &itsSaliency->intens, dx, h, mapdrawfac, 18);
      jevois::rawimage::writeText(outimg, "Intensity", dx+3, h+3, jevois::yuyv::White);
      dx += mapdw;
            
      drawMap(outimg, &itsSaliency->ori, dx, h, mapdrawfac, 18);
      jevois::rawimage::writeText(outimg, "Orientation", dx+3, h+3, jevois::yuyv::White);
      dx += mapdw;
            
      drawMap(outimg, &itsSaliency->flicker, dx, h, mapdrawfac, 18);
      jevois::rawimage::writeText(outimg, "Flicker", dx+3, h+3, jevois::yuyv::White);
      dx += mapdw;
      
      drawMap(outimg, &itsSaliency->motion, dx, h, mapdrawfac, 18);
      jevois::rawimage::writeText(outimg, "Motion", dx+3, h+3, jevois::yuyv::White);
      dx += mapdw;

      // Blank out free space in bottom-right corner, we will then draw the gist (which may only partially occupy that
      // available space):
      unsigned int const gw = outimg.width - dx, gh = outimg.height - h;
      jevois::rawimage::drawFilledRect(outimg, dx, h, gw, gh, 0x8000);

      // Draw the gist vector, picking a zoom factor to maximize the area filled:
      unsigned int const gscale = int(sqrt((gw * gh) / itsSaliency->gist_size));

      drawGist(outimg, itsSaliency->gist, itsSaliency->gist_size, dx, h, gw / gscale, gscale);
      jevois::rawimage::drawRect(outimg, dx, h, gw, gh, 0x80a0);
      jevois::rawimage::writeText(outimg, "Gist", dx+3, h+3, jevois::yuyv::White);

      // Wait for all drawings to complete:
      draw_fut.get();

      // Draw a small square at most salient location in image and in saliency map:
      jevois::rawimage::drawFilledRect(outimg, dmx + 1, dmy + 1, smfac/2, smfac/2, 0xffff);
      jevois::rawimage::drawFilledRect(outimg, w + dmx + 1, dmy + 1, smfac/2, smfac/2, 0xffff);

      // Draw an ROI box around the most salient point:
      jevois::rawimage::drawRect(outimg, rx - roihw, ry - roihw, roihw*2, roihw*2, 0xf0f0);
      jevois::rawimage::drawRect(outimg, rx - roihw + 1, ry - roihw + 1, roihw*2-2, roihw*2-2, 0xf0f0);

      // Show processing fps:
      std::string const & fpscpu = itsTimer.stop();
      jevois::rawimage::writeText(outimg, fpscpu, 3, h - 13, jevois::yuyv::White);
      
      // Send the output image with our processing results to the host over USB:
      outframe.send();
    }

  protected:
    std::shared_ptr<Saliency> itsSaliency;
    std::shared_ptr<Kalman2D> itsKF;
    jevois::Timer itsTimer;
};

// Allow the module to be loaded as a shared object (.so) file:
JEVOIS_REGISTER_MODULE(DemoSaliency);
