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
#include <stdlib.h>

// Neon-related:
#include <Ne10/inc/NE10_imgproc.h>

// GPU-related:
#include <jevoisbase/Components/FilterGPU/FilterGPU.H>

// icon by Vectors Market in nature at www.flaticon.com

//! This is a burn test: run the quad-core saliency demo while also loading up CPU, GPU and NEON in the background
/*! This burn test exercises all aspects of your JeVois smart camera to the maximum, namely:

    - launch two instances of whetstone (floating point benchmark test) running in the background
    - launch two instances of dhrystone (integer benchmark test) running in the background
    - grab frames from the camera sensor
    - run the quad-core visual attention algorithm
    - in parallel, run the NEON demo that blurs the video frames using NEON accelerated processor instructions
    - in parallel, run the GPU demo that processes the video through 4 image filters (shaders)
    - stream attention video results over USB
    - issue messages over the serial port

    This burn test is useful to test JeVois hardware for any malfunction. It should run forever without crashing on
    JeVois hardware. Demo display layout and markings are the same as for the \jvmod{DemoSaliency} module.

    This burn test is one of the tests that every JeVois camera produced is tested with at the factory, before the unit
    is shipped out.

    Things to try
    -------------

    Select the burntest video mode (note that it is 640x300 \@ 10fps, while the default MicroSD card also includes a
    mode with 640x300 \@ 60fps that runs the \jvmod{DemoSaliency} module instead). You need to activate it (remove the
    leading \b # sign) in <b>JEVOIS:/config/videomappings.cfg</b> as it is disabled by default. Observe the CPU
    temperature at the bottom of the live video window. If it ever reaches 75C (which it should not under normal
    conditions given the high power fan on the JeVois smart camera), the CPU frequency shown next to the temperature
    will drop down below 1344 MHz, and will then come back up as the CPU temperature drops below 75C.

    Connect your JeVois camera to your host computer through a USB Tester device that measures voltage, current, and
    power. You should reach about 3.7 Watts under the burn test, which is the maximum we have ever been able to achieve
    with a JeVois unit.


    @author Laurent Itti

    @videomapping YUYV 640 300 10.0 YUYV 320 240 10.0 JeVois BurnTest
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
class BurnTest : public jevois::Module
{
  public:
    //! Constructor
    BurnTest(std::string const & instance) :
        jevois::Module(instance), itsTimer("BurnTest", 30, LOG_DEBUG)
    {
      itsSaliency = addSubComponent<Saliency>("saliency");
      itsKF = addSubComponent<Kalman2D>("kalman");
      itsFilter = addSubComponent<FilterGPU>("gpu");
    }

    //! Set our GPU program after we are fully constructed and our Component path has been set
    void postInit() override
    {
      itsFilter->setProgram("shaders/simplevertshader.glsl", "shaders/combofragshader.glsl");
      itsFilter->setProgramParam2f("offset", -1.0F, -1.0F);
      itsFilter->setProgramParam2f("scale", 2.0F, 2.0F);
      itsRunning.store(true);
    }

    //! Kill our external processes on uninit
    void postUninit() override
    {
      itsRunning.store(false);
      system("/bin/rm /tmp/jevois-burntest");
      system("killall -9 dhrystone");
      system("killall -9 whetstone");
      try { itsGPUfut.get(); } catch (...) { }
      try { itsNEONfut.get(); } catch (...) { }
    }
    
    //! Virtual destructor for safe inheritance
    virtual ~BurnTest() { }

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
      jevois::rawimage::writeText(outimg, "JeVois CPU+GPU+NEON BurnTest", 3, 3, jevois::yuyv::White);

      // Once saliency is done using the input image, let camera know we are done with it:
      itsSaliency->waitUntilDoneWithInput();
      inframe.done();

      // If our grayimg is empty, compute it and launch the GPU and NEON threads:
      if (itsGrayImg.empty())
      {
        // Create a temp file:
        system("touch /tmp/jevois-burntest");
        
        // start a couple of whetstones:
        system("( while [ -f /tmp/jevois-burntest ]; do whetstone 2000000000; done ) &");
        system("( while [ -f /tmp/jevois-burntest ]; do whetstone 2000000000; done ) &");

        // dhrystne too
        system("( while [ -f /tmp/jevois-burntest ]; do dhrystone 2000000000; done ) &");
        system("( while [ -f /tmp/jevois-burntest ]; do dhrystone 2000000000; done ) &");

        // then load the GPU
        itsGrayImg = jevois::rawimage::convertToCvGray(inimg);
        itsGPUfut = std::async(std::launch::async, [&](unsigned int ww, unsigned int hh) {
            cv::Mat gpuout(hh, ww, CV_8UC4);
            while (itsRunning.load()) itsFilter->process(itsGrayImg, gpuout);
          }, w, h);

        // and load NEON too
        itsRGBAimg = jevois::rawimage::convertToCvRGBA(inimg);

        itsNEONfut = std::async(std::launch::async, [&](unsigned int ww, unsigned int hh) {
            cv::Mat neonresult(hh, ww, CV_8UC4);
            ne10_size_t src_size { ww, hh }, kernel_size { 5, 5 };
            while (itsRunning.load())
            {
#ifdef __ARM_NEON__
              // Neon version:
              ne10_img_boxfilter_rgba8888_neon(itsRGBAimg.data, neonresult.data, src_size, ww * 4, ww * 4, kernel_size);
#else
              // On non-ARM/NEON host, revert to CPU version again:
              ne10_img_boxfilter_rgba8888_c(itsRGBAimg.data, neonresult.data, src_size, ww * 4, ww * 4, kernel_size);
#endif
            }
          }, w, h);
      }
    
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
            
            // Send saliency info to serial port (for arduino, etc):
            //sendSerial(jevois::sformat("T2D %d %d", int(kfxraw), int(kfyraw)));

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
    std::future<void> itsGPUfut;
    std::shared_ptr<FilterGPU> itsFilter;
    cv::Mat itsGrayImg;
    std::future<void> itsNEONfut;
    cv::Mat itsRGBAimg;
    std::atomic<bool> itsRunning;
};

// Allow the module to be loaded as a shared object (.so) file:
JEVOIS_REGISTER_MODULE(BurnTest);
