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
    using the JeVois camera in pass-through mode as input, to simulate what will happen when your code runs on the
    JeVois embedded processor.

    Any video mapping is possible here, as long as camera and USB pixel types match, and camera and USB image
    resolutions also match.

    See \ref PixelFormats for information about pixel formats; with this module you can use the formats supported by the
    camera sensor: YUYV, BAYER, RGB565, and resolutions:
    
    - SXGA (1280 x 1024): up to 15 fps
    - VGA (640 x 480): up to 30 fps
    - CIF (352 x 288): up to 60 fps
    - QVGA (320 x 240): up to 60 fps
    - QCIF (176 x 144): up to 120 fps
    - QQVGA (160 x 120): up to 60 fps
    - QQCIF (88 x 72): up to 120 fps

    Things to try
    -------------

    Edit <b>JEVOIS:/config/videomappings.cfg</b> on your MicroSD card (see \ref VideoMapping) and try to add some new
    pass-through mappings. Not all of the possible pass-through mappings have been included in the card to avoid having
    too many of these simple "dumb camera" mappings in the base software distribution. For example, you can try

    \verbatim
    YUYV 176 144 115.0 YUYV 176 144 115.0 JeVois PassThrough
    \endverbatim
    
    will grab YUYV frames on the sensor, with resolution 176x144 at 115 frames/s, and will directly send them to the
    host computer over the USB link. To test this mapping, select the corresponding resolution and framerate in your
    video viewing software (here, YUYV 176x144 \@ 115fps). Although the sensor can capture at up to 120fps at this
    resolution, here we used 115fps to avoid a conflict with a mapping using YUYV 176x144 \@ 120fps USB output and the
    \jvmod{SaveVideo} module that is already in the default <b>videomappings.cfg</b> file.

    Note that this module may suffer from DMA coherency artifacts if the \p camturbo parameter of the jevois::Engine is
    turned on, which it is by default. The \p camturbo parameter relaxes some of the cache coherency constraints on the
    video buffers captured by the camera sensor, which allows the JeVois processor to access video pixel data from
    memory faster. But with modules that do not do much processing, sometimes this yields video artifacts, we presume
    because some of the video data from previous frames still is in the CPU cache and hence is not again fetched from
    main memory by the CPU. If you see short stripes of what appears to be wrong pixel colors in the video, try to
    disable \p camturbo, by editing <b>JEVOIS:/config/params.cfg</b> on your MicroSD card and in there turning \p
    camturbo to false.


    @author Laurent Itti

    @videomapping YUYV 1280 1024 7.5 YUYV 1280 1024 7.5 JeVois PassThrough
    @videomapping YUYV 640 480 30.0 YUYV 640 480 30.0 JeVois PassThrough
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

#ifdef JEVOIS_PRO
    //! Processing function with zero-copy and GUI on JeVois-Pro
    virtual void process(jevois::InputFrame && inframe, jevois::GUIhelper & helper) override
    {
      // Start the frame: Internally, this initializes or resizes the display as needed, polls and handles events
      // (inputs, window resize, etc), and clears the frame. Variables winw and winh are set by startFrame() to the
      // current window size, and true is returned if no keyboard/mouse action in a while (can be used to hide any GUI
      // elements when user is not interacting with JeVois):
      unsigned short winw, winh;
      bool idle = helper.startFrame(winw, winh);

      // Test mode: replace frame by checkerboard:
      static bool testmode = false;
      static bool halt = true;
      static bool valt = false;
      static bool noalias = true;
      static bool showscaled = true;
      static bool usedma = false;
      static int scaledx = 0;
      static int scaledy = 0;

      if (testmode)
      {
        jevois::RawImage const & img = inframe.get();
        unsigned char * pix = (unsigned char *)img.pixels<unsigned char>();
        int w = img.width, h = img.height;
        bool oddv = false, oddh = false;

        switch (img.fmt)
        {
        case V4L2_PIX_FMT_RGB24:
        {
          for (int j = 0; j < h; ++j)
          {
            bool hh = valt ? oddv : halt;
            oddh = hh ? false : oddv;
            for (int i = 0; i < w; ++i)
            {
              if (oddh) { *pix++ = 0; *pix++ = 0; *pix++ = 0; }
              else { *pix++ = 255; *pix++ = 255; *pix++ = 255; }
              if (hh) oddh = !oddh;
            }
            oddv = !oddv;
          }
        }
        break;
        case V4L2_PIX_FMT_YUYV:
        {
          for (int j = 0; j < h; ++j)
          {
            bool hh = valt ? oddv : halt;
            oddh = hh ? false : oddv;
            for (int i = 0; i < w; ++i)
            {
              if (oddh) { *pix++ = 0; *pix++ = 0x80; }
              else { *pix++ = 0xff; *pix++ = 0x80; }
              if (hh) oddh = !oddh;
            }
            oddv = !oddv;
          }
        }
        break;
        case V4L2_PIX_FMT_RGB32:
        {
          for (int j = 0; j < h; ++j)
          {
            bool hh = valt ? oddv : halt;
            oddh = hh ? false : oddv;
            for (int i = 0; i < w; ++i)
            {
              if (oddh) { *pix++ = 0; *pix++ = 0; *pix++ = 0; *pix++ = 255; }
              else { *pix++ = 255; *pix++ = 255; *pix++ = 255; *pix++ = 255; }
              if (hh) oddh = !oddh;
            }
            oddv = !oddv;
          }
        }
        break;
        }
      }

      jevois::RawImage const & imgdebug = inframe.get(); // to get dims only
      
      // In the PassThrough module, just draw the camera input frame, as large as we can, which is achieved by passing
      // zero dims. The helper with compute the image position and size to make it as large as possible without changing
      // aspect ratio, and size and position variables will be updated so we know what they are:
      int x = 0, y = 0; unsigned short w = 0, h = 0;
      if (usedma) helper.drawInputFrame("dc", inframe, x, y, w, h, noalias);
      else
      {
        jevois::RawImage const & img = inframe.get();
        helper.drawImage("c", img, x, y, w, h, noalias);
      }
      inframe.done();

      // If we have a second image from the ISP, display it:
      if (inframe.hasScaledImage() && showscaled)
      {
        jevois::RawImage const & img2 = inframe.get2();
        unsigned short w2 = img2.width, h2 = img2.height;
        if (usedma) helper.drawInputFrame2("ds", inframe, scaledx , scaledy, w2, h2, noalias);
        else helper.drawImage("s", img2, scaledx, scaledy, w2, h2, noalias);

        inframe.done2();
      }

      if (idle == false)
      {
        // To draw things on top of input video but behind ImGui windows, use ImGui global background draw list:
        auto dlb = ImGui::GetBackgroundDrawList(); // or use GetForegroundDrawList to draw in front of ImGui
        ImVec2 const p = ImGui::GetMousePos();
        dlb->AddRect(ImVec2(p.x-30, p.y-30), ImVec2(p.x+30, p.y+30), ImColor(255, 0, 0, 255) /* red */);
      
        // Just draw a simple ImGui window that shows fps if we are not idle:
        ImGuiIO & io = ImGui::GetIO();
        ImGui::Begin("JeVois-Pro PassThrough Module");
        ImGui::Text("Framerate: %3.2f fps", io.Framerate);
        ImGui::Text("Video: raw %dx%d aspect=%f, render %dx%d @ %d,%d", imgdebug.width, imgdebug.height,
                    float(imgdebug.width) / float(imgdebug.height), w, h, x, y);
        if (inframe.hasScaledImage())
        {
          ImGui::Checkbox("Show ISP scaled image", &showscaled);
          ImGui::SliderInt("Scaled image x", &scaledx, 0, winw);
          ImGui::SliderInt("Scaled image y", &scaledy, 0, winh);
        }
        
        ImGui::Checkbox("Use DMABUF", &usedma);
        ImGui::Checkbox("Test mode", &testmode);
        if (testmode)
        {
          ImGui::Checkbox("Pattern 1", &halt);
          ImGui::Checkbox("Pattern 2", &valt);
          ImGui::Checkbox("No aliasing", &noalias);
        }

        ImGui::End();


        // To draw things on top of input video and on top of ImGui windows, use ImGui global foregound draw list:
        auto dlf = ImGui::GetForegroundDrawList();
        dlf->AddCircle(ImVec2(p.x, p.y), 20, ImColor(0, 255, 0, 128) /* semi transparent green */);
      }

      // Render the image and GUI:
      helper.endFrame();
    }
#endif // JEVOIS_PRO
};

// Allow the module to be loaded as a shared object (.so) file:
JEVOIS_REGISTER_MODULE(PassThrough);
