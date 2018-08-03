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
#include <jevois/Debug/Profiler.H>
#include <jevois/Image/RawImageOps.H>
#include <jevois/Image/ColorConversion.h>
#include <jevoisbase/Components/Saliency/Saliency.H>
#include <jevoisbase/Components/Tracking/Kalman2D.H>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <future>
#include <linux/videodev2.h> // for v4l2 pixel types

// icon by Freepik in other at flaticon

//! Simple saliency map and gist computation module
/*! Computes a saliency map and gist, intended for use by machines.

    See the \jvmod{DemoSaliency} modute for more explanations about saliency and gist algorithms and for a demo output
    intended for human viewing.

    What is returned depends on the selected output image resolution; it should always be grayscale, and can contain any
    of:

    - saliency map only
    - saliency + gist
    - saliency + feature maps
    - saliency + feature maps + gist

    See the example video mappings provided with this module for sample resolutions and the associated elements that are
    returned.

    Serial Messages
    ---------------

    This module can send standardized serial messages as described in \ref UserSerialStyle. One message is issued on
    every video frame at the temporally filtered attended location. The \p id field in the messages simply is \b salient
    for all messages.

    See \ref UserSerialStyle for more on standardized serial messages, and \ref coordhelpers for more info on
    standardized coordinates.


    @author Laurent Itti

    @videomapping GREY 120 25 60.0 YUYV 320 240 60.0 JeVois SaliencyGist # saliency + feature maps + gist
    @videomapping GREY 120 15 60.0 YUYV 320 240 60.0 JeVois SaliencyGist # saliency + feature maps
    @videomapping GREY 20 73 60.0 YUYV 320 240 60.0 JeVois SaliencyGist # saliency + gist
    @videomapping GREY 20 15 60.0 YUYV 320 240 60.0 JeVois SaliencyGist # saliency only
    @videomapping GREY 72 16 60.0 YUYV 320 240 60.0 JeVois SaliencyGist # gist only
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
class SaliencyGist : public jevois::StdModule
{
  public:
    //! Constructor
    SaliencyGist(std::string const & instance) : jevois::StdModule(instance)
    {
      itsSaliency = addSubComponent<Saliency>("saliency");
      itsKF = addSubComponent<Kalman2D>("kalman");
    }

    //! Virtual destructor for safe inheritance
    virtual ~SaliencyGist() { }

    //! Processing function with video output
    virtual void process(jevois::InputFrame && inframe, jevois::OutputFrame && outframe) override
    {
      // Wait for next available camera image:
      jevois::RawImage inimg = inframe.get(); unsigned int const w = inimg.width, h = inimg.height;
      inimg.require("input", w, h, V4L2_PIX_FMT_YUYV); // accept any image size but require YUYV pixels

      // Launch the saliency computation in a thread:
      auto sal_fut = std::async(std::launch::async, [&](){ itsSaliency->process(inimg, true); });
      
      // While computing, wait for an image from our gadget driver into which we will put our results:
      jevois::RawImage outimg = outframe.get();

      // Enforce the correct output image pixel format (will enforce size later):
      unsigned int const ow = outimg.width, oh = outimg.height;
      outimg.require("output", ow, oh, V4L2_PIX_FMT_GREY);

      // Once saliency is done using the input image, let camera know we are done with it:
      itsSaliency->waitUntilDoneWithInput();
      inframe.done();
      
      // Wait until saliency computation is complete:
      sal_fut.get();

      // Find most salient point:
      int mx, my; intg32 msal; itsSaliency->getSaliencyMax(mx, my, msal);
      
      // Compute attended location in original frame coordinates:
      int const smlev = itsSaliency->smscale::get();
      int const smadj = smlev > 0 ? (1 << (smlev-1)) : 0; // half a saliency map pixel adjustment
      unsigned int const dmx = (mx << smlev) + smadj;
      unsigned int const dmy = (my << smlev) + smadj;

      // Filter these locations:
      itsKF->set(dmx, dmy, w, h);
      float kfxraw, kfyraw; itsKF->get(kfxraw, kfyraw, 1.0F); // round to int for serial
      
      // Send kalman-filtered most-salient-point info to serial port (for arduino, etc):
      sendSerialStd2D(kfxraw, kfyraw, 0.0F, 0.0F, "salient");

      // Paste results into the output image, first check for valid output dims:
      unsigned int const mapw = itsSaliency->salmap.dims.w, maph = itsSaliency->salmap.dims.h;
      unsigned int const gistonlyw = 72; unsigned int const gistsize = itsSaliency->gist_size;
      unsigned int gisth = (gistsize + ow - 1) / ow; // divide gist_size by ow and round up

      if (false == ( (ow == mapw && oh == maph) ||
                     (ow == mapw * 6 && oh == maph) ||
                     (ow == gistonlyw && oh == gistsize / gistonlyw) ||
                     (ow == mapw && oh == maph + (gistsize + mapw - 1) / mapw) ||
                     (ow == mapw * 6 && oh ==  maph + (gistsize + mapw*6 - 1) / (mapw*6)) ) )
           LFATAL("Incorrect output size. With current saliency parameters, valid sizes are: " <<
                  mapw << 'x' << maph << " (saliency map only), " <<
                  mapw * 6 << 'x' << maph << " (saliency map + feature maps), " <<
                  gistonlyw << 'x' << gistsize / gistonlyw << " (gist only), " <<
                  mapw << 'x' << maph + (gistsize + mapw - 1) / mapw << " (saliency map + gist), " <<
                  mapw * 6 << 'x' << maph + (gistsize + mapw*6 - 1) / (mapw*6) << " (saliency + feature maps + gist).");

      // Paste saliency and feature maps if desired:
      unsigned int offset = 0;
      if (oh == maph || oh == maph + gisth)
      {
        pasteGrayMap(outimg, itsSaliency->salmap, offset, 20);
        if (ow == mapw * 6)
        {
          pasteGrayMap(outimg, itsSaliency->color, offset, 18);
          pasteGrayMap(outimg, itsSaliency->intens, offset, 18);
          pasteGrayMap(outimg, itsSaliency->ori, offset, 18);
          pasteGrayMap(outimg, itsSaliency->flicker, offset, 18);
          pasteGrayMap(outimg, itsSaliency->motion, offset, 18);
        }
      }

      // Paste gist if desired:
      if (oh == gisth || oh == maph + gisth)
      {
        unsigned char * d = outimg.pixelsw<unsigned char>(); if (oh == maph + gisth) d += ow * maph;
        memcpy(d, itsSaliency->gist, gistsize);

        // Possibly blank out the remainder of the last line of gist:
        int const rem = ow * gisth - gistsize;
        if (rem > 0) memset(d + gistsize, 0, rem);
      }
      
      // Send the output image with our processing results to the host over USB:
      outframe.send();
    }

    // ####################################################################################################
    //! Paste a map and add its width to the dx offset
    /*! Beware this is for a gray outimg only. */
    void pasteGrayMap(jevois::RawImage & outimg, env_image const & fmap, unsigned int & dx, unsigned int bitshift)
    {
      env_size_t const fw = fmap.dims.w, fh = fmap.dims.h;
      unsigned int const ow = outimg.width, oh = outimg.height;
      
      if (fh > oh) LFATAL("Map would extend past output image bottom");
      if (fw + dx > ow) LFATAL("Map would extend past output image width");
      
      unsigned int const stride = ow - fw;
      
      intg32 * s = fmap.pixels; unsigned char * d = outimg.pixelsw<unsigned char>() + dx;
      
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
    
  protected:
    std::shared_ptr<Saliency> itsSaliency;
    std::shared_ptr<Kalman2D> itsKF;
};

// Allow the module to be loaded as a shared object (.so) file:
JEVOIS_REGISTER_MODULE(SaliencyGist);
