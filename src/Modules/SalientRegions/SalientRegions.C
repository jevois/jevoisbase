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

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <future>
#include <linux/videodev2.h> // for v4l2 pixel types

// icon by Freepik in other at flaticon

// Module parameters:
static jevois::ParameterCategory const ParamCateg("Salient Regions Options");

//! Parameter \relates SalientRegions
JEVOIS_DECLARE_PARAMETER(inhsigma, float, "Sigma (pixels) used for inhibition of return", 32.0F, ParamCateg);

//! Extract the most salient regions and send them out over USB
/*! This module extracts cropped images around the N most salient (i.e., conspicuous, or attention-grabbing) locations
    in the JeVois camera's field of view. These regions are then streamed over USB, one of top of another.  This module
    thus produces an output mainly intended for machine use: a host computer might grab the regions detected by JeVois,
    and run, for example, some object recognition algorithm on each region.

    See \jvmod{DemoSaliency} for more information about visual attention and saliency.

    The number of regions extracted (N) is decided by the height of the desired USB output image, while the (square)
    region size (width and height) is determined by the output image width. Note that region width and height must be a
    multiple of 4. ALso note that Mac computers may not be able to grab and display video of width that is not a
    multiple of 16.

    The most salient region is on top, the second most salient region is below the first one, etc.

    Example use
    -----------

    With video mapping

    \verbatim
    YUYV 64 192 25.0 YUYV 320 240 25.0 JeVois SalientRegions
    \endverbatim

    in <b>JEVOIS:/config/videomappings.cfg</b>, this module will extract three 64x64 salient regions, and will send them
    over USB one on top of the other (since USB video width is 64, which determines region size, and USB video height is
    3x64 = 192, which determines the number of regions).


    @author Laurent Itti

    @videomapping YUYV 64 192 25.0 YUYV 320 240 25.0 JeVois SalientRegions
    @videomapping YUYV 100 400 10.0 YUYV 640 480 10.0 JeVois SalientRegions
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
class SalientRegions : public jevois::Module,
                       public jevois::Parameter<inhsigma>
{
  public:
    //! Constructor
    SalientRegions(std::string const & instance) : jevois::Module(instance)
    { itsSaliency = addSubComponent<Saliency>("saliency"); }

    //! Virtual destructor for safe inheritance
    virtual ~SalientRegions() { }

    //! Processing function
    virtual void process(jevois::InputFrame && inframe, jevois::OutputFrame && outframe) override
    {
      // Wait for next available camera image:
      jevois::RawImage inimg = inframe.get(); unsigned int const w = inimg.width, h = inimg.height;
      inimg.require("input", w, h, V4L2_PIX_FMT_YUYV); // accept any image size but require YUYV pixels
      
      // Compute the saliency map, no gist:
      itsSaliency->process(inimg, false);

      // Wait for an image from our gadget driver into which we will put our results:
      jevois::RawImage outimg = outframe.get();
      int const rwh = outimg.width & (~3); // output image width sets region width and height, must be multiple of 4
      int const nr = outimg.height / rwh; // output image height sets the number of regions
      outimg.require("output", rwh, nr * rwh, V4L2_PIX_FMT_YUYV);

      // Get some info from the saliency computation:
      int const smlev = itsSaliency->smscale::get();
      int const smfac = (1 << smlev);

      // Copy each region to output:
      for (int i = 0; i < nr; ++i)
      {
        // Find most salient point:
        int mx, my; intg32 msal; itsSaliency->getSaliencyMax(mx, my, msal);
      
        // Compute attended ROI (note: coords must be even to avoid flipping U/V when we later paste):
        unsigned int const dmx = (mx << smlev) + (smfac >> 2);
        unsigned int const dmy = (my << smlev) + (smfac >> 2);
        int rx = (std::min(int(w) - rwh/2, std::max(rwh/2, int(dmx + 1 + (smfac >> 2))))) & (~1);
        int ry = (std::min(int(h) - rwh/2, std::max(rwh/2, int(dmy + 1 + (smfac >> 2))))) & (~1);

        // Paste the roi:
        jevois::rawimage::roipaste(inimg, rx - rwh/2, ry - rwh/2, rwh, rwh, outimg, 0, i * rwh);

        // Inhibit this salient location so we move to the next one:
        itsSaliency->inhibitionOfReturn(mx, my, inhsigma::get() / smfac);
      }

      // Let camera know we are done processing the raw YUV input image:
      inframe.done();
      
      // Send the output image with our processing results to the host over USB:
      outframe.send();
    }

  protected:
    std::shared_ptr<Saliency> itsSaliency;
};

// Allow the module to be loaded as a shared object (.so) file:
JEVOIS_REGISTER_MODULE(SalientRegions);
