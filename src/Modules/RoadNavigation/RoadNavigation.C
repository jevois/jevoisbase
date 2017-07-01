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
#include <jevoisbase/Components/RoadFinder/RoadFinder.H>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <future>
#include <linux/videodev2.h> // for v4l2 pixel types

// icon by Dave Gandy in transport at flaticon

//! Road finder demo
/*! This algorithm detects road using a compination of edge detection and tracking, and texture analysis. The algorithm
    is an implementation of Chang, Siagian and Itti, IROS 2012, available at
    http://ilab.usc.edu/publications/doc/Chang_etal12iros.pdf

    The algorithms combines detection and tracking of line segments at the edges of the road or on the road (e.g., lane
    dividers), and texture analysis to distinguish the road region from its surroundings.

    The algorithm outputs the horizontal coordinate of the vanishing point of the road, which usually is a good
    indication of the road heading (except in very tight bends or corners).

    Demo display outputs
    --------------------

    Detected line segments are shown in black and white, while segments that have been reliably tracked over multiple
    frames are shown in thick purple. Estimated vanishing point location and confidence is shown as a big green disk on
    the horizon line.

    Serial Messages
    ---------------

    This module can send standardized serial messages as described in \ref UserSerialStyle. One 1D message is issued for
    on every video frame at vanishing point horizontal location. The \p id field in the messages simply is \b vp for all
    messages.

    Trying it out
    -------------

    To casually try out this module, just search the web for pictures of roads and point the JeVois camera to one of
    them. Make sure that you align the horizon line of the algorithm (which has a bumber of purple and green disks)
    roughly with the horizon line in your picture. As you move the camera left and right, the location of the large
    green disk that marks the detected vanishing point should move left and right, and should point to the vanishing
    point of the road in your image.

    When using on a mobile robot in th ereal world, setting the proper horizon line is essential for good operation of
    the algorithm. This is determined by parameter \c horizon, which should be tuned according to the height and
    tilt angle of the JeVois camera on your vehicle.

    @author Laurent Itti

    @videomapping NONE 0 0 0 YUYV 320 240 30.0 JeVois RoadNavigation
    @videomapping NONE 0 0 0 YUYV 176 144 120.0 JeVois RoadNavigation
    @videomapping YUYV 320 256 30.0 YUYV 320 240 30.0 JeVois RoadNavigation
    @videomapping YUYV 176 160 120.0 YUYV 176 144 120.0 JeVois RoadNavigation
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
class RoadNavigation : public jevois::Module
{
  public:
    // ####################################################################################################
    //! Constructor
    // ####################################################################################################
    RoadNavigation(std::string const & instance) :
        jevois::Module(instance), itsProcessingTimer("Processing", 30, LOG_DEBUG)
    {
      itsRoadFinder = addSubComponent<RoadFinder>("roadfinder");
    }
    
    // ####################################################################################################
    //! Virtual destructor for safe inheritance
    // ####################################################################################################
    virtual ~RoadNavigation() { }

    // ####################################################################################################
    //! Processing function, no video out
    // ####################################################################################################
    virtual void process(jevois::InputFrame && inframe) override
    {
      // Wait for next available camera image, convert it to gray, and release it:
      cv::Mat imggray = inframe.getCvGray();

      // Compute the vanishing point, with no drawings:
      jevois::RawImage visual; // unallocated pixels, will not draw anything
      itsRoadFinder->process(imggray, visual);
      
      // Get the filtered target point x and send to serial:
      sendSerialImg1Dx(imggray.cols, itsRoadFinder->getFilteredTargetX(), 0.0F, "vp");
    }

    // ####################################################################################################
    //! Processing function with USB video out
    // ####################################################################################################
    virtual void process(jevois::InputFrame && inframe, jevois::OutputFrame && outframe) override
    {
      // Wait for next available camera image:
      jevois::RawImage inimg = inframe.get(); unsigned int const w = inimg.width, h = inimg.height;
      inimg.require("input", w, h, V4L2_PIX_FMT_YUYV); // accept any image size but require YUYV pixels

      itsProcessingTimer.start();
      
      // Convert it to gray:
      cv::Mat imggray = jevois::rawimage::convertToCvGray(inimg);

      // Compute the vanishing point. Note: the results will be drawn into inimg, so that we don't have to wait for
      // outimg to be ready. It's ok to modify the input image, its buffer will be sent back to the camera driver for
      // capture once we are done here, and will be overwritten anyway:
      itsRoadFinder->process(imggray, inimg);
      
      // Wait for an image from our gadget driver into which we will put our results:
      jevois::RawImage outimg = outframe.get();

      // Enforce the correct output image size and format:
      outimg.require("output", w, h + 16, V4L2_PIX_FMT_YUYV);

      // Paste the original image + drawings to the top-left corner of the display:
      unsigned short const txtcol = 0xa0ff; //WHITE: 0x80ff;
      jevois::rawimage::paste(inimg, outimg, 0, 0);
      jevois::rawimage::writeText(outimg, "JeVois Road Navigation Demo", 3, 3, txtcol);
      
      // Let camera know we are done processing the raw YUV input image:
      inframe.done();
      
      // Clear out the bottom section of the image:
      jevois::rawimage::drawFilledRect(outimg, 0, h, w, outimg.height-h, jevois::yuyv::Black);

      // Get the filtered target point x and send to serial:
      sendSerialImg1Dx(w, itsRoadFinder->getFilteredTargetX(), 0.0F, "vp");
      
      // Write some extra info about the vp:
      std::ostringstream otxt; otxt << std::fixed << std::setprecision(3);
      auto vp = itsRoadFinder->getCurrVanishingPoint();
      otxt << "VP x=" << vp.first.i << " (" << vp.second << ") CTR=" << std::setprecision(1);
      auto cp = itsRoadFinder->getCurrCenterPoint();
      auto tp = itsRoadFinder->getCurrTargetPoint();
      otxt << cp.i << " TGT=" << tp.i << " fTPX=" << tpx;
      jevois::rawimage::writeText(outimg, otxt.str().c_str(), 3, h + 3, jevois::yuyv::White);

      // Show processing fps:
      std::string const & fpscpu = itsProcessingTimer.stop();
      jevois::rawimage::writeText(outimg, fpscpu, 3, h - 13, jevois::yuyv::White);
      
      // Send the output image with our processing results to the host over USB:
      outframe.send();
    }

    // ####################################################################################################
    //! Module internals
    // ####################################################################################################
  protected:
    jevois::Timer itsProcessingTimer;
    std::shared_ptr<RoadFinder> itsRoadFinder;
};

// Allow the module to be loaded as a shared object (.so) file:
JEVOIS_REGISTER_MODULE(RoadNavigation);
