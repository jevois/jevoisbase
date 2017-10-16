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

#include <jevoisbase/Components/OpticalFlow/FastOpticalFlow.H>

#include <linux/videodev2.h>

// icon by Gregor Cresnar in arrows at flaticon

//! Fast optical flow computation using OF_DIS
/*! Computes horizontal and vertical optical flow into two separate optical flow maps. This module is intended for use
    as a pre-processor, that is, the images sent out may be further processed by the host computer.

    The optical flow image output is twice taller than the camera input image:

    - On top is the horizontal motion map
      + darker shades of gray for things moving rightward from the viewpoint of the camera,
      + lighter shades for things that move leftward,
      + mid-grey level for no motion

    - below it is the vertical motion map
      + dark for things moving downward,
      + bright for upward,
      + mid-grey for no vertical motion.

    You should be able to run this algorithm at 100 frames/s with 176x144 video camera resolution.

    This module has parameters to tune the algorithm. Interested users are referred to the original paper and
    implementation for how to set these:

    "Fast Optical Flow using Dense Inverse Search" by Till Kroeger, Radu Timofte, Dengxin Dai and Luc Van Gool, Proc
    ECCV, 2016.

    Also see here: http://www.vision.ee.ethz.ch/~kroegert/OFlow/

    Trying it out
    -------------

    Have JeVois point towards an otherwise static scene, and swipe one finger in front of it, either moving from left to
    right of the field of view, from top to bottom, etc to confirm the different greyscale values as described above.


    @author Laurent Itti

    @videomapping GREY 176 288 100 YUYV 176 144 100 JeVois OpticalFlow
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
class OpticalFlow : public jevois::Module
{
  public:
    //! Constructor
    OpticalFlow(std::string const & instance) : jevois::Module(instance)
    { itsOpticalFlow = addSubComponent<FastOpticalFlow>("fastflow"); }
    
    //! Virtual destructor for safe inheritance
    virtual ~OpticalFlow() { }

    //! Processing function
    virtual void process(jevois::InputFrame && inframe, jevois::OutputFrame && outframe) override
    {
      // Wait for next available camera image:
      jevois::RawImage inimg = inframe.get(); unsigned int const w = inimg.width, h = inimg.height;

      // Convert to openCV grayscale:
      cv::Mat cvimg = jevois::rawimage::convertToCvGray(inimg);

      // Let camera know we are done processing the input image:
      inframe.done();

      // Get the output image and require same width as input, 2x taller, and grey pixels:
      jevois::RawImage outimg = outframe.get();
      outimg.require("output", w, h * 2, V4L2_PIX_FMT_GREY);

      // Process the input and get output:
      cv::Mat cvout = jevois::rawimage::cvImage(outimg); // cvout's pixels point to outimg's, no copy
      itsOpticalFlow->process(cvimg, cvout);
      
      // Send the output image with our processing results to the host over USB:
      outframe.send();
    }

  private:
    std::shared_ptr<FastOpticalFlow> itsOpticalFlow;
};

// Allow the module to be loaded as a shared object (.so) file:
JEVOIS_REGISTER_MODULE(OpticalFlow);
