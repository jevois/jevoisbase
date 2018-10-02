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
#include <jevoisbase/Components/EyeTracker/EyeTracker.H>
#include <jevois/Image/RawImageOps.H>
#include <jevois/Debug/Timer.H>

#include <linux/videodev2.h> // for v4l2 pixel types
#include <opencv2/imgproc/imgproc.hpp>

// icon by Icomoon in medical at flaticon

//! Pupil detection and eye-tracker using the openEyes toolkit
/*! This module implements an eye tracker, which is based on detecting the outline of the pupil.

    Note that the camera has to be very close to the eye for this to work well. To be useful in practice, some sort of
    prism or tele-lens should be used so that the camera can be out of the field of view of the human participant.

    This module can easily run at 120 frames/s.

    The original eye tracking software used here can be found at http://thirtysixthspan.com/openEyes/software.html
    
    This module only performs the required image processing. To build a complete eye tracking system, additional
    software is required (which can be provided by several open-source eye-tracking toolkits):

    - Establish a calibration procedure, whereby the participant will be asked to look at small dots at known locations
      on the computer screen. This will allow a comparison between coordinates of the pupil center as reported by JeVois
      and coordinates on the computer screen.

    - Compute the calibration transform (mapping from pupil coordinates reported by JeVois to screen coordinates) based
      on the data obtained by the calibration procedure. This has to be done once for each participant and eye-tracking
      session, or every time a participant moves her/his head by a large amount.

    - On new eye-tracking sessions, convert pupil coordinates reported by JeVois to screen coordinates.

    - If desired, extract events in the raw calibrated data stream, such as fixations, rapid saccadic eye movements,
      smooth pursuit movements, and blinks.


    @author Laurent Itti

    @videomapping GREY 640 480 30.0 YUYV 640 480 30.0 JeVois DemoEyeTracker
    @videomapping GREY 320 240 60.0 YUYV 320 240 60.0 JeVois DemoEyeTracker
    @videomapping GREY 176 144 120.0 YUYV 176 144 120.0 JeVois DemoEyeTracker
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
class DemoEyeTracker : public jevois::Module
{
  public:
    
    //! Constructor
    DemoEyeTracker(std::string const & instance) : jevois::Module(instance)
    { itsEyeTracker = addSubComponent<EyeTracker>("eyetracker"); }

    //! Virtual destructor for safe inheritance
    virtual ~DemoEyeTracker() { }

    //! Processing function
    virtual void process(jevois::InputFrame && inframe, jevois::OutputFrame && outframe) override
    {
      static jevois::Timer timer("processing");
      static double pupell[5]; // pupil ellipse data
      
      // Wait for next available camera image:
      jevois::RawImage const inimg = inframe.get(true);
      
      // Wait for an image from our gadget driver into which we will put our results:
      jevois::RawImage outimg = outframe.get();

      // Enforce that the input and output dims match, input should be YUYV, output gray:
      inimg.require("input", outimg.width, outimg.height, V4L2_PIX_FMT_YUYV);
      outimg.require("output", inimg.width, inimg.height, V4L2_PIX_FMT_GREY);

      timer.start();

      // Convert the input to grayscale, directly into the output's pixel buffer:
      cv::Mat cvin = jevois::rawimage::cvImage(inimg);
      cv::Mat cvout = jevois::rawimage::cvImage(outimg);
      cv::cvtColor(cvin, cvout, cv::COLOR_YUV2GRAY_YUYV);

      // Let camera know we are done processing the input image:
      inframe.done();

      // Process through the eye tracker, it wil draw a bunch of things into the output image:
      itsEyeTracker->process(cvout, pupell);

      // Show processing fps:
      std::string const & fpscpu = timer.stop();
      jevois::rawimage::writeText(outimg, fpscpu, 3, outimg.height - 13, 255);
      jevois::rawimage::writeText(outimg, "JeVois EyeTracker", 3, 3, 255);
 
      // Send the output image with our processing results to the host over USB:
      outframe.send();
    }

  protected:
    std::shared_ptr<EyeTracker> itsEyeTracker;
};

// Allow the module to be loaded as a shared object (.so) file:
JEVOIS_REGISTER_MODULE(DemoEyeTracker);
