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
// Contact information: Shixian Wen - 3641 Watt Way, HNB-10A - Los Angeles, BA 90089-2520 - USA.
// Tel: +1 213 740 3527 - shixianw@usc.edu - http://iLab.usc.edu - http://jevois.org
// ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/*! \file */

#include <jevois/Core/Module.H>
#include <jevois/Debug/Timer.H>
#include <jevois/Image/RawImageOps.H>
#include <jevois/Util/Coordinates.H>
#include <jevoisbase/Components/ARtoolkit/ARtoolkit.H>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

//! Augmented reality markers using ARtoolkit
/*! Detect and decode patterns known as ARtoolkit markers, which are small 2D barcodes often used in augmented
    reality and robotics. See https://archive.artoolkit.org/ - This module uses version 5.x for now.

    This algorithm is very fast. It runs at 400+ frames/s at 320x240 and can easily sustain the maximum 15 frames/s of
    the JeVois camera sensor chip at 1280x1024. It may be interesting to combine it with other algorithms running on the
    quad-core JeVois processor (see for example the \jvmod{MarkersCombo} module).

    Sample patterns
    ---------------

    This module uses by default the dictionary of 3x3 patterns with parity (32 different patterns). You can download
    them from http://jevois.org/data/ARtoolkit3x3par.zip

    Also see the files in <b>jevoisbase/Contrib/ARToolKit5arm/doc/patterns/</b> for more.

    Serial Messages
    ---------------

    This module can send standardized serial messages as described in \ref UserSerialStyle.

    When \p msg3d is turned on, 3D messages will be sent, otherwise 2D messages.

    One message is issued for every detected marker, on every video frame.

    2D messages when \p msg3d is off:

    - Serial message type: \b 2D
    - `id`: decoded ARtoolkit marker ID, with a prefix 'A'
    - `x`, `y`, or vertices: standardized 2D coordinates of marker center or corners
    - `w`, `h`: standardized marker size
    - `extra`: none (empty string)

    3D messages when \p msg3d is on:

    - Serial message type: \b 3D
    - `id`: decoded ARtoolkit marker ID, with a prefix 'A'
    - `x`, `y`, `z`, or vertices: 3D coordinates in millimeters of marker center or corners
    - `w`, `h`, `d`: marker size in millimeters, a depth of 1mm is always used
    - `extra`: none (empty string)

    If you will use the quaternion data (Detail message style; see \ref UserSerialStyle), you should probably set the \p
    serprec parameter to something non-zero to get enough accuracy in the quaternion values.

    See \ref UserSerialStyle for more on standardized serial messages, and \ref coordhelpers for more info on
    standardized coordinates.


    @author Shixian Wen

    @displayname Demo ARtoolkit
    @videomapping NONE 0 0 0 YUYV 320 240 60.0 JeVois DemoARtoolkit
    @videomapping NONE 0 0 0 YUYV 640 480 30.0 JeVois DemoARtoolkit
    @videomapping NONE 0 0 0 YUYV 1280 1024 15.0 JeVois DemoARtoolkit
    @videomapping YUYV 320 258 60.0 YUYV 320 240 60.0 JeVois DemoARtoolkit
    @videomapping YUYV 640 498 30.0 YUYV 640 480 30.0 JeVois DemoARtoolkit
    @email shixianw\@usc.edu
    @address University of Southern California, HNB-10A, 3641 Watt Way, Los Angeles, CA 90089-2520, USA
    @copyright Copyright (C) 2017 by Shixian Wen, iLab and the University of Southern California
    @mainurl http://jevois.org
    @supporturl http://jevois.org/doc
    @otherurl http://iLab.usc.edu
    @license GPL v3
    @distribution Unrestricted
    @restrictions None
    \ingroup modules */
class DemoARtoolkit :public jevois::StdModule
{
  public:
    // ####################################################################################################
    DemoARtoolkit(std::string const & instance) : jevois::StdModule(instance)
    {
      itsARtoolkit = addSubComponent<ARtoolkit>("artoolkit");
    }

    // ####################################################################################################
    virtual ~DemoARtoolkit()
    { }

    // ####################################################################################################
    virtual void process(jevois::InputFrame && inframe) override
    {
      static jevois::Timer timer("processing");
      
      // Wait for next available camera image:
      jevois::RawImage const inimg = inframe.get();
      timer.start();
      
      // ARtoolkit component can process YUYV images directly with no conversion required:
      itsARtoolkit->detectMarkers(inimg);
      
      // We are done with the input frame:
      inframe.done();

      // Send serial messages:
      itsARtoolkit->sendSerial(this);
      timer.stop();
    }
    
    // ####################################################################################################
    virtual void process(jevois::InputFrame && inframe, jevois::OutputFrame && outframe) override
    {
      static jevois::Timer timer("processing", 100, LOG_DEBUG);

      // Wait for next available camera image:
      jevois::RawImage const inimg = inframe.get(); unsigned int const w = inimg.width, h = inimg.height;
      timer.start();
      
      // While we process it, start a thread to wait for out frame and paste the input into it:
      jevois::RawImage outimg;
      auto paste_fut = std::async(std::launch::async, [&]() {
          outimg = outframe.get();
          outimg.require("output", w, h + 18, inimg.fmt);
          jevois::rawimage::paste(inimg, outimg, 0, 0);
          jevois::rawimage::writeText(outimg, "JeVois ARtoolkit Demo", 3, 3, jevois::yuyv::White);
          jevois::rawimage::drawFilledRect(outimg, 0, h, w, outimg.height - h, jevois::yuyv::Black);
        });

      // ARtoolkit component can process YUYV images directly with no conversion required:
      itsARtoolkit->detectMarkers(inimg);
      
      // Wait for paste to finish up:
      paste_fut.get();

      // We are done with the input frame:
      inframe.done();

      // Draw the detections & send serial messages:
      itsARtoolkit->drawDetections(outimg, 3, h + 3);
      itsARtoolkit->sendSerial(this);
      
      // Show processing fps:
      std::string const & fpscpu = timer.stop();
      jevois::rawimage::writeText(outimg, fpscpu, 3, h - 13, jevois::yuyv::White);

      // Send the output image with our processing results to the host over USB:
      outframe.send();
    }

    // ####################################################################################################
  protected:
    std::shared_ptr<ARtoolkit> itsARtoolkit;
};

// Allow the module to be loaded as a shared object (.so) file:
JEVOIS_REGISTER_MODULE(DemoARtoolkit);
