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
// Contact information: Laurent Itti - 3641 Watt Way, HNB-07A - Los Angeles, BA 90089-2520 - USA.
// Tel: +1 213 740 3527 - itti@pollux.usc.edu - http://iLab.usc.edu - http://jevois.org
// ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/*! \file */

#include <jevois/Core/Module.H>
#include <jevois/Debug/Timer.H>
#include <jevois/Image/RawImageOps.H>
#include <jevoisbase/Components/ArUco/ArUco.H>
#include <jevoisbase/Components/QRcode/QRcode.H>
#include <jevoisbase/Components/ARtoolkit/ARtoolkit.H>
#include <opencv2/core/core.hpp>

//! Simple demo of QRcode + ARtoolkit + ArUco markers detection and decoding
/*! Detect and decode 3 kinds of coded patterns which can be useful to a robot:

    - QR-codes as in \jvmod{DemoQRcode}
    - AR-toolkit markers as in \jvmod{DemoARtoolkit}
    - ArUco markers as in \jvmod{DemoArUco}

    The three algorithms run in parallel. You should be able to sustain 50 frames/s at 320x240 camera resolution, and 20
    frames/s at 640x480 camera resolution.

    Serial Messages
    ---------------

    This module can send standardized serial messages as described in \ref UserSerialStyle.

    When \p dopose is turned on, 3D messages will be sent, otherwise 2D messages.

    One message is issued for every detected marker, on every video frame.

    2D messages when \p msg3d and \p dopose are off:

    - Serial message type: \b 2D
    - `id`: decoded marker ID (with prefix 'U' for ArUco, or 'A' for ARtoolkit),
       or type of symbol (e.g., \b QR-Code, \b ISBN13, etc).
    - `x`, `y`, or vertices: standardized 2D coordinates of marker center or corners
    - `w`, `h`: standardized marker size
    - `extra`: none (empty string) for ArUco and ARtoolkit markers, otherwise decoded barcode or QRcode content.

    3D messages when \p msg3d and \p dopose are on:
 
    - Serial message type: \b 3D
    - `id`: decoded marker ID (with prefix 'U' for ArUco, or 'A' for ARtoolkit),
       or type of symbol (e.g., \b QR-Code, \b ISBN13, etc).
    - `x`, `y`, `z`, or vertices: 3D coordinates in millimeters of marker center or corners
    - `w`, `h`, `d`: marker size in millimeters, a depth of 1mm is always used
    - `extra`: none (empty string) for ArUco and ARtoolkit markers, otherwise decoded barcode or QRcode content.

    If you will use the quaternion data (Detail message style; see \ref UserSerialStyle), you should probably set the \p
    serprec parameter to something non-zero to get enough accuracy in the quaternion values.

    See \ref UserSerialStyle for more on standardized serial messages, and \ref coordhelpers for more info on
    standardized coordinates.


    @author Laurent Itti

    @videomapping NONE 0 0 0 YUYV 320 240 30.0 JeVois MarkersCombo
    @videomapping YUYV 320 306 50.0 YUYV 320 240 50.0 JeVois MarkersCombo
    @videomapping YUYV 640 546 20.0 YUYV 640 480 20.0 JeVois MarkersCombo
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
class MarkersCombo : public jevois::StdModule
{
  public: 
    // ####################################################################################################
    //! Constructor
    // ####################################################################################################
    MarkersCombo(std::string const & instance) : jevois::StdModule(instance)
    {
      itsArUco = addSubComponent<ArUco>("aruco");
      itsQRcode = addSubComponent<QRcode>("qrcode");
      itsARtoolkit = addSubComponent<ARtoolkit>("artoolkit");
    }

    // ####################################################################################################
    //! Virtual destructor for safe inheritance
    // ####################################################################################################
    virtual ~MarkersCombo()
    { }

    // ####################################################################################################
    //! Processing function, no video output
    // ####################################################################################################
    virtual void process(jevois::InputFrame && inframe) override
    {
      // Wait for next available camera image as grayscale:
      cv::Mat cvimg = inframe.getCvGRAY();

      // Launch QRcode:
      auto qr_fut = std::async(std::launch::async, [&]() {
          zbar::Image zgray(cvimg.cols, cvimg.rows, "Y800", cvimg.data, cvimg.total());
          itsQRcode->process(zgray);
          itsQRcode->sendSerial(this, zgray, cvimg.cols, cvimg.rows);
          zgray.set_data(nullptr, 0);
        });

      // Launch AR toolkit:
      auto ar_fut = std::async(std::launch::async, [&]() {
          itsARtoolkit->detectMarkers(cvimg);
          itsARtoolkit->sendSerial(this);
        });
      
      // Process ArUco in the main thread:
      std::vector<int> ids; std::vector<std::vector<cv::Point2f> > corners; std::vector<cv::Vec3d> rvecs, tvecs;
      itsArUco->detectMarkers(cvimg, ids, corners);
      if (itsArUco->dopose::get() && ids.empty() == false) itsArUco->estimatePoseSingleMarkers(corners, rvecs, tvecs);
      itsArUco->sendSerial(this, ids, corners, cvimg.cols, cvimg.rows, rvecs, tvecs);
    }

    // ####################################################################################################
    //! Processing function with video output to USB
    // ####################################################################################################
    virtual void process(jevois::InputFrame && inframe, jevois::OutputFrame && outframe) override
    {
      static jevois::Timer timer("processing", 100, LOG_DEBUG);

      // Wait for next available camera image:
      jevois::RawImage const inimg = inframe.get();

      timer.start();
      
      // We only handle one specific pixel format, any size in this demo:
      unsigned int const w = inimg.width, h = inimg.height;
      inimg.require("input", w, h, V4L2_PIX_FMT_YUYV);

      // While we process it, start a thread to wait for out frame and paste the input into it. It will hold a unique
      // lock onto mtx. Other threads should attempt a shared_lock onto mtx before they draw into outimg:
      jevois::RawImage outimg; //boost::shared_mutex mtx; boost::unique_lock<boost::shared_mutex> ulck(mtx);
      auto paste_fut = std::async(std::launch::async, [&]() {
          outimg = outframe.get();
          outimg.require("output", w, h + 66, inimg.fmt);
          jevois::rawimage::paste(inimg, outimg, 0, 0);
          //ulck.unlock();
          jevois::rawimage::writeText(outimg, "JeVois Makers Combo", 3, 3, jevois::yuyv::White);
          jevois::rawimage::drawFilledRect(outimg, 0, h, w, outimg.height-h, jevois::yuyv::Black);
          inframe.done();
        });

      // Convert the image to grayscale:
      cv::Mat cvimg = jevois::rawimage::convertToCvGray(inimg);

      // Launch QRcode:
      auto qr_fut = std::async(std::launch::async, [&]() {
          zbar::Image zgray(cvimg.cols, cvimg.rows, "Y800", cvimg.data, cvimg.total());
          itsQRcode->process(zgray);
          itsQRcode->sendSerial(this, zgray, w, h);
          //boost::shared_lock<boost::shared_mutex> _(mtx); // wait until paste is complete
          itsQRcode->drawDetections(outimg, 3, h + 23, zgray, w, h, 3);
          zgray.set_data(nullptr, 0);
        });

      // Launch AR toolkit:
      auto ar_fut = std::async(std::launch::async, [&]() {
          itsARtoolkit->detectMarkers(cvimg);
          itsARtoolkit->sendSerial(this);
          //boost::shared_lock<boost::shared_mutex> _(mtx);
          itsARtoolkit->drawDetections(outimg, 3, h + 13); // wait until paste is complete
        });
      
      // Process ArUco in the main thread:
      std::vector<int> ids; std::vector<std::vector<cv::Point2f> > corners; std::vector<cv::Vec3d> rvecs, tvecs;
      itsArUco->detectMarkers(cvimg, ids, corners);
      if (itsArUco->dopose::get() && ids.empty() == false) itsArUco->estimatePoseSingleMarkers(corners, rvecs, tvecs);

      // Show aruco results and send serial:
      itsArUco->sendSerial(this, ids, corners, w, h, rvecs, tvecs);
      //boost::shared_lock<boost::shared_mutex> _(mtx);
      itsArUco->drawDetections(outimg, 3, h + 3, ids, corners, rvecs, tvecs);

      // Show processing fps:
      std::string const & fpscpu = timer.stop();
      jevois::rawimage::writeText(outimg, fpscpu, 3, h - 13, jevois::yuyv::White);
    
      // Wait for all threads to finish up:
      qr_fut.get();
      ar_fut.get();
      paste_fut.get();
      
      // Send the output image with our processing results to the host over USB:
      outframe.send();
    }

    // ####################################################################################################
  protected:
    std::shared_ptr<ArUco> itsArUco;
    std::shared_ptr<QRcode> itsQRcode;
    std::shared_ptr<ARtoolkit> itsARtoolkit;
};

// Allow the module to be loaded as a shared object (.so) file:
JEVOIS_REGISTER_MODULE(MarkersCombo);
