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
#include <jevois/Debug/Timer.H>
#include <jevois/Image/RawImageOps.H>
#include <jevoisbase/Components/QRcode/QRcode.H>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

// icon by Vaadin in shapes at flaticon

//! Simple demo of QR-code and barcode detection and decoding using the ZBar library
/*! Detect barcodes and QR-codes, and decode their contents.

    QR-codes (Quick Response Codes) are popular 2D patterns that contain embedded information, such as a string of text,
    a URL, etc. They basically work like barcodes, coding information into a high-contrast, geometric pattern that is
    easier to detect and decode by a machine that more conventional human-written text or drawings.

    One can generate QR-codes containing different kinds of information, for example using online QR-code generators,
    such as http://www.qr-code-generator.com/

    JeVois detects and decodes QR-codes and other barcodes. The implementation of the detection and decoding algorithm
    used in JeVois is from the popular library ZBar, found at http://zbar.sourceforge.net/

    You should be able to sustain 30 frames/s with camera resolution 320x240, and 15 frames/s with camera resolution
    640x480 when running this module inside the JeVois smart camera.

    Serial Messages
    ---------------

    This module can send standardized serial messages as described in \ref UserSerialStyle. One message is issued for
    every detected QR-code or barcode, on every video frame.

    - Serial message type: \b 2D
    - `id`: type of symbol (e.g., \b QR-Code, \b ISBN13, etc).
    - `x`, `y`, or vertices: standardized 2D coordinates of symbol center or of corners of bounding box
      (depending on \p serstyle)
    - `w`, `h`: standardized object size
    - `extra`: decoded contents (e.g., URL that was in a QR-code, ISBN number from a barcode, etc)

    Note that when \p serstyle is \b Fine, only 4 corners are returned for each detected QR-code, but many points are
    returned all over each detected barcode. Beware to not exceed your serial bandwidth in that case.

    See \ref UserSerialStyle for more on standardized serial messages, and \ref coordhelpers for more info on
    standardized coordinates.

    @author Laurent Itti

    @displayname Demo QR-code
    @videomapping YUYV 640 526 15.0 YUYV 640 480 15.0 JeVois DemoQRcode
    @videomapping YUYV 320 286 30.0 YUYV 320 240 30.0 JeVois DemoQRcode
    @videomapping NONE 0 0 0 YUYV 640 480 15.0 JeVois DemoQRcode
    @videomapping NONE 0 0 0 YUYV 320 240 30.0 JeVois DemoQRcode
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
class DemoQRcode : public jevois::StdModule
{
  public:
    
    // ####################################################################################################
    //! Constructor
    // ####################################################################################################
    DemoQRcode(std::string const & instance) : jevois::StdModule(instance)
    { itsQRcode = addSubComponent<QRcode>("qrcode"); }

    // ####################################################################################################
    //! Virtual destructor for safe inheritance
    // ####################################################################################################
    virtual ~DemoQRcode()
    { }

    // ####################################################################################################
    //! Processing function, no video output
    // ####################################################################################################
    virtual void process(jevois::InputFrame && inframe) override
    {
      // Wait for next available camera image:
      jevois::RawImage const inimg = inframe.get();
      unsigned int const w = inimg.width, h = inimg.height;

      // Convert the image to grayscale:
      cv::Mat grayimg = jevois::rawimage::convertToCvGray(inimg);

      // Let camera know we are done processing the input image:
      inframe.done();

      // Process gray image through zbar:
      zbar::Image zgray(grayimg.cols, grayimg.rows, "Y800", grayimg.data, grayimg.total());
      itsQRcode->process(zgray);

      // Send all the results over serial:
      itsQRcode->sendSerial(this, zgray, w, h);

      // Cleanup zbar image data:
      zgray.set_data(nullptr, 0);
    }
    
    // ####################################################################################################
    //! Processing function with video output to USB
    // ####################################################################################################
    virtual void process(jevois::InputFrame && inframe, jevois::OutputFrame && outframe) override
    {
      static jevois::Timer timer("processing", 60, LOG_DEBUG);
      static unsigned short const txtcol = jevois::yuyv::White;
      size_t const nshow = 4; // number of lines to show
      
      // Wait for next available camera image:
      jevois::RawImage const inimg = inframe.get();

      timer.start();
      
      // We only handle one specific pixel format, any size in this demo:
      unsigned int const w = inimg.width, h = inimg.height;
      inimg.require("input", w, h, V4L2_PIX_FMT_YUYV);

      // While we process it, start a thread to wait for out frame and paste the input into it:
      jevois::RawImage outimg;
      auto paste_fut = std::async(std::launch::async, [&]() {
          outimg = outframe.get();
          outimg.require("output", w, h + nshow * 10 + 6, inimg.fmt);
          jevois::rawimage::paste(inimg, outimg, 0, 0);
          jevois::rawimage::writeText(outimg, "JeVois QR-code + Barcode Detection Demo", 3, 3, txtcol);
          jevois::rawimage::drawFilledRect(outimg, 0, h, w, outimg.height-h, 0x8000);
        });

      // Convert the image to grayscale and process it through zbar:
      cv::Mat grayimg = jevois::rawimage::convertToCvGray(inimg);
      zbar::Image zgray(grayimg.cols, grayimg.rows, "Y800", grayimg.data, grayimg.total());
      itsQRcode->process(zgray);

      // Wait for paste to finish up:
      paste_fut.get();

      // Let camera know we are done processing the input image:
      inframe.done();

      // Draw all detections:
      itsQRcode->drawDetections(outimg, 3, h + 3, zgray, w, h, nshow);
      
      // Send all serial messages:
      itsQRcode->sendSerial(this, zgray, w, h);
      
      // Cleanup zbar image data:
      zgray.set_data(nullptr, 0);

      // Show processing fps:
      std::string const & fpscpu = timer.stop();
      jevois::rawimage::writeText(outimg, fpscpu, 3, h - 13, txtcol);
    
      // Send the output image with our processing results to the host over USB:
      outframe.send();
    }

    // ####################################################################################################
  protected:
    std::shared_ptr<QRcode> itsQRcode;
};

// Allow the module to be loaded as a shared object (.so) file:
JEVOIS_REGISTER_MODULE(DemoQRcode);
