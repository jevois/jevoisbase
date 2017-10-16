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
#include <jevois/Util/Utils.H>
#include <jevois/Image/RawImageOps.H>
#include <jevois/Debug/Timer.H>
#include <jevois/Util/Coordinates.H>

#include <linux/videodev2.h>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <string.h>

// This code was loosely inspired by:

// https://raw.githubusercontent.com/kylehounslow/opencv-tuts/master/object-tracking-tut/objectTrackingTut.cpp

// Written by Kyle Hounslow 2013

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
// documentation files (the "Software") , to deal in the Software without restriction, including without limitation the
// rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to
// permit persons to whom the Software is furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the
// Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
// OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

// icon by Roundicons in miscellaneous at flaticon

static jevois::ParameterCategory const ParamCateg("ObjectTracker Options");

//! Parameter \relates ObjectTracker
JEVOIS_DECLARE_PARAMETER(hrange, jevois::Range<unsigned char>, "Range of H values for HSV window",
                         jevois::Range<unsigned char>(10, 245), ParamCateg);

//! Parameter \relates ObjectTracker
JEVOIS_DECLARE_PARAMETER(srange, jevois::Range<unsigned char>, "Range of S values for HSV window",
                         jevois::Range<unsigned char>(10, 245), ParamCateg);

//! Parameter \relates ObjectTracker
JEVOIS_DECLARE_PARAMETER(vrange, jevois::Range<unsigned char>, "Range of V values for HSV window",
                         jevois::Range<unsigned char>(10, 245), ParamCateg);

//! Parameter \relates ObjectTracker
JEVOIS_DECLARE_PARAMETER(maxnumobj, size_t, "Max number of objects to declare a clean image",
                         10, ParamCateg);

//! Parameter \relates ObjectTracker
JEVOIS_DECLARE_PARAMETER(objectarea, jevois::Range<unsigned int>, "Range of object area (in pixels) to track",
                         jevois::Range<unsigned int>(20*20, 100*100), ParamCateg);

//! Parameter \relates ObjectTracker
JEVOIS_DECLARE_PARAMETER(erodesize, size_t, "Erosion structuring element size (pixels)",
                         3, ParamCateg);

//! Parameter \relates ObjectTracker
JEVOIS_DECLARE_PARAMETER(dilatesize, size_t, "Dilation structuring element size (pixels)",
                         8, ParamCateg);

//! Parameter \relates ObjectTracker
JEVOIS_DECLARE_PARAMETER(debug, bool, "Show contours of all object candidates if true",
                         true, ParamCateg);


//! Simple color-based object detection/tracking
/*! This modules isolates pixels within a given HSV range (hue, saturation, and value of color pixels), does some
    cleanups, and extracts object contours. It sends information about object centers over serial.

    This module usually works best with the camera sensor set to manual exposure, manual gain, manual color balance, etc
    so that HSV color values are reliable. See the file \b script.cfg file in this module's directory for an example of
    how to set the camera settings each time this module is loaded.

    This code was loosely inspired by:
    https://raw.githubusercontent.com/kylehounslow/opencv-tuts/master/object-tracking-tut/objectTrackingTut.cpp written
    by Kyle Hounslow, 2013. 

    Serial Messages
    ---------------

    This module can send standardized serial messages as described in \ref UserSerialStyle. One message is issued on
    every video frame for each detected and good object (good objects have a pixel area within the range specified by \p
    objectarea, and are only reported when the image is clean enough according to \p maxnumobj). The \p id field in the
    messages simply is \b blob for all messages.

    - Serial message type: \b 2D
    - `id`: always \b blob
    - `x`, `y`, or vertices: standardized 2D coordinates of blob center or of corners of bounding box
      (depending on \p serstyle)
    - `w`, `h`: standardized object size
    - `extra`: none (empty string)

    Trying it out
    -------------

    The default parameter settings (which are set in \b script.cfg explained below) attempt to detect light blue
    objects. Present a light blue object to the JeVois camera and see whether it is detected. When detected and good
    enough according to \p objectarea and \p maxnumobj, a green circle will be drawn at the center of each good object.

    For further use of this module, you may want to check out the following tutorials:

    - [Tuning the color-based object tracker using a python graphical
      interface](http://jevois.org/tutorials/UserColorTracking.html)
    - [Making a motorized pan-tilt head for JeVois and tracking
      objects](http://jevois.org/tutorials/UserPanTilt.html)
    - \ref ArduinoTutorial

    Tuning
    ------

    You should adjust parameters \p hrange, \p srange, and \p vrange to isolate the range of Hue, Saturation, and Value
    (respectively) that correspond to the objects you want to detect. Note that there is a \b script.cfg file in this
    module's directory that provides a range tuned to a lighgt blue object, as shown in the demo screenshot.

    Tuning the parameters is best done interactively by connecting to your JeVois camera while it is looking at some
    object of the desired color. Once you have achieved a tuning, you may want to set the hrange, srange, and vrange
    parameters in your \b script.cfg file for this module on the microSD card (see below).

    Typically, you would start by narrowing down on the hue, then the value, and finally the saturation. Make sure you
    also move your camera around and show it typical background clutter so check for false positives (detections of
    things which you are not interested, which can happen if your ranges are too wide).

    Config file
    -----------

    JeVois allows you to store parameter settings and commands in a file named \b script.cfg stored in the directory of
    a module. The file \b script.cfg may contain any sequence of commands as you would type them interactively in the
    JeVois command-line interface. For the \jvmod{ObjectTracker} module, a default script is provided that sets the
    camera to manual color, gain, and exposure mode (for more reliable color values), and to setup communication with a
    pan/tilt head as described in \ref ArduinoTutorial.

    The \b script.cfg file for ObjectTracker is stored on your microSD at
    <b>JEVOIS:/modules/JeVois/ObjectTracker/script.cfg</b> and is shown in \ref ArduinoTutorial as an example.


    @author Laurent Itti

    @videomapping YUYV 320 254 60.0 YUYV 320 240 60.0 JeVois ObjectTracker
    @videomapping NONE 0 0 0.0 YUYV 320 240 60.0 JeVois ObjectTracker
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
class ObjectTracker : public jevois::StdModule,
                      public jevois::Parameter<hrange, srange, vrange, maxnumobj, objectarea, erodesize,
                                               dilatesize, debug>
{
  public:
    //! Default base class constructor ok
    using jevois::StdModule::StdModule;

    //! Virtual destructor for safe inheritance
    virtual ~ObjectTracker() { }

    //! Processing function, no USB video output
    virtual void process(jevois::InputFrame && inframe) override
    {
      // Wait for next available camera image. Any resolution and format ok:
      jevois::RawImage inimg = inframe.get(); unsigned int const w = inimg.width, h = inimg.height;

      // Convert input image to BGR24, then to HSV:
      cv::Mat imgbgr = jevois::rawimage::convertToCvBGR(inimg);
      cv::Mat imghsv; cv::cvtColor(imgbgr, imghsv, cv::COLOR_BGR2HSV);

      // Let camera know we are done processing the input image:
      inframe.done();

      // Threshold the HSV image to only keep pixels within the desired HSV range:
      cv::Mat imgth;
      cv::inRange(imghsv, cv::Scalar(hrange::get().min(), srange::get().min(), vrange::get().min()),
                  cv::Scalar(hrange::get().max(), srange::get().max(), vrange::get().max()), imgth);

      // Apply morphological operations to cleanup the image noise:
      cv::Mat erodeElement = getStructuringElement(cv::MORPH_RECT, cv::Size(erodesize::get(), erodesize::get()));
      cv::erode(imgth, imgth, erodeElement);

      cv::Mat dilateElement = getStructuringElement(cv::MORPH_RECT, cv::Size(dilatesize::get(), dilatesize::get()));
      cv::dilate(imgth, imgth, dilateElement);

      // Detect objects by finding contours:
      std::vector<std::vector<cv::Point> > contours; std::vector<cv::Vec4i> hierarchy;
      cv::findContours(imgth, contours, hierarchy, CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE);

      // Identify the "good" objects:
      if (hierarchy.size() > 0 && hierarchy.size() <= maxnumobj::get())
      {
        double refArea = 0.0; int x = 0, y = 0; int refIdx = 0;

        for (int index = 0; index >= 0; index = hierarchy[index][0])
        {
          cv::Moments moment = cv::moments((cv::Mat)contours[index]);
          double area = moment.m00;
          if (objectarea::get().contains(int(area + 0.4999)) && area > refArea) { refArea = area; refIdx = index; }
        }
        
        // Send coords to serial port (for arduino, etc):
        if (refArea > 0.0) sendSerialContour2D(w, h, contours[refIdx], "blob");
      }
    }

    //! Processing function, with USB video output
    virtual void process(jevois::InputFrame && inframe, jevois::OutputFrame && outframe) override
    {
      static jevois::Timer timer("processing");

      // Wait for next available camera image. Any resolution ok, but require YUYV since we assume it for drawings:
      jevois::RawImage inimg = inframe.get(); unsigned int const w = inimg.width, h = inimg.height;
      inimg.require("input", w, h, V4L2_PIX_FMT_YUYV);

      timer.start();

      // While we process it, start a thread to wait for output frame and paste the input image into it:
      jevois::RawImage outimg; // main thread should not use outimg until paste thread is complete
      auto paste_fut = std::async(std::launch::async, [&]() {
          outimg = outframe.get();
          outimg.require("output", w, h + 14, inimg.fmt);
          jevois::rawimage::paste(inimg, outimg, 0, 0);
          jevois::rawimage::writeText(outimg, "JeVois Color Object Tracker", 3, 3, jevois::yuyv::White);
          jevois::rawimage::drawFilledRect(outimg, 0, h, w, outimg.height-h, 0x8000);
        });

      // Convert input image to BGR24, then to HSV:
      cv::Mat imgbgr = jevois::rawimage::convertToCvBGR(inimg);
      cv::Mat imghsv; cv::cvtColor(imgbgr, imghsv, cv::COLOR_BGR2HSV);

      // Threshold the HSV image to only keep pixels within the desired HSV range:
      cv::Mat imgth;
      cv::inRange(imghsv, cv::Scalar(hrange::get().min(), srange::get().min(), vrange::get().min()),
                  cv::Scalar(hrange::get().max(), srange::get().max(), vrange::get().max()), imgth);

      // Wait for paste to finish up:
      paste_fut.get();

      // Let camera know we are done processing the input image:
      inframe.done();
      
      // Apply morphological operations to cleanup the image noise:
      cv::Mat erodeElement = getStructuringElement(cv::MORPH_RECT, cv::Size(erodesize::get(), erodesize::get()));
      cv::erode(imgth, imgth, erodeElement);

      cv::Mat dilateElement = getStructuringElement(cv::MORPH_RECT, cv::Size(dilatesize::get(), dilatesize::get()));
      cv::dilate(imgth, imgth, dilateElement);

      // Detect objects by finding contours:
      std::vector<std::vector<cv::Point> > contours; std::vector<cv::Vec4i> hierarchy;
      cv::findContours(imgth, contours, hierarchy, CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE);

      // If desired, draw all contours in a thread:
      std::future<void> draw_fut;
      if (debug::get())
        draw_fut = std::async(std::launch::async, [&]() {
            // We reinterpret the top portion of our YUYV output image as an opencv 8UC2 image:
            cv::Mat outuc2(imgth.rows, imgth.cols, CV_8UC2, outimg.pixelsw<unsigned char>()); // pixel data shared
            for (size_t i = 0; i < contours.size(); ++i)
              cv::drawContours(outuc2, contours, i, jevois::yuyv::LightPink, 2, 8, hierarchy);
          });
      
      // Identify the "good" objects:
      int numobj = 0;
      if (hierarchy.size() > 0 && hierarchy.size() <= maxnumobj::get())
      {
        double refArea = 0.0; int x = 0, y = 0; int refIdx = 0;

        for (int index = 0; index >= 0; index = hierarchy[index][0])
        {
          cv::Moments moment = cv::moments((cv::Mat)contours[index]);
          double area = moment.m00;
          if (objectarea::get().contains(int(area + 0.4999)) && area > refArea)
          { x = moment.m10 / area + 0.4999; y = moment.m01 / area + 0.4999; refArea = area; refIdx = index; }
        }
        
        if (refArea > 0.0)
        {
          ++numobj;
          jevois::rawimage::drawCircle(outimg, x, y, 20, 1, jevois::yuyv::LightGreen);

          // Send coords to serial port (for arduino, etc):
          sendSerialContour2D(w, h, contours[refIdx], "blob");
        }
      }

      // Show number of detected objects:
      jevois::rawimage::writeText(outimg, "Detected " + std::to_string(numobj) + " objects.",
                                  3, h + 2, jevois::yuyv::White);
      
      // Show processing fps:
      std::string const & fpscpu = timer.stop();
      jevois::rawimage::writeText(outimg, fpscpu, 3, h - 13, jevois::yuyv::White);

      // Possibly wait until all contours are drawn, if they had been requested:
      if (draw_fut.valid()) draw_fut.get();
      
      // Send the output image with our processing results to the host over USB:
      outframe.send();
    }
};

// Allow the module to be loaded as a shared object (.so) file:
JEVOIS_REGISTER_MODULE(ObjectTracker);
