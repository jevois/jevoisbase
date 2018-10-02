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
#include <jevois/Debug/Timer.H>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d.hpp>

// icon by Madebyoliver in Game Collection at flaticon

//! Counting dice pips
/*! This module can help you automate counting your dice values, for example when playing games that involve throwing
    multiple dice.

    This application scenario was suggested by JeVois user mapembert at the [JeVois Tech Zone](http://jevois.org/qa)
    in this post:

    http://jevois.org/qa/index.php?qa=328

    The code implemented by this module is a modified version of original code (mentioned in the above post) contributed
    by Yohann Payet.

    This module is the result of JeVois tutorial [A JeVois dice counting module in
    C++](http://jevois.org/tutorials/ProgrammerDice.html). Also see [JeVois python tutorial: A dice counting
    module](http://jevois.org/tutorials/ProgrammerPythonDice.html) for a Python version.

    Serial messages
    ---------------

    This module sends the following serial message (remember to turn serial outputs on, using `setpar serout Hard` or
    similar; see \ref UserCli):
    \verbatim
    PIPS n
    \endverbatim
    where \a n is the total number of pips detected. No message is sent if \a n is zero.

    @author Laurent Itti

    @videomapping YUYV 640 480 7.5 YUYV 640 480 7.5 SampleVendor DiceCounter
    @email itti\@usc.edu
    @address University of Southern California, HNB-07A, 3641 Watt Way, Los Angeles, CA 90089-2520, USA
    @copyright Copyright (C) 2017 by Laurent Itti, iLab and the University of Southern California
    @mainurl http://jevois.org
    @supporturl http://jevois.org/doc
    @otherurl http://iLab.usc.edu
    @license GPL v3
    @distribution Unrestricted
    @restrictions None
    \ingroup modules */
class DiceCounter : public jevois::Module
{
  public:
    //! Constructor
    DiceCounter(std::string const & instance) : jevois::Module(instance)
    {
      // Setting detector parameters
      cv::SimpleBlobDetector::Params params;
      params.filterByCircularity = true;
      params.filterByArea = true;
      params.minArea = 200.0f;

      // Creating a detector object
      itsDetector = cv::SimpleBlobDetector::create(params);
    }

    //! Virtual destructor for safe inheritance
    virtual ~DiceCounter() { }

    //! Processing function
    virtual void process(jevois::InputFrame && inframe, jevois::OutputFrame && outframe) override
    {
      static jevois::Timer timer("processing");

      // Wait for next available camera image:
      jevois::RawImage const inimg = inframe.get(); unsigned int const w = inimg.width, h = inimg.height;

      timer.start();

      // We only support YUYV pixels in this example, any resolution:
      inimg.require("input", inimg.width, inimg.height, V4L2_PIX_FMT_YUYV);

      // Start a thread to wait for output image anc opy input into output:
      jevois::RawImage outimg;
      std::future<void> fut = std::async(std::launch::async, [&]() {
          // Wait for an image from our gadget driver into which we will put our results:
          outimg = outframe.get();

          // Enforce that the input and output formats and image sizes match:
          outimg.require("output", w, h, inimg.fmt);
      
          // Just copy the pixel data over:
          jevois::rawimage::paste(inimg, outimg, 0, 0);
        });

      // Detect dice pips: First convert input to grayscale:
      cv::Mat grayImage = jevois::rawimage::convertToCvGray(inimg);

      // filter noise
      cv::GaussianBlur(grayImage, grayImage, cv::Size(5, 5), 0, 0);

      // apply automatic threshold
      cv::threshold(grayImage, grayImage, 0.0, 255, cv::THRESH_BINARY_INV | cv::THRESH_OTSU);

      // background area
      cv::Mat kernel; // not initialized??
      int const morphBNo2 = 2;
      cv::dilate(grayImage, grayImage, kernel, cv::Point(-1, -1), morphBNo2);
      cv::Mat image(grayImage.rows, grayImage.cols, CV_8U, cv::Scalar(255, 255, 255));
      cv::Mat invBack2 = image - grayImage;

      // blob detection
      std::vector<cv::KeyPoint> keypoints;
      itsDetector->detect(invBack2, keypoints);
      int nrOfBlobs = keypoints.size();

      // Wait until our other thread is done:
      fut.get();

      // Let camera know we are done processing the input image:
      inframe.done();
      
      // draw keypoints
      for (cv::KeyPoint const & kp : keypoints)
        jevois::rawimage::drawCircle(outimg, int(kp.pt.x + 0.5F), int(kp.pt.y + 0.5F), int(kp.size * 0.5F),
                                     2, jevois::yuyv::LightGreen);

      // Show number of detected pips:
      jevois::rawimage::writeText(outimg, "JeVois dice counter: " + std::to_string(nrOfBlobs) + " pips",
                                  3, 3, jevois::yuyv::White);
      
      // Show processing fps:
      std::string const & fpscpu = timer.stop();
      jevois::rawimage::writeText(outimg, fpscpu, 3, h - 13, jevois::yuyv::White);
      
      // Send the output image with our processing results to the host over USB:
      outframe.send();

      // Send serial message:
      if (nrOfBlobs) sendSerial("PIPS " + std::to_string(nrOfBlobs));
    }

  private:
    cv::Ptr<cv::SimpleBlobDetector> itsDetector;
};

// Allow the module to be loaded as a shared object (.so) file:
JEVOIS_REGISTER_MODULE(DiceCounter);
