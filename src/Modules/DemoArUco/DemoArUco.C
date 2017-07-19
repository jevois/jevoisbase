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
#include <jevois/Util/Coordinates.H>
#include <jevoisbase/Components/ArUco/ArUco.H>
#include <linux/videodev2.h> // for v4l2 pixel types
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/calib3d.hpp> // for projectPoints()
#include <sstream>

#include <Eigen/Geometry> // for AngleAxis and Quaternion

//! Parameter \relates DemoArUco
JEVOIS_DECLARE_PARAMETER(dopose, bool, "Compute and show pose vectors, requires a valid camera calibration",
                         false, aruco::ParamCateg);

//! Parameter \relates DemoArUco
JEVOIS_DECLARE_PARAMETER(markerlen, float, "Marker side length (millimeters), used only for pose estimation",
                         100.0F, aruco::ParamCateg);

//! Parameter \relates DemoArUco
JEVOIS_DECLARE_PARAMETER(showcube, bool, "Show a 3D cube on top of each detected marker, when dopose is also true",
                         true, aruco::ParamCateg);

//! Simple demo of ArUco augmented reality markers detection and decoding
/*! Detect and decode patterns known as ArUco markers, which are small 2D barcodes often used in augmented
    reality and robotics.

    ArUco markers are small 2D barcodes. Each ArUco marker corresponds to a number, encoded into a small grid of black
    and white pixels. The ArUco decoding algorithm is capable of locating, decoding, and of estimating the pose
    (location and orientation in space) of any ArUco markers in the camera's field of view.

    ArUcos are very useful as tags for many robotics and augmented reality applications. For example, one may place an
    ArUco next to a robot's charging station, an elevator button, or an object that a robot should manipulate.

    For more information about ArUco, see https://www.uco.es/investiga/grupos/ava/node/26

    The implementation of ArUco used by JeVois is the one of OpenCV-Contrib, documented here:
    http://docs.opencv.org/3.2.0/d5/dae/tutorial_aruco_detection.html

    ArUco markers can be created with several standard dictionaries. Different dictionaries give rise to different
    numbers of pixels in the markers, and to different numbers of possible symbols that can be created using the
    dictionary. The default dictionary used by JeVois is 4x4 with 50 symbols. Other dictionaries are also supported by
    setting the appropriate parameter over serial port or in a config file, up to 7x7 with 1000 symbols.

    Creating and printing markers
    -----------------------------

    We have created the 50 markers available in the default dictionary (4x4_50) as PNG images that you can download and
    print, at http://jevois.org/data/ArUco.zip

    To make your own, for example, using another dictionary, see the documentation of the ArUco component of
    JeVoisBase. Some utilities are provided with the component.

    Serial Messages
    ---------------

    This module can send standardized serial messages as described in \ref UserSerialStyle.

    When \p dopose is turned on, 3D messages will be sent, otherwise 2D messages.

    One message is issued for every detected ArUco, on every video frame.

    2D messages when \p dopose is off:

    - Serial message type: \b 2D
    - `id`: decoded ArUco marker ID
    - `x`, `y`, or vertices: standardized 2D coordinates of marker center or corners
    - `w`, `h`: standardized marker size
    - `extra`: none (empty string)

    3D messages when \p dopose is on:

    - Serial message type: \b 3D
    - `id`: decoded ArUco marker ID
    - `x`, `y`, `z`, or vertices: 3D coordinates in millimeters of marker center or corners
    - `w`, `h`, `d`: marker size in millimeters, a depth of 1mm is always used
    - `extra`: none (empty string)

    If you will use the quaternion data (Detail message style; see \ref UserSerialStyle), you should probably set the \p
    serprec parameter to something non-zero to get enough accuracy in the quaternion values.

    Things to try
    -------------

    - First, use a video viewer software on a host computer and select one of the video modes with video output over
      USB. Point your JeVois camera towards one of the screenshots provided with this module, or towards some ArUco
      markers that you find on the web or that you have printed from the collection above (note: the default dictionary
      is 4x4_50, see parameter \p dictionary).

    - Then try it with no video output, as it would be used by a robot. Connect to the command-line interface of your
      JeVois camera through the serial-over-USB connection (see \ref UserCli; on Linux, you would use <b>sudo screen
      /dev/ttyACM0 115200</b>) and try:
      \verbatim
      setpar serout USB
      setmapping2 YUYV 320 240 30.0 JeVois DemoArUco
      streamon
      \endverbatim
      and point the camera to some markers; the camera should issue messages about all the markers it identifies.

    Computing and showing 3D pose
    -----------------------------

    The OpenCV ArUco module can also compute the 3D location and orientation of each marker in the world when \p dopose
    is true. The requires that the camera be calibrated, see the documentation of the \ref ArUco component in
    JeVoisBase. A generic calibration that is for a JeVois camera with standard lens is included in files \b
    calibration640x480.yaml, \b calibration352x288.yaml, etc in the module's directory (on the MicroSD, this is in
    <b>JEVOIS:/modules/JeVois/DemoArUco/</b>).

    When doing pose estimation, you should set the \p markerlen parameter to the size (width) in millimeters of your
    actual physical markers. Knowing that size will allow the pose estimation algorithm to know where in the world your
    detected markers are.


    @author Laurent Itti

    @displayname Demo ArUco
    @videomapping NONE 0 0 0 YUYV 320 240 30.0 JeVois DemoArUco
    @videomapping YUYV 320 260 30.0 YUYV 320 240 30.0 JeVois DemoArUco
    @videomapping YUYV 640 500 20.0 YUYV 640 480 20.0 JeVois DemoArUco
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
class DemoArUco : public jevois::StdModule,
                  public jevois::Parameter<dopose, markerlen, showcube>
{
  public: 
    // ####################################################################################################
    //! Constructor
    // ####################################################################################################
    DemoArUco(std::string const & instance) : jevois::StdModule(instance)
    {
      itsArUco = addSubComponent<ArUco>("aruco");
    }

    // ####################################################################################################
    //! Virtual destructor for safe inheritance
    // ####################################################################################################
    virtual ~DemoArUco()
    { }

    // ####################################################################################################
    //! Processing function, no video output
    // ####################################################################################################
    virtual void process(jevois::InputFrame && inframe) override
    {
      // Wait for next available camera image, any format and resolution ok here:
      jevois::RawImage const inimg = inframe.get(); unsigned int const w = inimg.width, h = inimg.height;

      // Convert the image to grayscale and process:
      cv::Mat cvimg = jevois::rawimage::convertToCvGray(inimg);
      std::vector<int> ids;
      std::vector<std::vector<cv::Point2f> > corners;
      itsArUco->detectMarkers(cvimg, ids, corners);

      // Do pose computation if desired:
      std::vector<cv::Vec3d> rvecs, tvecs;
      if (dopose::get() && ids.empty() == false)
        itsArUco->estimatePoseSingleMarkers(corners, markerlen::get(), rvecs, tvecs);
      
      // Let camera know we are done processing the input image:
      inframe.done();

      // Send serial output:
      sendAllSerial(ids, corners, w, h, rvecs, tvecs);
    }

    // ####################################################################################################
    //! Send the serial messages
    // ####################################################################################################
    void sendAllSerial(std::vector<int> ids, std::vector<std::vector<cv::Point2f> > corners,
                       unsigned int w, unsigned int h, std::vector<cv::Vec3d> const & rvecs,
                       std::vector<cv::Vec3d> const & tvecs)
    {
      if (rvecs.empty() == false)
      {
        float const siz = markerlen::get();
        
        // If we have rvecs and tvecs, we are doing 3D pose estimation, so send a 3D message:
        for (size_t i = 0; i < corners.size(); ++i)
        {
          std::vector<cv::Point2f> const & currentMarker = corners[i];
          cv::Vec3d const & rv = rvecs[i];
          cv::Vec3d const & tv = tvecs[i];
          
          // Compute quaternion:
          float theta = std::sqrt(rv[0] * rv[0] + rv[1] * rv[1] + rv[2] * rv[2]);
          Eigen::Vector3f axis(rv[0], rv[1], rv[2]);
          Eigen::Quaternion<float> q(Eigen::AngleAxis<float>(theta, axis));
          
          sendSerialStd3D(tv[0], tv[1], tv[2],        // position
                          siz, siz, 1.0F,             // size
                          q.w(), q.x(), q.y(), q.z(), // pose
                          std::to_string(ids[i]));    // decoded ID
        }
      }
      else
      {
        // Send one 2D message per parker:
        for (size_t i = 0; i < corners.size(); ++i)
        {
          std::vector<cv::Point2f> const & currentMarker = corners[i];
          sendSerialContour2D(w, h, currentMarker, std::to_string(ids[i]));
        }
      }
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

      // While we process it, start a thread to wait for out frame and paste the input into it:
      jevois::RawImage outimg;
      auto paste_fut = std::async(std::launch::async, [&]() {
          outimg = outframe.get();
          outimg.require("output", w, h + 20, inimg.fmt);
          jevois::rawimage::paste(inimg, outimg, 0, 0);
          jevois::rawimage::writeText(outimg, "JeVois ArUco Demo", 3, 3, jevois::yuyv::White);
          jevois::rawimage::drawFilledRect(outimg, 0, h, w, outimg.height-h, jevois::yuyv::Black);
        });

      // Convert the image to grayscale and process:
      cv::Mat cvimg = jevois::rawimage::convertToCvGray(inimg);
      std::vector<int> ids;
      std::vector<std::vector<cv::Point2f> > corners;
      std::vector<cv::Vec3d> rvecs, tvecs;
      itsArUco->detectMarkers(cvimg, ids, corners);

      if (dopose::get() && ids.empty() == false)
        itsArUco->estimatePoseSingleMarkers(corners, markerlen::get(), rvecs, tvecs);
      
      // Wait for paste to finish up:
      paste_fut.get();

      // Let camera know we are done processing the input image:
      inframe.done();

      // Show all the results:

      // This code is like drawDetectedMarkers() in cv::aruco, but for YUYV output image:
      int nMarkers = int(corners.size());
      for (int i = 0; i < nMarkers; ++i)
      {
        std::vector<cv::Point2f> const & currentMarker = corners[i];
        
        // draw marker sides and prepare serial out string:
        for (int j = 0; j < 4; ++j)
        {
          cv::Point2f const & p0 = currentMarker[j];
          cv::Point2f const & p1 = currentMarker[ (j+1) % 4 ];
          jevois::rawimage::drawLine(outimg, int(p0.x + 0.5F), int(p0.y + 0.5F),
                                     int(p1.x + 0.5F), int(p1.y + 0.3999F), 1, jevois::yuyv::LightGreen);
        }
        
        // draw first corner mark
        jevois::rawimage::drawDisk(outimg, int(currentMarker[0].x + 0.5F), int(currentMarker[0].y + 0.5F),
                                   3, jevois::yuyv::LightGreen);

        // draw ID
        if (ids.empty() == false)
        {
          cv::Point2f cent(0.0F, 0.0F); for (int p = 0; p < 4; ++p) cent += currentMarker[p] * 0.25F;
          jevois::rawimage::writeText(outimg, std::string("id=") + std::to_string(ids[i]),
                                      int(cent.x + 0.5F), int(cent.y + 0.5F) - 5, jevois::yuyv::LightPink);
        }
      }

      // Send serial output:
      sendAllSerial(ids, corners, w, h, rvecs, tvecs);

      // This code is like drawAxis() in cv::aruco, but for YUYV output image:
      if (dopose::get() && ids.empty() == false)
      {
        float const length = markerlen::get() * 0.4F;
        
        for (size_t i = 0; i < ids.size(); ++i)
        {
          // Project axis points:
          std::vector<cv::Point3f> axisPoints;
          axisPoints.push_back(cv::Point3f(0.0F, 0.0F, 0.0F));
          axisPoints.push_back(cv::Point3f(length, 0.0F, 0.0F));
          axisPoints.push_back(cv::Point3f(0.0F, length, 0.0F));
          axisPoints.push_back(cv::Point3f(0.0F, 0.0F, length));

          std::vector<cv::Point2f> imagePoints;
          cv::projectPoints(axisPoints, rvecs[i], tvecs[i], itsArUco->itsCamMatrix,
                            itsArUco->itsDistCoeffs, imagePoints);

          // Draw axis lines
          jevois::rawimage::drawLine(outimg, int(imagePoints[0].x + 0.5F), int(imagePoints[0].y + 0.5F),
                                     int(imagePoints[1].x + 0.5F), int(imagePoints[1].y + 0.5F),
                                     2, jevois::yuyv::MedPurple);
          jevois::rawimage::drawLine(outimg, int(imagePoints[0].x + 0.5F), int(imagePoints[0].y + 0.5F),
                                     int(imagePoints[2].x + 0.5F), int(imagePoints[2].y + 0.5F),
                                     2, jevois::yuyv::MedGreen);
          jevois::rawimage::drawLine(outimg, int(imagePoints[0].x + 0.5F), int(imagePoints[0].y + 0.5F),
                                     int(imagePoints[3].x + 0.5F), int(imagePoints[3].y + 0.5F),
                                     2, jevois::yuyv::MedGrey);

          // Also draw a cube if requested:
          if (showcube::get())
          {
            float const len = markerlen::get() * 0.5F;

            std::vector<cv::Point3f> cubePoints;
            cubePoints.push_back(cv::Point3f(-len, -len, 0.0F));
            cubePoints.push_back(cv::Point3f(len, -len, 0.0F));
            cubePoints.push_back(cv::Point3f(len, len, 0.0F));
            cubePoints.push_back(cv::Point3f(-len, len, 0.0F));
            cubePoints.push_back(cv::Point3f(-len, -len, len * 2.0F));
            cubePoints.push_back(cv::Point3f(len, -len, len * 2.0F));
            cubePoints.push_back(cv::Point3f(len, len, len * 2.0F));
            cubePoints.push_back(cv::Point3f(-len, len, len * 2.0F));

            std::vector<cv::Point2f> cuf;
            cv::projectPoints(cubePoints, rvecs[i], tvecs[i], itsArUco->itsCamMatrix,
                              itsArUco->itsDistCoeffs, cuf);

            // Round all the coordinates:
            std::vector<cv::Point> cu;
            for (auto const & p : cuf) cu.push_back(cv::Point(int(p.x + 0.5F), int(p.y + 0.5F)));
            
            // Draw cube lines:
            jevois::rawimage::drawLine(outimg, cu[0].x, cu[0].y, cu[1].x, cu[1].y, 2, jevois::yuyv::LightGreen);
            jevois::rawimage::drawLine(outimg, cu[1].x, cu[1].y, cu[2].x, cu[2].y, 2, jevois::yuyv::LightGreen);
            jevois::rawimage::drawLine(outimg, cu[2].x, cu[2].y, cu[3].x, cu[3].y, 2, jevois::yuyv::LightGreen);
            jevois::rawimage::drawLine(outimg, cu[3].x, cu[3].y, cu[0].x, cu[0].y, 2, jevois::yuyv::LightGreen);
            jevois::rawimage::drawLine(outimg, cu[4].x, cu[4].y, cu[5].x, cu[5].y, 2, jevois::yuyv::LightGreen);
            jevois::rawimage::drawLine(outimg, cu[5].x, cu[5].y, cu[6].x, cu[6].y, 2, jevois::yuyv::LightGreen);
            jevois::rawimage::drawLine(outimg, cu[6].x, cu[6].y, cu[7].x, cu[7].y, 2, jevois::yuyv::LightGreen);
            jevois::rawimage::drawLine(outimg, cu[7].x, cu[7].y, cu[4].x, cu[4].y, 2, jevois::yuyv::LightGreen);
            jevois::rawimage::drawLine(outimg, cu[0].x, cu[0].y, cu[4].x, cu[4].y, 2, jevois::yuyv::LightGreen);
            jevois::rawimage::drawLine(outimg, cu[1].x, cu[1].y, cu[5].x, cu[5].y, 2, jevois::yuyv::LightGreen);
            jevois::rawimage::drawLine(outimg, cu[2].x, cu[2].y, cu[6].x, cu[6].y, 2, jevois::yuyv::LightGreen);
            jevois::rawimage::drawLine(outimg, cu[3].x, cu[3].y, cu[7].x, cu[7].y, 2, jevois::yuyv::LightGreen);
          }
          
        }
      }

      jevois::rawimage::writeText(outimg, "Detected " + std::to_string(ids.size()) + " ArUco markers.",
                                  3, h + 5, jevois::yuyv::White);

      // Show processing fps:
      std::string const & fpscpu = timer.stop();
      jevois::rawimage::writeText(outimg, fpscpu, 3, h - 13, jevois::yuyv::White);
    
      // Send the output image with our processing results to the host over USB:
      outframe.send();
    }

    // ####################################################################################################
  protected:
    std::shared_ptr<ArUco> itsArUco;
};

// Allow the module to be loaded as a shared object (.so) file:
JEVOIS_REGISTER_MODULE(DemoArUco);
