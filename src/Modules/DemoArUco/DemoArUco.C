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

//! Parameter \relates DemoArUco
JEVOIS_DECLARE_PARAMETER(showpose, bool, "Show pose vectors, requires a valid camera calibration matrix",
                         false, aruco::ParamCateg);

//! Parameter \relates DemoArUco
JEVOIS_DECLARE_PARAMETER(markerlen, float, "Marker side length (meters), used only for pose estimation",
                         0.1F, aruco::ParamCateg);

//! Parameter \relates DemoArUco
JEVOIS_DECLARE_PARAMETER(serstyle, int, "Serial output style for strings issued for each detected ArUco marker: "
                         "0: [ArUco ID] where ID is the decoded ArUco ID; "
                         "1: [ArUco ID X,Y] sends ID and standardized coordinates of the marker center;"
                         "2: [ArUco ID X1,Y1 X2,Y2 X3,Y3 X4,Y4] sends ID and standardized coords of the 4 corners.",
                         0, jevois::Range<int>(0, 2), aruco::ParamCateg);

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

    ArUco markers can be created with several standard dictionaries. Different disctionaries give rise to different
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

    This module can send the following messages over serial port (make sure you set the \p serout parameter to
    designate the serial port to which you want to send these messages, see the documentation for \p serout under
    the \ref UserCli).
    
    - When serstyle is 0:
      \verbatim
      ArUco ID
      \endverbatim
      where
      + ID is the decoded ID of the ArUco that was detected

    - When serstyle is 1:
      \verbatim
      ArUco ID X,Y
      \endverbatim
      where
      + ID is the decoded ID of the ArUco that was detected
      + X,Y are the coordinates of the marker's center

    - When serstyle is 2:
      \verbatim
      ArUco ID X1,Y1 X2,Y2 X3,Y3 X4,Y4
      \endverbatim
      where
      + ID is the decoded ID of the ArUco that was detected
      + X1,Y1 are the coordinates of the first corner
      + X2,Y2 are the coordinates of the second corner
      + X3,Y3 are the coordinates of the third corner
      + X4,Y4 are the coordinates of the fourth corner

    The coordinates are normalized so that the serial outputs are independent of camera resolution. x=0, y=0 is the
    center of the field of view, the left edge of the camera image is at x=-1000, the right edge at x=1000, the top edge
    at y=-750, and the bottom edge at y=750 (since the camera has a 4:3 aspect ratio). See \ref coordhelpers for more
    information on standardized coordinates in JeVois.

    One message is issued for every detected ArUco, on every video frame.

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

    Showing the pose vectors
    ------------------------

    The OpenCV ArUco module can also compute normals to each marker, which allows you to recover the marker's
    orientation and distance. The requires that the camera be calibrated, see the documentation of the ArUco component
    in JeVois. A generic calibration that is for a JeVois camera with standard lens is included in file \b
    calibration.yaml in the module's directory (on the MicroSD, this is
    <b>JEVOIS:/modules/JeVois/DemoArUco/calibration.yaml</b>).

    FIXME this crashes since the update to opencv 3.2, need to figure out what changed since 3.1.


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
class DemoArUco : public jevois::Module,
                  public jevois::Parameter<showpose, markerlen, serstyle>
{
  public: 
    // ####################################################################################################
    //! Constructor
    // ####################################################################################################
    DemoArUco(std::string const & instance) : jevois::Module(instance)
    {
      itsArUco = addSubComponent<ArUco>("aruco");
      itsArUco->camparams::set("calibration.yaml"); // use camera calibration parameters in module path
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

      // Let camera know we are done processing the input image:
      inframe.done();

      // Send serial output:
      sendAllSerial(ids, corners, w, h);
    }

    // ####################################################################################################
    //! Send the serial messages
    // ####################################################################################################
    void sendAllSerial(std::vector<int> ids, std::vector<std::vector<cv::Point2f> > corners,
                       unsigned int w, unsigned int h)
    {
      int const ss = serstyle::get(); int const nMarkers = int(corners.size());
      for (int i = 0; i < nMarkers; ++i)
      {
        std::vector<cv::Point2f> const & currentMarker = corners[i];
        std::ostringstream serout; serout << "ArUco " << ids[i] << std::fixed << std::setprecision(1);
        switch (ss)
        {
        case 0: // Send only the ID
          break;
          
        case 1: // Send the coordinates of the center
        {
          float cx = 0.0F, cy = 0.0F;
          for (cv::Point2f cm : currentMarker)
          {
            // Normalize the coordinates:
            float x = cm.x, y = cm.y; jevois::coords::imgToStd(x, y, w, h);
            cx += x; cy += y;
          }
          serout << ' ' << int(0.25F * cx + 0.5F) << ',' << int(0.25F * cy + 0.5F);
        }
        break;
          
        default: // Send the 4 corner coordinates:
          for (cv::Point2f cm : currentMarker)
          {
            // Normalize the coordinates:
            float x = cm.x, y = cm.y; jevois::coords::imgToStd(x, y, w, h, 1.0F);
            serout << ' ' << int(x) << ',' << int(y);
          }
        }
        
        // Send to serial:
        sendSerial(serout.str());
      }
    }

    // ####################################################################################################
    //! Processing function with video output to USB
    // ####################################################################################################
    virtual void process(jevois::InputFrame && inframe, jevois::OutputFrame && outframe) override
    {
      static jevois::Timer timer("processing");
      
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

      if (showpose::get() && ids.empty() == false)
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
          jevois::rawimage::drawLine(outimg, int(p0.x + 0.4999F), int(p0.y + 0.4999F),
                                     int(p1.x + 0.4999F), int(p1.y + 0.3999F), 1, jevois::yuyv::LightGreen);
        }
        
        // draw first corner mark
        jevois::rawimage::drawDisk(outimg, int(currentMarker[0].x + 0.4999F), int(currentMarker[0].y + 0.4999F),
                                   3, jevois::yuyv::LightGreen);

        // draw ID
        if (ids.empty() == false)
        {
          cv::Point2f cent(0.0F, 0.0F); for (int p = 0; p < 4; ++p) cent += currentMarker[p] * 0.25F;
          jevois::rawimage::writeText(outimg, std::string("id=") + std::to_string(ids[i]),
                                      int(cent.x + 0.4999F), int(cent.y + 0.4999F) - 5, jevois::yuyv::LightPink);
        }
      }

      // Send serial output:
      sendAllSerial(ids, corners, w, h);

      // This code is like drawAxis() in cv::aruco, but for YUYV output image:
      if (showpose::get() && ids.empty() == false)
      {
        float const length = markerlen::get() * 0.5F;
        
        for (size_t i = 0; i < ids.size(); ++i)
        {
          // project axis points
          std::vector<cv::Point3f> axisPoints;
          axisPoints.push_back(cv::Point3f(0.0F, 0.0F, 0.0F));
          axisPoints.push_back(cv::Point3f(length, 0.0F, 0.0F));
          axisPoints.push_back(cv::Point3f(0.0F, length, 0.0F));
          axisPoints.push_back(cv::Point3f(0.0F, 0.0F, length));

          std::vector<cv::Point2f> imagePoints;
          cv::projectPoints(axisPoints, rvecs[i], tvecs[i], itsArUco->itsCamMatrix,
                            itsArUco->itsDistCoeffs, imagePoints);

          // draw axis lines
          jevois::rawimage::drawLine(outimg, int(imagePoints[0].x + 0.4999F), int(imagePoints[0].y + 0.4999F),
                                     int(imagePoints[1].x + 0.4999F), int(imagePoints[1].y + 0.4999F),
                                     2, jevois::yuyv::MedPurple);
          jevois::rawimage::drawLine(outimg, int(imagePoints[0].x + 0.4999F), int(imagePoints[0].y + 0.4999F),
                                     int(imagePoints[2].x + 0.4999F), int(imagePoints[2].y + 0.4999F),
                                     2, jevois::yuyv::MedGreen);
          jevois::rawimage::drawLine(outimg, int(imagePoints[0].x + 0.4999F), int(imagePoints[0].y + 0.4999F),
                                     int(imagePoints[3].x + 0.4999F), int(imagePoints[3].y + 0.4999F),
                                     2, jevois::yuyv::MedGrey);
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
