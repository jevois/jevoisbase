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

#include <jevoisbase/Components/ArUco/ArUco.H>
#include <jevois/Image/RawImageOps.H>
#include <jevois/Core/Module.H>
#include <jevois/Core/Engine.H>
#include <opencv2/calib3d.hpp> // for projectPoints()
#include <opencv2/imgproc/imgproc.hpp>

#include <Eigen/Geometry> // for AngleAxis and Quaternion

// ##############################################################################################################
cv::aruco::Dictionary ArUco::getDictionary(aruco::Dict d)
{
  switch (d)
  {
  case aruco::Dict::Original:   return cv::aruco::getPredefinedDictionary(cv::aruco::DICT_ARUCO_ORIGINAL);
  case aruco::Dict::D4X4_50:    return cv::aruco::getPredefinedDictionary(cv::aruco::DICT_4X4_50);
  case aruco::Dict::D4X4_100:   return cv::aruco::getPredefinedDictionary(cv::aruco::DICT_4X4_100);
  case aruco::Dict::D4X4_250:   return cv::aruco::getPredefinedDictionary(cv::aruco::DICT_4X4_250);
  case aruco::Dict::D4X4_1000:  return cv::aruco::getPredefinedDictionary(cv::aruco::DICT_4X4_1000);
  case aruco::Dict::D5X5_50:    return cv::aruco::getPredefinedDictionary(cv::aruco::DICT_5X5_50);
  case aruco::Dict::D5X5_100:   return cv::aruco::getPredefinedDictionary(cv::aruco::DICT_5X5_100);
  case aruco::Dict::D5X5_250:   return cv::aruco::getPredefinedDictionary(cv::aruco::DICT_5X5_250);
  case aruco::Dict::D5X5_1000:  return cv::aruco::getPredefinedDictionary(cv::aruco::DICT_5X5_1000);
  case aruco::Dict::D6X6_50:    return cv::aruco::getPredefinedDictionary(cv::aruco::DICT_6X6_50);
  case aruco::Dict::D6X6_100:   return cv::aruco::getPredefinedDictionary(cv::aruco::DICT_6X6_100);
  case aruco::Dict::D6X6_250:   return cv::aruco::getPredefinedDictionary(cv::aruco::DICT_6X6_250);
  case aruco::Dict::D6X6_1000:  return cv::aruco::getPredefinedDictionary(cv::aruco::DICT_6X6_1000);
  case aruco::Dict::D7X7_50:    return cv::aruco::getPredefinedDictionary(cv::aruco::DICT_7X7_50);
  case aruco::Dict::D7X7_100:   return cv::aruco::getPredefinedDictionary(cv::aruco::DICT_7X7_100);
  case aruco::Dict::D7X7_250:   return cv::aruco::getPredefinedDictionary(cv::aruco::DICT_7X7_250);
  case aruco::Dict::D7X7_1000:  return cv::aruco::getPredefinedDictionary(cv::aruco::DICT_7X7_1000);
  case aruco::Dict::ATAG_16h5:  return cv::aruco::getPredefinedDictionary(cv::aruco::DICT_APRILTAG_16h5);
  case aruco::Dict::ATAG_25h9:  return cv::aruco::getPredefinedDictionary(cv::aruco::DICT_APRILTAG_25h9);
  case aruco::Dict::ATAG_36h10: return cv::aruco::getPredefinedDictionary(cv::aruco::DICT_APRILTAG_36h10);
  case aruco::Dict::ATAG_36h11: return cv::aruco::getPredefinedDictionary(cv::aruco::DICT_APRILTAG_36h11);
  }
}

// ##############################################################################################################
ArUco::~ArUco()
{ }

// ##############################################################################################################
void ArUco::postInit()
{
  //! Load camera calibration:
  itsCalib = engine()->loadCameraCalibration();

  // Init detector parameters:
  aruco::dictionary::freeze(true);
  aruco::detparams::freeze(true);
  cv::aruco::DetectorParameters dparams;
  
  switch (aruco::dictionary::get())
  {
  case aruco::Dict::ATAG_16h5:
  case aruco::Dict::ATAG_25h9:
  case aruco::Dict::ATAG_36h10:
  case aruco::Dict::ATAG_36h11:
    dparams.cornerRefinementMethod = cv::aruco::CORNER_REFINE_APRILTAG;
    break;
    
  default:
    dparams.cornerRefinementMethod = cv::aruco::CORNER_REFINE_SUBPIX;
  }
  
  // Read detector parameters if any:
  std::string const dpf = aruco::detparams::get();
  if (dpf.empty() == false)
  {
    cv::FileStorage fs(dpf, cv::FileStorage::READ);
    if (fs.isOpened())
    {
      if (dparams.readDetectorParameters(fs.root()) == false)
        LERROR("Error reading ArUco detector parameters from file [" << dpf <<"] -- IGNORED");
    }
    else LERROR("Failed to read ArUco detector parameters from file [" << dpf << "] -- IGNORED");
  }
  
  // Instantiate the dictionary:
  cv::aruco::Dictionary dico = getDictionary(aruco::dictionary::get());

  // Instantiate the detector (we use default refinement parameters):
  itsDetector = cv::Ptr<cv::aruco::ArucoDetector>(new cv::aruco::ArucoDetector(dico, dparams));
}

// ##############################################################################################################
void ArUco::postUninit()
{
  itsDetector.release();
  itsCalib = jevois::CameraCalibration();
  aruco::detparams::freeze(false);
  aruco::dictionary::freeze(false);
}

// ##############################################################################################################
void ArUco::detectMarkers(cv::InputArray image, cv::OutputArray ids, cv::OutputArrayOfArrays corners)
{
  itsDetector->detectMarkers(image, corners, ids);
}

// ##############################################################################################################
void ArUco::estimatePoseSingleMarkers(cv::InputArrayOfArrays corners, cv::OutputArray rvecs, cv::OutputArray tvecs)
{
  cv::aruco::estimatePoseSingleMarkers(corners, markerlen::get(),
                                       itsCalib.camMatrix, itsCalib.distCoeffs, rvecs, tvecs);
}

// ##############################################################################################################
void ArUco::sendSerial(jevois::StdModule * mod, std::vector<int> ids, std::vector<std::vector<cv::Point2f> > corners,
                       unsigned int w, unsigned int h, std::vector<cv::Vec3d> const & rvecs,
                       std::vector<cv::Vec3d> const & tvecs)
{
  if (rvecs.empty() == false)
  {
    float const siz = markerlen::get();
    
    // If we have rvecs and tvecs, we are doing 3D pose estimation, so send a 3D message:
    for (size_t i = 0; i < corners.size(); ++i)
    {
      cv::Vec3d const & rv = rvecs[i];
      cv::Vec3d const & tv = tvecs[i];
      
      // Compute quaternion:
      float theta = std::sqrt(rv[0] * rv[0] + rv[1] * rv[1] + rv[2] * rv[2]);
      Eigen::Vector3f axis(rv[0], rv[1], rv[2]);
      Eigen::Quaternion<float> q(Eigen::AngleAxis<float>(theta, axis));
      
      mod->sendSerialStd3D(tv[0], tv[1], tv[2],             // position
                           siz, siz, 1.0F,                  // size
                           q.w(), q.x(), q.y(), q.z(),      // pose
                           "U" + std::to_string(ids[i]));   // decoded ID with "U" prefix for ArUco
    }
  }
  else
  {
    // Send one 2D message per marker:
    for (size_t i = 0; i < corners.size(); ++i)
    {
      std::vector<cv::Point2f> const & currentMarker = corners[i];
      mod->sendSerialContour2D(w, h, currentMarker, "U" + std::to_string(ids[i]));
    }
  }
}

// ##############################################################################################################
void ArUco::drawDetections(jevois::RawImage & outimg, int txtx, int txty, std::vector<int> ids,
                           std::vector<std::vector<cv::Point2f> > corners, std::vector<cv::Vec3d> const & rvecs,
                           std::vector<cv::Vec3d> const & tvecs)
{
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
                                 int(p1.x + 0.5F), int(p1.y + 0.5F), 1, jevois::yuyv::LightGreen);
    }
        
    // draw first corner mark
    jevois::rawimage::drawDisk(outimg, int(currentMarker[0].x + 0.5F), int(currentMarker[0].y + 0.5F),
                               3, jevois::yuyv::LightGreen);

    // draw ID
    if (ids.empty() == false)
    {
      cv::Point2f cent(0.0F, 0.0F); for (int p = 0; p < 4; ++p) cent += currentMarker[p] * 0.25F;
      jevois::rawimage::writeText(outimg, std::string("id=") + std::to_string(ids[i]),
                                  int(cent.x + 0.5F), int(cent.y + 0.5F) - 5, jevois::yuyv::LightGreen);
    }
  }

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
      cv::projectPoints(axisPoints, rvecs[i], tvecs[i], itsCalib.camMatrix, itsCalib.distCoeffs, imagePoints);
      
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
        cv::projectPoints(cubePoints, rvecs[i], tvecs[i], itsCalib.camMatrix, itsCalib.distCoeffs, cuf);
        
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

  if (txtx >=0 && txty >= 0)
    jevois::rawimage::writeText(outimg, "Detected " + std::to_string(ids.size()) + " ArUco markers.",
                                txtx, txty, jevois::yuyv::White);
}

#ifdef JEVOIS_PRO
// ##############################################################################################################
void ArUco::drawDetections(jevois::GUIhelper & helper, std::vector<int> ids,
                           std::vector<std::vector<cv::Point2f> > corners, std::vector<cv::Vec3d> const & rvecs,
                           std::vector<cv::Vec3d> const & tvecs)
{
  ImU32 const col = ImColor(128, 255, 128, 255); // light green for lines

  // This code is like drawDetectedMarkers() in cv::aruco, but for ImGui:
  int nMarkers = int(corners.size());
  for (int i = 0; i < nMarkers; ++i)
  {
    std::vector<cv::Point2f> const & currentMarker = corners[i];
        
    // draw marker sides and prepare serial out string:
    helper.drawPoly(currentMarker, col, true);
        
    // draw first corner mark
    helper.drawCircle(currentMarker[0].x, currentMarker[0].y, 3.0F, col, true);

    // draw ID
    if (ids.empty() == false)
    {
      cv::Point2f cent(0.0F, 0.0F); for (int p = 0; p < 4; ++p) cent += currentMarker[p] * 0.25F;
      helper.drawText(cent.x, cent.y - 10, ("id=" + std::to_string(ids[i])).c_str(), col);
    }
  }

  // This code is like drawAxis() in cv::aruco, but for ImGui:
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
      cv::projectPoints(axisPoints, rvecs[i], tvecs[i], itsCalib.camMatrix, itsCalib.distCoeffs, imagePoints);
      
      // Draw axis lines
      helper.drawLine(imagePoints[0].x, imagePoints[0].y, imagePoints[1].x, imagePoints[1].y, 0xff0000ff);
      helper.drawLine(imagePoints[0].x, imagePoints[0].y, imagePoints[2].x, imagePoints[2].y, 0xff00ff00);
      helper.drawLine(imagePoints[0].x, imagePoints[0].y, imagePoints[3].x, imagePoints[3].y, 0xffff0000);
      
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
        cv::projectPoints(cubePoints, rvecs[i], tvecs[i], itsCalib.camMatrix, itsCalib.distCoeffs, cuf);

        auto drawface =
          [&](int a, int b, int c, int d)
          {
            std::vector<cv::Point2f> p { cuf[a], cuf[b], cuf[c], cuf[d] };
            helper.drawPoly(p, col, true);
          };
        
        // Draw cube lines and faces. For faces, vertices must be in clockwise order:
        drawface(0, 1, 2, 3);
        drawface(0, 1, 5, 4);
        drawface(1, 2, 6, 5);
        drawface(2, 3, 7, 6);
        drawface(3, 0, 4, 7);
        drawface(4, 5, 6, 7);
      }
    }
  }

  helper.itext("Detected " + std::to_string(ids.size()) + " ArUco markers.");
}

#endif
