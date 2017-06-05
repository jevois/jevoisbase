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

// ##############################################################################################################
ArUco::~ArUco()
{ }

// ##############################################################################################################
void ArUco::postInit()
{
  // Read camera parameters if any:
  aruco::camparams::freeze();
  std::string const cpf = aruco::camparams::get();
  if (cpf.empty() == false)
  {
    cv::FileStorage fs(cpf, cv::FileStorage::READ);
    if (fs.isOpened())
    {
      fs["camera_matrix"] >> itsCamMatrix;
      fs["distortion_coefficients"] >> itsDistCoeffs;
    }
    else LERROR("Failed to read camera parameters from file [" << cpf << "] -- IGNORED");
  }
  
  // Read detector parameters if any:
  aruco::detparams::freeze();
  std::string const dpf = aruco::detparams::get();
  itsDetectorParams.reset(new cv::aruco::DetectorParameters());
  itsDetectorParams->doCornerRefinement = true; // do corner refinement in markers by default
  if (dpf.empty() == false)
  {
    cv::FileStorage fs(dpf, cv::FileStorage::READ);
    if (fs.isOpened())
    {
      fs["adaptiveThreshWinSizeMin"] >> itsDetectorParams->adaptiveThreshWinSizeMin;
      fs["adaptiveThreshWinSizeMax"] >> itsDetectorParams->adaptiveThreshWinSizeMax;
      fs["adaptiveThreshWinSizeStep"] >> itsDetectorParams->adaptiveThreshWinSizeStep;
      fs["adaptiveThreshConstant"] >> itsDetectorParams->adaptiveThreshConstant;
      fs["minMarkerPerimeterRate"] >> itsDetectorParams->minMarkerPerimeterRate;
      fs["maxMarkerPerimeterRate"] >> itsDetectorParams->maxMarkerPerimeterRate;
      fs["polygonalApproxAccuracyRate"] >> itsDetectorParams->polygonalApproxAccuracyRate;
      fs["minCornerDistanceRate"] >> itsDetectorParams->minCornerDistanceRate;
      fs["minDistanceToBorder"] >> itsDetectorParams->minDistanceToBorder;
      fs["minMarkerDistanceRate"] >> itsDetectorParams->minMarkerDistanceRate;
      fs["doCornerRefinement"] >> itsDetectorParams->doCornerRefinement;
      fs["cornerRefinementWinSize"] >> itsDetectorParams->cornerRefinementWinSize;
      fs["cornerRefinementMaxIterations"] >> itsDetectorParams->cornerRefinementMaxIterations;
      fs["cornerRefinementMinAccuracy"] >> itsDetectorParams->cornerRefinementMinAccuracy;
      fs["markerBorderBits"] >> itsDetectorParams->markerBorderBits;
      fs["perspectiveRemovePixelPerCell"] >> itsDetectorParams->perspectiveRemovePixelPerCell;
      fs["perspectiveRemoveIgnoredMarginPerCell"] >> itsDetectorParams->perspectiveRemoveIgnoredMarginPerCell;
      fs["maxErroneousBitsInBorderRate"] >> itsDetectorParams->maxErroneousBitsInBorderRate;
      fs["minOtsuStdDev"] >> itsDetectorParams->minOtsuStdDev;
      fs["errorCorrectionRate"] >> itsDetectorParams->errorCorrectionRate;
    }
    else LERROR("Failed to read detector parameters from file [" << dpf << "] -- IGNORED");
  }
  
  // Instantiate the disctionary:
  switch (aruco::dictionary::get())
  {
  case aruco::Dict::Original: itsDictionary = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_ARUCO_ORIGINAL);break;
  case aruco::Dict::D4X4_50: itsDictionary = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_4X4_50); break;
  case aruco::Dict::D4X4_100: itsDictionary = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_4X4_100); break;
  case aruco::Dict::D4X4_250: itsDictionary = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_4X4_250); break;
  case aruco::Dict::D4X4_1000: itsDictionary = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_4X4_1000); break;
  case aruco::Dict::D5X5_50: itsDictionary = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_5X5_50); break;
  case aruco::Dict::D5X5_100: itsDictionary = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_5X5_100); break;
  case aruco::Dict::D5X5_250: itsDictionary = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_5X5_250); break;
  case aruco::Dict::D5X5_1000: itsDictionary = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_5X5_1000); break;
  case aruco::Dict::D6X6_50: itsDictionary = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_6X6_50); break;
  case aruco::Dict::D6X6_100: itsDictionary = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_6X6_100); break;
  case aruco::Dict::D6X6_250: itsDictionary = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_6X6_250); break;
  case aruco::Dict::D6X6_1000: itsDictionary = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_6X6_1000); break;
  case aruco::Dict::D7X7_50: itsDictionary = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_7X7_50); break;
  case aruco::Dict::D7X7_100: itsDictionary = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_7X7_100); break;
  case aruco::Dict::D7X7_250: itsDictionary = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_7X7_250); break;
  case aruco::Dict::D7X7_1000: itsDictionary = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_7X7_1000); break;
  }
}

// ##############################################################################################################
void ArUco::postUninit()
{
  aruco::camparams::unFreeze();
  aruco::detparams::unFreeze();
  itsDictionary.release();
  itsDetectorParams.release();
  itsCamMatrix = cv::Mat();
  itsDistCoeffs = cv::Mat();
}

// ##############################################################################################################
void ArUco::detectMarkers(cv::InputArray image, cv::OutputArray ids, cv::OutputArrayOfArrays corners)
{
  cv::aruco::detectMarkers(image, itsDictionary, corners, ids, itsDetectorParams);
}

// ##############################################################################################################
void ArUco::estimatePoseSingleMarkers(cv::InputArrayOfArrays corners, float markerLength,
                                      cv::OutputArray rvecs, cv::OutputArray tvecs)
{
  cv::aruco::estimatePoseSingleMarkers(corners, markerLength, itsCamMatrix, itsDistCoeffs, rvecs, tvecs);
}


