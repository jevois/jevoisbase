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

#include <jevoisbase/Components/FaceDetection/FaceDetector.H>
#include <jevois/Debug/Log.H>

// ####################################################################################################
FaceDetector::FaceDetector(std::string const & instance) :
    jevois::Component(instance)
{ }

// ####################################################################################################
FaceDetector::~FaceDetector()
{ }

// ####################################################################################################
void FaceDetector::postInit()
{
  LINFO("my path is " << absolutePath());
  std::string const & facename = facedetector::face_cascade::get();
  if (facename.empty()) LFATAL("face_cascade paremeter cannot be empty");
  std::string const facefile = absolutePath(facename);
  itsFaceCascade.reset(new cv::CascadeClassifier(facefile));
  if (itsFaceCascade->empty()) LFATAL("Error loading face cascade file " << facefile);
  
  std::string const & eyename = facedetector::eye_cascade::get();
  if (eyename.empty() == false)
  {
    std::string const eyefile = absolutePath(eyename);
    itsEyesCascade.reset(new cv::CascadeClassifier(eyefile));
    if (itsEyesCascade->empty()) LFATAL("Error loading eye cascade file " << eyefile);
  }
}

// ####################################################################################################
void FaceDetector::process(cv::Mat const & img, std::vector<cv::Rect> & faces,
                           std::vector<std::vector<cv::Rect> > & eyes, bool detect_eyes)
{
  // Clear any input junk:
  faces.clear();
  eyes.clear();
  
  // First, detect the faces:
  itsFaceCascade->detectMultiScale(img, faces, 1.1, 2, 0 | CV_HAAR_SCALE_IMAGE, cv::Size(img.cols / 2, img.rows / 2));

  // Create one entry in eyes vector for each face:
  eyes.resize(faces.size());
  
  // Then, for each face, detect the eyes:
  if (detect_eyes)
    for (size_t i = 0; i < faces.size(); ++i)
    {
      // Get a crop around this face:
      cv::Mat faceROI = img(faces[i]);
      
      // Detect eyes in the ROI:
      itsEyesCascade->detectMultiScale(faceROI, eyes[i], 1.1, 2, 0 | CV_HAAR_SCALE_IMAGE,
                                       cv::Size(img.cols / 8, img.rows / 8), cv::Size(img.cols / 2, img.rows / 2));
    }
}

