// ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//
// JeVois Smart Embedded Machine Vision Toolkit - Copyright (C) 2018 by Laurent Itti, the University of Southern
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

#include <jevoisbase/Components/ObjectDetection/BlobDetector.H>

#include <jevois/Image/RawImageOps.H>
#include <jevois/Util/Coordinates.H>
#include <opencv2/imgproc/imgproc.hpp>

// ##############################################################################################################
BlobDetector::~BlobDetector()
{ }

// ##############################################################################################################
std::vector<std::vector<cv::Point> > BlobDetector::detect(cv::Mat const & imghsv)
{
  std::vector<std::vector<cv::Point> > retcontours;

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
    for (int index = 0; index >= 0; index = hierarchy[index][0])
    {
      // Let's examine this contour:
      std::vector<cv::Point> const & c = contours[index];
      
      cv::Moments moment = cv::moments(c);
      double const area = moment.m00;

      // If this contour fits the bill, we will return it:
      if (objectarea::get().contains(int(area + 0.4999))) retcontours.push_back(c);
    }

  return retcontours;
}
