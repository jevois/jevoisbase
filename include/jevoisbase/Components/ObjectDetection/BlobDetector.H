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

#pragma once

#include <jevois/Component/Component.H>
#include <opencv2/core/core.hpp>

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

namespace blobdetector
{
  static jevois::ParameterCategory const ParamCateg("BlobDetector Options");

  //! Parameter \relates ObjectTracker
  JEVOIS_DECLARE_PARAMETER(hrange, jevois::Range<unsigned char>, "Range of H values for HSV window (0=red/do not "
			   "use because of wraparound, 30=yellow, 45=light green, 60=green, 75=green cyan, 90=cyan, "
			   "105=light blue, 120=blue, 135=purple, 150=pink)",
			   jevois::Range<unsigned char>(10, 245), ParamCateg);
  
  //! Parameter \relates ObjectTracker
  JEVOIS_DECLARE_PARAMETER(srange, jevois::Range<unsigned char>, "Range of S (saturation) values for HSV window",
			   jevois::Range<unsigned char>(10, 245), ParamCateg);
  
  //! Parameter \relates ObjectTracker
  JEVOIS_DECLARE_PARAMETER(vrange, jevois::Range<unsigned char>, "Range of V (brightness) values for HSV window",
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
}


  
//! Simple color-based object/blob detection/tracking
/*! This component isolates pixels within a given HSV range (hue, saturation, and value of color pixels), does some
    cleanups, and extracts object contours. It sends information about object centers over serial.

    This algorithm usually works best with the camera sensor set to manual exposure, manual gain, manual color balance,
    etc so that HSV color values are reliable.

    This code was loosely inspired by:
    https://raw.githubusercontent.com/kylehounslow/opencv-tuts/master/object-tracking-tut/objectTrackingTut.cpp written
    by Kyle Hounslow, 2013. 

    For more information, see \jvmod{ObjectTracker}

    \ingroup components */
class BlobDetector : public jevois::Component,
		     public jevois::Parameter<blobdetector::hrange, blobdetector::srange, blobdetector::vrange,
					      blobdetector::maxnumobj, blobdetector::objectarea,
					      blobdetector::erodesize, blobdetector::dilatesize>
{
  public:
    
    //! Constructor
    using jevois::Component::Component;
    
    //! Virtual destructor for safe inheritance
    virtual ~BlobDetector();

    //! Detect blobs, each is represented as a contour (vector of (x,y) coordinates)
    std::vector<std::vector<cv::Point> > detect(cv::Mat const & imghsv);
};
