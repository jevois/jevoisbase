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

#pragma once

#include <jevois/Component/Component.H>
#include <jevois/Types/Enum.H>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

namespace superpixel
{
  static jevois::ParameterCategory const ParamCateg("SuperPixel Options");

  //! Enum for parameter \relates SuperPixel
  JEVOIS_DEFINE_ENUM_CLASS(Algo, (SLIC) (SLICO) (SEEDS) ); // could later also support LSC

  //! Parameter \relates SuperPixel
  JEVOIS_DECLARE_PARAMETER(algo, Algo, "Algorithm variant to use", Algo::SLICO, Algo_Values, ParamCateg);

  //! Parameter \relates SuperPixel
  JEVOIS_DECLARE_PARAMETER(regionsize, int, "SLIC Region size", 30, ParamCateg);

  //! Parameter \relates SuperPixel
  JEVOIS_DECLARE_PARAMETER(numpix, int, "Nubmer of SEEDS superpixels", 200, ParamCateg);

  //! Parameter \relates SuperPixel
  JEVOIS_DECLARE_PARAMETER(iterations, int, "Number of iterations", 6, jevois::Range<int>(1, 100), ParamCateg);

  //! Enum for parameter \relates SuperPixel
  JEVOIS_DEFINE_ENUM_CLASS(OutType, (Labels) (Contours) );

  //! Parameter \relates SuperPixel
  JEVOIS_DECLARE_PARAMETER(output, OutType, "Output labels or demo grayscale input image with superimposed contours",
                           OutType::Contours, OutType_Values, ParamCateg);
} // namespace superpixel

//! Superpixel image segmentation from OpenCV
/*! See tutorial at http://docs.opencv.org/3.2.0/df/d6c/group__ximgproc__superpixel.html
    and http://docs.opencv.org/3.0-beta/modules/ximgproc/doc/superpixels.html \ingroup components */
class SuperPixel : public jevois::Component,
                   public jevois::Parameter<superpixel::algo, superpixel::regionsize, superpixel::numpix,
                                            superpixel::iterations, superpixel::output>
{
  public:
    //! Default Component constructor ok
    using jevois::Component::Component;

    //! Virtual destructor for safe inheritance
    virtual ~SuperPixel() { }

    //! Process an RGB image and get some greyscale results
    /*! outimg should be allocated with same size as the input and CV_8UC1 pixels. */
    void process(cv::Mat const & inimg, cv::Mat & outimg);
};


  
