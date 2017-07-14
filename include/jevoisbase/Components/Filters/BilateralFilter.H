// ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//
// JeVois Smart Embedded Machine Vision Toolkit - Copyright (C) 2017 by Laurent Itti, the University of Southern
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

#include <jevoisbase/Components/Filters/Filter.H>
#include <jevois/Types/Enum.H>

namespace bilateralfilter
{
  static jevois::ParameterCategory const ParamCateg("Bilateral Filter Parameters");

  //! Parameter \relates BilateralFilter
  JEVOIS_DECLARE_PARAMETER(d, int, "Diameter of each pixel neighborhood that is used during filtering. "
                           "If it is non-positive, it is computed from sigmaSpace",
                           0, ParamCateg);

  //! Parameter \relates BilateralFilter
  JEVOIS_DECLARE_PARAMETER(sigmaColor, double, "Filter sigma in the color space. A larger value of the parameter "
                           "means that farther colors within the pixel neighborhood (see sigmaSpace) will be mixed "
                           "together, resulting in larger areas of semi-equal color. ",
                           2.0, ParamCateg);

  //! Parameter \relates BilateralFilter
  JEVOIS_DECLARE_PARAMETER(sigmaSpace, double, "Filter sigma in the coordinate space. A larger value of the parameter "
                           "means that farther pixels will influence each other as long as their colors are close "
                           "enough (see sigmaColor). When d>0, it specifies the neighborhood size regardless of "
                           "sigmaSpace. Otherwise, d is proportional to sigmaSpace.",
                           2.0, ParamCateg);
}

//! A bilateral filter
/*! See http://docs.opencv.org/3.2.0/d4/d86/group__imgproc__filter.html for details. */
class BilateralFilter : public Filter,
                        public jevois::Parameter<bilateralfilter::d, bilateralfilter::sigmaColor,
                                                 bilateralfilter::sigmaSpace>
{
  public:
    //! Inherited constructor ok
    using Filter::Filter;

    //! Virtual destructor for safe inheritance
    virtual ~BilateralFilter();

    //! Processing function: filters the image src and puts the results into dst
    virtual std::string process(cv::Mat const & src, cv::Mat & dst) override;
};
