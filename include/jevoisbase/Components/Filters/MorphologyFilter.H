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
#include <opencv2/core.hpp>

namespace morphologyfilter
{
  static jevois::ParameterCategory const ParamCateg("Morphology Filter Parameters");

  //! Enum \relates MorphologyFilter
  JEVOIS_DEFINE_ENUM_CLASS(MorphoOp, (Erode) (Dilate) (Open) (Close) (Gradient) (TopHat) (BlackHat) (HitMiss) )

  //! Parameter \relates MorphologyFilter
  JEVOIS_DECLARE_PARAMETER(op, MorphoOp, "Morphological operation to apply",
                           MorphoOp::Close, MorphoOp_Values, ParamCateg);

  //! Enum \relates MorphologyFilter
  JEVOIS_DEFINE_ENUM_CLASS(KernelShape, (Rectangle) (Cross) (Ellipse) );

  //! Parameter \relates MorphologyFilter
  JEVOIS_DECLARE_PARAMETER(kshape, KernelShape, "Kernel shape",
                           KernelShape::Rectangle, KernelShape_Values, ParamCateg);

  //! Parameter \relates MorphologyFilter
  JEVOIS_DECLARE_PARAMETER(ksize, cv::Size, "Kernel width and height",
                           cv::Size(3, 3), ParamCateg);

  //! Parameter \relates MorphologyFilter
  JEVOIS_DECLARE_PARAMETER(anchor, cv::Point, "Anchor point x and y; default value [-1 -1] means that the anchor "
                           "is at the kernel center",
                           cv::Point(-1, -1), ParamCateg);

  //! Parameter \relates MorphologyFilter
  JEVOIS_DECLARE_PARAMETER(iter, unsigned int, "Number of iterations of the filter to apply",
                           1U, ParamCateg);
}

//! A morphology filter
/*! See http://docs.opencv.org/3.2.0/d4/d86/group__imgproc__filter.html for details. */
class MorphologyFilter : public Filter,
                         public jevois::Parameter<morphologyfilter::op, morphologyfilter::kshape,
                                                  morphologyfilter::ksize, morphologyfilter::anchor,
                                                  morphologyfilter::iter>
{
  public:
    //! Inherited constructor ok
    using Filter::Filter;

    //! Virtual destructor for safe inheritance
    virtual ~MorphologyFilter();

    //! Processing function: filters the image src and puts the results into dst
    virtual void process(cv::Mat const & src, cv::Mat & dst) override;
};
