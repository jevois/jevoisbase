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

#include <jevoisbase/Components/Filters/MorphologyFilter.H>

#include <opencv2/imgproc/imgproc.hpp>

// ##############################################################################################################
MorphologyFilter::~MorphologyFilter()
{ }

// ##############################################################################################################
std::string MorphologyFilter::process(cv::Mat const & src, cv::Mat & dst)
{
  cv::Mat kernel;
  
  switch (kshape::get())
  {
  case morphologyfilter::KernelShape::Rectangle:
    kernel = cv::getStructuringElement(cv::MORPH_RECT, ksize::get()); break;
  case morphologyfilter::KernelShape::Cross:
    kernel = cv::getStructuringElement(cv::MORPH_CROSS, ksize::get()); break;
  case morphologyfilter::KernelShape::Ellipse:
    kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, ksize::get()); break;
  }

  cv::MorphTypes mop;

  switch (op::get())
  {
  case morphologyfilter::MorphoOp::Erode: mop = cv::MORPH_ERODE; break;
  case morphologyfilter::MorphoOp::Dilate: mop = cv::MORPH_DILATE; break;
  case morphologyfilter::MorphoOp::Open: mop = cv::MORPH_OPEN; break;
  case morphologyfilter::MorphoOp::Close: mop = cv::MORPH_CLOSE; break;
  case morphologyfilter::MorphoOp::Gradient: mop = cv::MORPH_GRADIENT; break;
  case morphologyfilter::MorphoOp::TopHat: mop = cv::MORPH_TOPHAT; break;
  case morphologyfilter::MorphoOp::BlackHat: mop = cv::MORPH_BLACKHAT; break;
  }
  
  cv::morphologyEx(src, dst, mop, kernel, anchor::get(), iter::get());

  return "op=" + op::strget() + ", kshape=" + kshape::strget() + ",\nksize=[" + ksize::strget() +
    "], anchor=[" + anchor::strget() + "], iter=" + iter::strget();
}

