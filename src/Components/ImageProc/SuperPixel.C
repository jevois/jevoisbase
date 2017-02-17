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

#include <jevoisbase/src/Components/ImageProc/SuperPixel.H>

#include <opencv2/ximgproc/slic.hpp>
#include <opencv2/ximgproc/seeds.hpp>

void SuperPixel::process(cv::Mat const & inimg, cv::Mat & outimg)
{
  if (inimg.rows != outimg.rows || inimg.cols != outimg.cols || inimg.type() != CV_8UC3 || outimg.type() != CV_8UC1)
    LFATAL("Need RGB byte input and gray byte output images of same dims");
  
  // Process depending on the algo chosen by the user:
  switch (algo::get())
  {
  case superpixel::Algo::SLIC:
  case superpixel::Algo::SLICO:
  {
    auto sp = cv::ximgproc::createSuperpixelSLIC(inimg, algo::get() == superpixel::Algo::SLIC ?
                                                 cv::ximgproc::SLIC : cv::ximgproc::SLICO,
                                                 regionsize::get());

    sp->iterate(iterations::get());

    sp->enforceLabelConnectivity(25);

    cv::Mat res; sp->getLabels(res);
    
    // The labels are in CV_32SC1 format, so we convert them to byte:
    switch (output::get())
    {
    case superpixel::OutType::Labels:
    {
      if (sp->getNumberOfSuperpixels() > 255) LERROR("More than 255 superpixels, graylevel confusion will occur");

      int const * src = reinterpret_cast<int const *>(res.data);
      unsigned char * dst = reinterpret_cast<unsigned char *>(outimg.data);
      int const siz = inimg.rows * inimg.cols;

      for (int i = 0; i < siz; ++i) *dst++ = *src++;
    }
    break;
    
    case superpixel::OutType::Contours:
    { 
      cv::cvtColor(inimg, outimg, CV_RGB2GRAY);

      cv::Mat mask; sp->getLabelContourMask(mask, false);

      // in OpenCV 3.1 release, the mask is now created as 8SC1 image, with -1 on contours, and that
      // does not work with setTo(), so let's set those contour pixels by hand:
      char const * s = reinterpret_cast<char const *>(mask.data);
      unsigned char * d = outimg.data; int sz = mask.total();
      for (int i = 0; i < sz; ++i) { if (*s < 0) *d = 255; ++s; ++d; }
    }
    break;
    }
  }
  break;

  case superpixel::Algo::SEEDS:
  {
    auto sp = cv::ximgproc::createSuperpixelSEEDS(inimg.cols, inimg.rows, inimg.channels(), numpix::get(),
                                                  4, 2, 5, false);
    
    sp->iterate(inimg, iterations::get());
    
    cv::Mat res; sp->getLabels(res);
    
    // The labels are in CV_32SC1 format, so we convert them to byte:
    switch (output::get())
    {
    case superpixel::OutType::Labels:
    {
      if (sp->getNumberOfSuperpixels() > 255) LERROR("More than 255 superpixels, graylevel confusion will occur");

      int const * src = reinterpret_cast<int const *>(res.data);
      unsigned char * dst = reinterpret_cast<unsigned char *>(outimg.data);
      int const siz = inimg.rows * inimg.cols;

      for (int i = 0; i < siz; ++i) *dst++ = *src++;
    }
    break;
    
    case superpixel::OutType::Contours:
    { 
      cv::cvtColor(inimg, outimg, CV_RGB2GRAY);

      cv::Mat mask; sp->getLabelContourMask(mask, false);

      // in OpenCV 3.1 release, the mask is now created as 8SC1 image, with -1 on contours, and that
      // does not work with setTo(), so let's set those contour pixels by hand:
      char const * s = reinterpret_cast<char const *>(mask.data);
      unsigned char * d = outimg.data; int sz = mask.total();
      for (int i = 0; i < sz; ++i) { if (*s < 0) *d = 255; ++s; ++d; }
    }
    break;
    }
  }
  break;

  default: LFATAL("oops");
  }
}


