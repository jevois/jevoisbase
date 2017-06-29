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

#include <jevois/Core/Module.H>
#include <jevois/Types/Enum.H>
#include <jevois/Debug/Timer.H>

#include <opencv2/imgproc/imgproc.hpp>

#include <jevoisbase/Components/Filters/Filter.H>
#include <jevoisbase/Components/Filters/BlurFilter.H>
#include <jevoisbase/Components/Filters/MedianFilter.H>
#include <jevoisbase/Components/Filters/MorphologyFilter.H>
#include <jevoisbase/Components/Filters/LaplacianFilter.H>
#include <jevoisbase/Components/Filters/BilateralFilter.H>

static jevois::ParameterCategory const ParamCateg("Color Filtering Parameters");

//! Enum \relates ColorFiltering
JEVOIS_DEFINE_ENUM_CLASS(Effect, (NoEffect) (Blur) (Median) (Morpho) (Laplacian) (Bilateral) );

//! Parameter \relates ColorFiltering
JEVOIS_DECLARE_PARAMETER_WITH_CALLBACK(effect, Effect, "Image processing effect to apply",
                                       Effect::NoEffect, Effect_Values, ParamCateg);

//! Image filtering using OpenCV
/*! This modules is to learn about basic image filtering. It implements a variety of filters using OpenCV.

    @author Laurent Itti

    @videomapping YUYV 640 496 25.0 BAYER 640 480 25.0 JeVois ColorFiltering
    @email itti\@usc.edu
    @address University of Southern California, HNB-07A, 3641 Watt Way, Los Angeles, CA 90089-2520, USA
    @copyright Copyright (C) 2016 by Laurent Itti, iLab and the University of Southern California
    @mainurl http://jevois.org
    @supporturl http://jevois.org/doc
    @otherurl http://iLab.usc.edu
    @license GPL v3
    @distribution Unrestricted
    @restrictions None
    \ingroup modules */
class ColorFiltering : public jevois::Module,
                       public jevois::Parameter<effect>
{
  public:
    //! Inherited constructor ok
    using jevois::Module::Module;

    //! Virtual destructor for safe inheritance
    virtual ~ColorFiltering()
    { }

    //! Processing function with video output to USB
    virtual void process(jevois::InputFrame && inframe, jevois::OutputFrame && outframe) override
    {
      static jevois::Timer timer("processing", 30, LOG_DEBUG);

      // Wait for next available camera image, convert it to BGR, and release the camera buffer:
      cv::Mat src = inframe.getCvBGR();

      // Apply the filter (if any):
      timer.start();

      cv::Mat dst;
      if (itsFilter) itsFilter->process(src, dst); else dst = src;

      std::string const & fpscpu = timer.stop();

      // Write a few things:
      std::ostringstream oss; oss << "JeVois image filter: " << effect::get();
      cv::putText(dst, oss.str().c_str(), cv::Point(3, 13), cv::FONT_HERSHEY_PLAIN, 1.0, cv::Scalar(255,255,255));
      cv::putText(dst, fpscpu.c_str(), cv::Point(3, dst.rows-4), cv::FONT_HERSHEY_PLAIN, 1.0, cv::Scalar(255,255,255));
      
      // Send out the result image after converting it:
      outframe.sendCvBGR(dst);
    }

  protected:
    //! Parameter callback: set the selected filter algo
    void onParamChange(effect const & param, Effect const & val)
    {
      if (itsFilter) { removeSubComponent(itsFilter); itsFilter.reset(); }
      
      switch (val)
      {
      case Effect::NoEffect: break;
      case Effect::Blur: itsFilter = addSubComponent<BlurFilter>("filter"); break;
      case Effect::Median: itsFilter = addSubComponent<MedianFilter>("filter"); break;
      case Effect::Morpho: itsFilter = addSubComponent<MorphologyFilter>("filter"); break;
      case Effect::Laplacian: itsFilter = addSubComponent<LaplacianFilter>("filter"); break;
      case Effect::Bilateral: itsFilter = addSubComponent<BilateralFilter>("filter"); break;
      }
    }
    
  private:
    std::shared_ptr<Filter> itsFilter;
};

// Allow the module to be loaded as a shared object (.so) file:
JEVOIS_REGISTER_MODULE(ColorFiltering);
