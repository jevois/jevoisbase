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
#include <jevois/Debug/Profiler.H>
#include <opencv2/core/core.hpp>

namespace fastopticalflow
{
  static jevois::ParameterCategory const ParamCateg("Fast Optical Flow Options");

  //! Parameter \relates FastOpticalFlow
  JEVOIS_DECLARE_PARAMETER_WITH_CALLBACK(opoint, int, "Algorithm operating point",
                                         1, jevois::Range<int>(1, 4), ParamCateg);

  //! Parameter \relates FastOpticalFlow
  JEVOIS_DECLARE_PARAMETER(factor, float, "Factor to use when converting float velocities to byte",
                           25.0F, ParamCateg);

  //! Parameter \relates FastOpticalFlow
  JEVOIS_DECLARE_PARAMETER_WITH_CALLBACK(thetasf, int, "Finest scale in the pyramid, or -1 for auto",
                                         -1, ParamCateg);

  //! Parameter \relates FastOpticalFlow
  JEVOIS_DECLARE_PARAMETER(thetait, int, "Number of gradient descent iterations, or -1 for auto",
                           5, ParamCateg);

  //! Parameter \relates FastOpticalFlow
  JEVOIS_DECLARE_PARAMETER_WITH_CALLBACK(thetaps, int, "Patch size in pixels, or -1 for auto",
                                         -1, ParamCateg);

  //! Parameter \relates FastOpticalFlow
  JEVOIS_DECLARE_PARAMETER_WITH_CALLBACK(thetaov, float, "Patch overlap on each scale, or -1 for auto",
                                         0.1F, ParamCateg);
  
  //! Parameter \relates FastOpticalFlow
  JEVOIS_DECLARE_PARAMETER(usevref, bool, "Use variational refinement when true",
                           false, ParamCateg);
}

//! Fast optical flow computation using dense inverse search
/*! This algorithm computes what moved between two images (optical flow). It is based on the paper "Fast Optical Flow
    using Dense Inverse Search" by Till Kroeger, Radu Timofte, Dengxin Dai and Luc Van Gool, Proc ECCV, 2016. Also see
    here: http://www.vision.ee.ethz.ch/~kroegert/OFlow/ \ingroup components */
class FastOpticalFlow : public jevois::Component,
                        public jevois::Parameter<fastopticalflow::opoint, fastopticalflow::factor,
                                                 fastopticalflow::thetasf, fastopticalflow::thetait,
                                                 fastopticalflow::thetaps, fastopticalflow::thetaov,
                                                 fastopticalflow::usevref>
{
  public:
    //! Constructor
    FastOpticalFlow(std::string const & instance);
    
    //! Destructor
    virtual ~FastOpticalFlow();

    //! Process a greyscale image and return flow in a pre-allocated greyscale image of same with and 2x height
    /*! The results are the concatenation of 2 images: vx on top of vy, both converted to byte. */
    void process(cv::Mat const & src, cv::Mat & dst);

  protected:
    void onParamChange(fastopticalflow::opoint const & param, int const & val) override;
    void onParamChange(fastopticalflow::thetasf const & param, int const & val) override;
    void onParamChange(fastopticalflow::thetaps const & param, int const & val) override;
    void onParamChange(fastopticalflow::thetaov const & param, float const & val) override;
    
    jevois::Profiler itsProfiler;
    std::mutex itsMtx;

    bool itsNuke; // nuke all caches when some sizes or params change
    
    const float ** img_bo_pyr;  // those are just shallow pointers to the cv::Mat versions, no memory alloc
    const float ** img_bo_dx_pyr;
    const float ** img_bo_dy_pyr;
  
    cv::Mat * img_bo_fmat_pyr;
    cv::Mat * img_bo_dx_fmat_pyr;
    cv::Mat * img_bo_dy_fmat_pyr;

    cv::Mat img_bo_mat;
    cv::Mat img_bo_fmat;

    int itsHeight, itsWidth;
    int itsPyrDepth;
};
