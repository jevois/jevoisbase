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
#include <opencv2/core/core.hpp>
#include <cvEyeTracker-1.2.5/ransac_ellipse.h>

namespace eyetracker
{
  static jevois::ParameterCategory const ParamCateg("Eye Tracker Options");

  //! Parameter \relates EyeTracker
  JEVOIS_DECLARE_PARAMETER(edgethresh, unsigned char, "Pupil edge threshold",
                           20, ParamCateg);

  //! Parameter \relates EyeTracker
  JEVOIS_DECLARE_PARAMETER(numrays, int, "Number of rays",
                           18, jevois::Range<int>(4, 180), ParamCateg);

  //! Parameter \relates EyeTracker
  JEVOIS_DECLARE_PARAMETER(mincand, int, "Minimum number of feature candidates",
                           10, jevois::Range<int>(1, 50), ParamCateg);

  //! Parameter \relates EyeTracker
  JEVOIS_DECLARE_PARAMETER(corneal, int, "Corneal window size",
                           51, /* FIXME enforce odd */ ParamCateg); // FIXME frameh
}

//! Eye-tracker class used to detect gaze direction from close-up video of one's eye
/*! The original eye tracking software used here can be found at http://thirtysixthspan.com/openEyes/software.html
    \ingroup components */
class EyeTracker : public jevois::Component,
                   public jevois::Parameter<eyetracker::edgethresh, eyetracker::numrays, eyetracker::mincand,
                                            eyetracker::corneal>
{
  public:
    //! Constructor
    EyeTracker(std::string const & instance);

    //! Destructor
    virtual ~EyeTracker();

    //! Process grayscale byte image from camera
    /*! Note: we will draw the detected ellipse and a few more things in the image.

        pupell will be filled with the pupil ellipse params, in the order used by opencv to draw ellipses: a, b, cx, cy,
        theta. */
    void process(cv::Mat & eyeimg, double pupell[5], bool debugdraw = false);

  private:
    int currWidth, currHeight;
    double *intensity_factor_hori;  //horizontal intensity factor for noise reduction
    double *avg_intensity_hori;  //horizontal average intensity
    stuDPoint start_point;
    double pupil_param[5];
    std::vector <stuDPoint> edge_point;
};
