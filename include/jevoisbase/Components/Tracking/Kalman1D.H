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

#include <jevois/Component/Component.H>
#include <opencv2/video/tracking.hpp> // for kalman filter

#pragma once

namespace kalman1d
{
  static jevois::ParameterCategory const ParamCateg("Kalman1D Options");

  //! Parameter \relates Kalman1D
  JEVOIS_DECLARE_PARAMETER_WITH_CALLBACK(usevel, bool, "Use velocity tracking, in addition to position",
                                         false, ParamCateg);
  
  //! Parameter \relates Kalman1D
  JEVOIS_DECLARE_PARAMETER_WITH_CALLBACK(procnoise, float, "Process noise standard deviation",
                                         0.003F, ParamCateg);

  //! Parameter \relates Kalman1D
  JEVOIS_DECLARE_PARAMETER_WITH_CALLBACK(measnoise, float, "Measurement noise standard deviation",
                                         0.05F, ParamCateg);
  
  //! Parameter \relates Kalman1D
  JEVOIS_DECLARE_PARAMETER_WITH_CALLBACK(postnoise, float, "A posteriori error estimate standard deviation",
                                         0.3F, ParamCateg);
}

//! Simple component to track a moving 2D point over time using a Kalman filter
/*! A Kalman filter allows one to filter noisy data over time. Here, it is used to filter the possibly noisy results of
    some machine vision algorith that is trying to detect a particular item in the camera's video stream. On occasion,
    the detection might abruptly jump to an erroneous location. Such jumps are filtered out by the Kalman filter.

    Note that, unlike Kalman2D, this class does not perform any normalization or range transformation.

    This code loosely inspired by https://github.com/abreheret/kalman-mouse/blob/master/src/main.cpp

    \ingroup components */
class Kalman1D : public jevois::Component,
                 public jevois::Parameter<kalman1d::usevel, kalman1d::procnoise,
                                          kalman1d::measnoise, kalman1d::postnoise>
{
  public:
    //! Constructor
    Kalman1D(std::string const & instance);

    //! Destructor
    virtual ~Kalman1D();

    //! Function to call each time you have a new measurement
    void set(float x);

    //! Function to call each time you want to get the Kalman-filtered coordinates
    /*! It is ok to call get() multiple times with no intervening set(), if you have no new measurements but still want
        to use the filter output.

        eps is used for rounding of returned value, which is convenient to avoid sending very long floating point
        values over serial port. */
    float get(float const eps = 0.01F);

  protected:
    void postInit() override;
    void onParamChange(kalman1d::usevel const & param, bool const & newval) override;
    void onParamChange(kalman1d::procnoise const & param, float const & newval) override;
    void onParamChange(kalman1d::measnoise const & param, float const & newval) override;
    void onParamChange(kalman1d::postnoise const & param, float const & newval) override;
    
    // state is (x, xdot)
    cv::KalmanFilter itsKF;
    cv::Mat itsProcessNoise;
    cv::Mat itsMeasurement;
    cv::Mat itsLatest;
    bool itsFresh;

}; // class Kalman1D

