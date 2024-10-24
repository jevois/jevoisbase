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

namespace kalman4d
{
  static jevois::ParameterCategory const ParamCateg("Kalman4D Options");

  //! Parameter \relates Kalman4D
  JEVOIS_DECLARE_PARAMETER_WITH_CALLBACK(usevel, bool, "Use velocity tracking, in addition to position",
                                         false, ParamCateg);
  
  //! Parameter \relates Kalman4D
  JEVOIS_DECLARE_PARAMETER_WITH_CALLBACK(procnoise, float, "Process noise standard deviation",
                                         0.003F, ParamCateg);

  //! Parameter \relates Kalman4D
  JEVOIS_DECLARE_PARAMETER_WITH_CALLBACK(measnoise, float, "Measurement noise standard deviation",
                                         0.05F, ParamCateg);
  
  //! Parameter \relates Kalman4D
  JEVOIS_DECLARE_PARAMETER_WITH_CALLBACK(postnoise, float, "A posteriori error estimate standard deviation",
                                         0.3F, ParamCateg);
}

//! Simple component to track a moving 2D box (position and size) over time using a Kalman filter
/*! A Kalman filter allows one to filter noisy data over time. Here, it is used to filter the possibly noisy results of
    some machine vision algorith that is trying to detect a particular item in the camera's video stream. On occasion,
    the detection might abruptly jump to an erroneous location. Such jumps are filtered out by the Kalman filter.

    See for example DemoSaliency to see the Kalman4d filtering over time the location of the most salient object in the
    video stream. The 4D version is to track a box which is assumed to be moving but of roughly constant size.

    To allow using this component in many different modules and to support various image resolutions, we here internally
    normalize all measurements and filtered outputs to between [-1000.0 ... 1000.0] horizontally and [-750 ... 750
    vertically, where (-1000.0, -750.0) is the top-left of the image, (0.0, 0.0) the center, and (1000.0, 750.0) the
    bottom-right corner.

    Helper functions are used to do this conversion, see \ref coordhelpers.

    The range [-1000.0 ... 1000.0] is fixed and is not exposed as a parameter so that users of the Kalman4D class will
    not be tempted to use various ranges, in an effort to standardize the interface between vision modules (which may
    track object locations using Kalman4D) and motor control (e.g., an Arduino controlling servos on a pan/tilt camera
    head). The writers of motor control code can reliably assume that object coordinates sent to them will be in the
    [-1000.0 ... 1000.0] horizontal and [-750 ... 750] vertical range with 0,0 at dead center, and control their motors
    accordingly.

    This code loosely inspired by https://github.com/abreheret/kalman-mouse/blob/master/src/main.cpp

    \ingroup components */
class Kalman4D : public jevois::Component,
                 public jevois::Parameter<kalman4d::usevel, kalman4d::procnoise,
                                          kalman4d::measnoise, kalman4d::postnoise>
{
  public:
    //! Constructor
    Kalman4D(std::string const & instance);

    //! Destructor
    virtual ~Kalman4D();

    //! Function to call each time you have a new measurement (output from a vision algorithm)
    /*! This variant does not normalize the coordinates, they should typically be in [-1000.0 .. 1000.0] */
    void set(float x, float y, float w, float h);

    //! Function to call each time you have a new measurement (output from a vision algorithm)
    /*! This variant normalizes the image coordinates to the range [-1000.0 .. 1000.0] horizontal and [-750 ... 750]
        vertical. */
    void set(float x, float y, float w, float h, unsigned int imgwidth, unsigned int imgheight);
    
    //! Function to call each time you want to get the Kalman-filtered coordinates
    /*! It is ok to call get() multiple times with no intervening set(), if you have no new measurements but still want
        to use the filter output. This variant does not normalize the coordinates, they should typically be in [-1000.0
        .. 1000.0] range.

        eps is used for rounding of returned coordinates, which is convenient to avoid sending very long floating point
        values over serial port. */
    void get(float & x, float & y, float & w, float & h, float const eps = 0.01F);

    //! Function to call each time you want to get the Kalman-filtered coordinates
    /*! It is ok to call get() multiple times with no intervening set(), if you have no new measurements but still want
        to use the filter output. This variant normalizes the coordinates back to image coordinates in [0 .. imgwidth[ x
        [0 .. imgheight[ */
    void get(float & x, float & y, float & w, float & h, unsigned int imgwidth, unsigned int imgheight,
             float const eps = 0.01F);

    //! Function to call each time you want to get the Kalman-filtered coordinates
    /*! It is ok to call get() multiple times with no intervening set(), if you have no new measurements but still want
        to use the filter output. This variant returns both the coordinates in [0 .. 1000] range (typically, so send to
        an Arduino for servo control, or other motor plant), and the image coordinates in [0 .. imgwidth[ x [0
        .. imgheight[ (typically, to draw on a debug/demo image). */
    void get(float & rawx, float & rawy, float & raww, float & rawh, float & imgx, float & imgy,
             float & imgw, float & imgh, unsigned int imgwidth, unsigned int imgheight,
             float const raweps = 0.01F, float const imgeps = 0.01F);

    //! Get the latest error covariance matrix (errorCovPost). It is updated after each get() or set()
    cv::Mat const & getErrorCov() const;
    
  protected:
    void postInit() override;
    void onParamChange(kalman4d::usevel const & param, bool const & newval) override;
    void onParamChange(kalman4d::procnoise const & param, float const & newval) override;
    void onParamChange(kalman4d::measnoise const & param, float const & newval) override;
    void onParamChange(kalman4d::postnoise const & param, float const & newval) override;
    
    // state is (x, y, width, height, xdot, ydot)
    cv::KalmanFilter itsKF;
    cv::Mat itsProcessNoise;
    cv::Mat itsMeasurement;
    cv::Mat itsLatest;
    bool itsFresh;

}; // class Kalman4D

