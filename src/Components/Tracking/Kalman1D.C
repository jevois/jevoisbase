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

#include <jevoisbase/Components/Tracking/Kalman1D.H>
#include <jevois/Util/Coordinates.H>
#include <cmath>

// ####################################################################################################
Kalman1D::Kalman1D(std::string const & instance) :
    jevois::Component(instance), itsKF(2, 1, 0), itsState(2, 1, CV_32F), itsProcessNoise(2, 1, CV_32F),
    itsMeasurement(1, 1, CV_32F), itsLatest(1, 1, CV_32F), itsFresh(false)
{
  cv::setIdentity(itsKF.measurementMatrix);
}

// ####################################################################################################
Kalman1D::~Kalman1D()
{ }

// ####################################################################################################
void Kalman1D::postInit()
{
  itsKF.statePost.at<float>(0) = 0.0F; // middle of screen
  itsKF.statePost.at<float>(1) = 0.0F; // not moving

  // Predict once to get rolling:
  itsKF.predict();
}

// ####################################################################################################
void Kalman1D::onParamChange(kalman1d::usevel const & JEVOIS_UNUSED_PARAM(param), bool const & newval)
{
  if (newval) itsKF.transitionMatrix = (cv::Mat_<float>(2, 2) << 1,1,  0,1);
  else        itsKF.transitionMatrix = (cv::Mat_<float>(2, 2) << 1,0,  0,1);
}  

// ####################################################################################################
void Kalman1D::onParamChange(kalman1d::procnoise const & JEVOIS_UNUSED_PARAM(param), float const & newval)
{
  itsKF.processNoiseCov = (cv::Mat_<float>(2, 2) << newval * newval, 0,     0, newval * newval );
}

// ####################################################################################################
void Kalman1D::onParamChange(kalman1d::measnoise const & JEVOIS_UNUSED_PARAM(param), float const & newval)
{
  cv::setIdentity(itsKF.measurementNoiseCov, cv::Scalar::all(newval * newval));
}

// ####################################################################################################
void Kalman1D::onParamChange(kalman1d::postnoise const & JEVOIS_UNUSED_PARAM(param), float const & newval)
{
  cv::setIdentity(itsKF.errorCovPost, cv::Scalar::all(newval * newval));
}

// ####################################################################################################
void Kalman1D::set(float x)
{
  itsMeasurement.at<float>(0) = x;
  itsLatest = itsKF.correct(itsMeasurement);
  itsFresh = true; // itsLatest is "fresh", no need to predict() at the next get()
}

// ####################################################################################################
float Kalman1D::get(float const eps)
{
  // If we just had a fresh set(), use our latest result form correct(), otherwise predict the next position:
  float x;
  if (itsFresh)
  {
    x = itsLatest.at<float>(0);
    itsFresh = false;

    // Now predict and update itsLatest:
    itsLatest = itsKF.predict();
  }
  else
  {
    // First predict and update itsLatest:
    itsLatest = itsKF.predict();

    x = itsLatest.at<float>(0);
  }

  if (eps) x = std::round(x / eps) * eps;

  return x;
}

