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

#include <jevoisbase/Components/Tracking/Kalman2D.H>
#include <jevois/Util/Coordinates.H>
#include <cmath>

// ####################################################################################################
Kalman2D::Kalman2D(std::string const & instance) :
    jevois::Component(instance), itsKF(4, 2, 0), itsState(4, 1, CV_32F), itsProcessNoise(4, 1, CV_32F),
    itsMeasurement(2, 1, CV_32F), itsLatest(2, 1, CV_32F), itsFresh(false)
{
  cv::setIdentity(itsKF.measurementMatrix);
}

// ####################################################################################################
Kalman2D::~Kalman2D()
{ }

// ####################################################################################################
void Kalman2D::postInit()
{
  itsKF.statePost.at<float>(0) = 0.0F; // middle of screen
  itsKF.statePost.at<float>(1) = 0.0F; // middle of screen
  itsKF.statePost.at<float>(2) = 0.0F; // not moving
  itsKF.statePost.at<float>(3) = 0.0F; // not moving

  // Predict once to get rolling:
  itsKF.predict();
}

// ####################################################################################################
void Kalman2D::onParamChange(kalman2d::usevel const & JEVOIS_UNUSED_PARAM(param), bool const & newval)
{
  if (newval) itsKF.transitionMatrix = (cv::Mat_<float>(4, 4) << 1,0,1,0,   0,1,0,1,  0,0,1,0,  0,0,0,1);
  else        itsKF.transitionMatrix = (cv::Mat_<float>(4, 4) << 1,0,0,0,   0,1,0,0,  0,0,1,0,  0,0,0,1);
}  

// ####################################################################################################
void Kalman2D::onParamChange(kalman2d::procnoise const & JEVOIS_UNUSED_PARAM(param), float const & newval)
{
  itsKF.processNoiseCov = (cv::Mat_<float>(4, 4) << 
                           newval * newval, 0, 0, 0,
                           0, newval * newval, 0, 0,
                           0, 0, 0.01F, 0,
                           0, 0, 0, 0.01F );
}

// ####################################################################################################
void Kalman2D::onParamChange(kalman2d::measnoise const & JEVOIS_UNUSED_PARAM(param), float const & newval)
{
  cv::setIdentity(itsKF.measurementNoiseCov, cv::Scalar::all(newval * newval));
}

// ####################################################################################################
void Kalman2D::onParamChange(kalman2d::postnoise const & JEVOIS_UNUSED_PARAM(param), float const & newval)
{
  cv::setIdentity(itsKF.errorCovPost, cv::Scalar::all(newval * newval));
}

// ####################################################################################################
void Kalman2D::set(float x, float y)
{
  itsMeasurement.at<float>(0) = x; itsMeasurement.at<float>(1) = y;
  itsLatest = itsKF.correct(itsMeasurement);
  itsFresh = true; // itsLatest is "fresh", no need to predict() at the next get()
}

// ####################################################################################################
void Kalman2D::set(float x, float y, unsigned int imgwidth, unsigned int imgheight)
{
  jevois::coords::imgToStd(x, y, imgwidth, imgheight, 0.0F);
  this->set(x, y);
}

// ####################################################################################################
void Kalman2D::get(float & x, float & y, float const eps)
{
  // If we just had a fresh set(), use our latest result form correct(), otherwise predict the next position:
  if (itsFresh)
  {
    x = itsLatest.at<float>(0);
    y = itsLatest.at<float>(1);
    itsFresh = false;

    // Now predict and update itsLatest:
    itsLatest = itsKF.predict();
  }
  else
  {
    // First predict and update itsLatest:
    itsLatest = itsKF.predict();

    x = itsLatest.at<float>(0);
    y = itsLatest.at<float>(1);
  }

  if (eps) { x = std::round(x / eps) * eps; y = std::round(y / eps) * eps; }
}

// ####################################################################################################
void Kalman2D::get(float & x, float & y, unsigned int imgwidth, unsigned int imgheight, float const eps)
{
  this->get(x, y, 0.0F);
  jevois::coords::stdToImg(x, y, imgwidth, imgheight, eps);
}

// ####################################################################################################
void Kalman2D::get(float & rawx, float & rawy, float & imgx, float & imgy,
                   unsigned int imgwidth, unsigned int imgheight, float const raweps, float const imgeps)
{
  // First get the raw coords with no rounding:
  this->get(rawx, rawy, 0.0F);
  imgx = rawx;
  imgy = rawy;

  // Now round the raw coords:
  if (raweps) { rawx = std::round(rawx / raweps) * raweps; rawy = std::round(rawy / raweps) * raweps; }

  // Now compute the image coords with rounding:
  jevois::coords::stdToImg(imgx, imgy, imgwidth, imgheight, imgeps);
}
