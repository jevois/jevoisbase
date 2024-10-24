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

#include <jevoisbase/Components/Tracking/Kalman4D.H>
#include <jevois/Util/Coordinates.H>
#include <cmath>

// ####################################################################################################
Kalman4D::Kalman4D(std::string const & instance) :
    jevois::Component(instance), itsKF(6, 4, 0), itsProcessNoise(6, 1, CV_32F),
    itsMeasurement(4, 1, CV_32F), itsLatest(4, 1, CV_32F), itsFresh(false)
{
  cv::setIdentity(itsKF.measurementMatrix);
}

// ####################################################################################################
Kalman4D::~Kalman4D()
{ }

// ####################################################################################################
void Kalman4D::postInit()
{
  itsKF.statePost.at<float>(0) = 0.0F; // middle of screen
  itsKF.statePost.at<float>(1) = 0.0F; // middle of screen
  itsKF.statePost.at<float>(2) = 1.0F; // small width
  itsKF.statePost.at<float>(3) = 1.0F; // small height
  itsKF.statePost.at<float>(4) = 0.0F; // not moving
  itsKF.statePost.at<float>(5) = 0.0F; // not moving

  // Predict once to get rolling:
  itsKF.predict();
}

// ####################################################################################################
void Kalman4D::onParamChange(kalman4d::usevel const & JEVOIS_UNUSED_PARAM(param), bool const & newval)
{
  if (newval)
    itsKF.transitionMatrix = (cv::Mat_<float>(6, 6) <<
                              1,0,0,0,1,0, // x updated by xdot
                              0,1,0,0,0,1, // y updated by ydot
                              0,0,1,0,0,0, // constant width
                              0,0,0,1,0,0, // constant height
                              0,0,0,0,1,0, // constant xdot
                              0,0,0,0,0,1  // constant ydot
                              );
  else
    itsKF.transitionMatrix = (cv::Mat_<float>(6, 6) <<
                              1,0,0,0,0,0, // constant x
                              0,1,0,0,0,0, // constant y
                              0,0,1,0,0,0, // constant width
                              0,0,0,1,0,0, // constant height
                              0,0,0,0,1,0, // constant xdot
                              0,0,0,0,0,1  // constant ydot
                              );
}  

// ####################################################################################################
void Kalman4D::onParamChange(kalman4d::procnoise const & JEVOIS_UNUSED_PARAM(param), float const & newval)
{
  float constexpr e = 0.01F; // epsilon squared
  float const n = newval * newval; // desired variance
  itsKF.processNoiseCov = (cv::Mat_<float>(6, 6) << 
                           n, 0, 0, 0, 0, 0,
                           0, n, 0, 0, 0, 0,
                           0, 0, n, 0, 0, 0,
                           0, 0, 0, n, 0, 0,
                           0, 0, 0, 0, e, 0,
                           0, 0, 0, 0, 0, e );
}

// ####################################################################################################
void Kalman4D::onParamChange(kalman4d::measnoise const & JEVOIS_UNUSED_PARAM(param), float const & newval)
{
  cv::setIdentity(itsKF.measurementNoiseCov, cv::Scalar::all(newval * newval));
}

// ####################################################################################################
void Kalman4D::onParamChange(kalman4d::postnoise const & JEVOIS_UNUSED_PARAM(param), float const & newval)
{
  cv::setIdentity(itsKF.errorCovPost, cv::Scalar::all(newval * newval));
}

// ####################################################################################################
void Kalman4D::set(float x, float y, float w, float h)
{
  itsMeasurement.at<float>(0) = x; itsMeasurement.at<float>(1) = y;
  itsMeasurement.at<float>(2) = w; itsMeasurement.at<float>(3) = h;
  itsLatest = itsKF.correct(itsMeasurement);
  itsFresh = true; // itsLatest is "fresh", no need to predict() at the next get()
}

// ####################################################################################################
void Kalman4D::set(float x, float y, float w, float h, unsigned int imgwidth, unsigned int imgheight)
{
  jevois::coords::imgToStd(x, y, imgwidth, imgheight, 0.0F);
  jevois::coords::imgToStdSize(w, h, imgwidth, imgheight, 0.0F);
  this->set(x, y, w, h);
}

// ####################################################################################################
void Kalman4D::get(float & x, float & y, float & w, float & h, float const eps)
{
  // If we just had a fresh set(), use our latest result form correct(), otherwise predict the next position:
  if (itsFresh)
  {
    x = itsLatest.at<float>(0);
    y = itsLatest.at<float>(1);
    w = itsLatest.at<float>(2);
    h = itsLatest.at<float>(3);
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
    w = itsLatest.at<float>(2);
    h = itsLatest.at<float>(3);
  }

  if (eps)
  {
    x = std::round(x / eps) * eps;
    y = std::round(y / eps) * eps;
    w = std::round(w / eps) * eps;
    h = std::round(h / eps) * eps;
  }
}

// ####################################################################################################
void Kalman4D::get(float & x, float & y, float & w, float & h, unsigned int imgwidth,
                   unsigned int imgheight, float const eps)
{
  this->get(x, y, w, h, 0.0F);
  jevois::coords::stdToImg(x, y, imgwidth, imgheight, eps);
  jevois::coords::imgToStdSize(w, h, imgwidth, imgheight, eps);
}

// ####################################################################################################
void Kalman4D::get(float & rawx, float & rawy, float & raww, float & rawh, float & imgx, float & imgy,
                   float & imgw, float & imgh, unsigned int imgwidth, unsigned int imgheight,
                   float const raweps, float const imgeps)
{
  // First get the raw coords with no rounding:
  this->get(rawx, rawy, raww, rawh, 0.0F);
  imgx = rawx;
  imgy = rawy;
  imgw = raww;
  imgh = rawh;
  
  // Now round the raw coords:
  if (raweps)
  {
    rawx = std::round(rawx / raweps) * raweps;
    rawy = std::round(rawy / raweps) * raweps;
    raww = std::round(raww / raweps) * raweps;
    rawh = std::round(rawh / raweps) * raweps;
  }

  // Now compute the image coords with rounding:
  jevois::coords::stdToImg(imgx, imgy, imgwidth, imgheight, imgeps);
  jevois::coords::stdToImgSize(imgw, imgh, imgwidth, imgheight, imgeps);
}

// ####################################################################################################
cv::Mat const & Kalman4D::getErrorCov() const
{ return itsKF.errorCovPost; }

