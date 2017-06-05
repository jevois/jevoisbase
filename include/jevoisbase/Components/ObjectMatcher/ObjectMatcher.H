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
#include <opencv2/features2d.hpp>

namespace objectmatcher
{
  static jevois::ParameterCategory const ParamCateg("Object Matcher Options");

  //! Parameter \relates ObjectMatcher
  JEVOIS_DECLARE_PARAMETER(hessian, double, "Hessian threshold", 800.0, ParamCateg);

  //! Parameter \relates ObjectMatcher
  JEVOIS_DECLARE_PARAMETER(traindir, std::string, "Directory where training images are", "images", ParamCateg);

  //! Parameter \relates ObjectMatcher
  JEVOIS_DECLARE_PARAMETER(goodpts, jevois::Range<size_t>, "Number range of good matches considered",
                           jevois::Range<size_t>(15, 100), ParamCateg);

  //! Parameter \relates ObjectMatcher
  JEVOIS_DECLARE_PARAMETER(distthresh, double, "Maximum distance for a match to be considered good",
                           0.2, ParamCateg);
}

//! Object matching using OpenCV keypoint detection and matching
/*! See tutorial at http://docs.opencv.org/3.0-beta/doc/py_tutorials/py_feature2d/py_surf_intro/py_surf_intro.html
    \ingroup components */
class ObjectMatcher : public jevois::Component,
                      public jevois::Parameter<objectmatcher::hessian, objectmatcher::traindir,
                                               objectmatcher::goodpts, objectmatcher::distthresh>
{
  public:
    //! Default inherited constructor ok
    using jevois::Component::Component;

    //! Destructor
    ~ObjectMatcher();
    
    //! Process a greyscale image, returns match score, object index, and bounding box corners for best match object
    /*! If return value is 1.0e30, then no good object match was found. This runs detect(), then compute() and then
        match(). One may also want to run those separately in some cases. */
    double process(cv::Mat const & img, size_t & trainidx, std::vector<cv::Point2f> & corners);

    //! Process a greyscale image, returns match score, object index
    /*! If return value is 1.0e30, then no good object match was found. This runs detect(), then compute() and then
        match(). One may also want to run those separately in some cases. */
    double process(cv::Mat const & img, size_t & trainidx);

    //! Load training images and compute keypoints and descriptors
    void postInit() override;

    //! Get number of training images
    size_t numtrain() const;

    //! Detect keypoints
    void detect(cv::Mat const & img, std::vector<cv::KeyPoint> & keypoints);

    //! Compute descriptors for given keypoints
    /*! \note keypoints is an input argument but opencv wants a non-const ref to it for some reason. */
    void compute(cv::Mat const & img, std::vector<cv::KeyPoint> & keypoints, cv::Mat & descriptors);

    //! Match given descriptors against those of our training images, return best match distance
    double match(std::vector<cv::KeyPoint> const & keypoints, cv::Mat const & descriptors,
                 size_t & trainidx, std::vector<cv::Point2f> & corners);

    //! Match given descriptors against those of our training images, return best match distance
    double match(std::vector<cv::KeyPoint> const & keypoints, cv::Mat const & descriptors, size_t & trainidx);

    //! Training data structure for ObjectMatcher
    /*! \relates ObjectMatcher */
    struct TrainData
    {
        std::string name;
        cv::Mat image;
        std::vector<cv::KeyPoint> keypoints;
        cv::Mat descriptors;
    };

    //! Get the training data for a given index
    TrainData const & traindata(size_t idx) const;

  private:
    struct MatchData
    {
        double avgdist;
        size_t trainidx;
        std::vector<cv::Point2f> corners;
    };
    
    MatchData matchcore(size_t corenum, std::vector<cv::KeyPoint> const & keypoints, cv::Mat const & descriptors,
                        size_t minidx, size_t maxidx, bool do_corners);
    
    cv::Ptr<cv::Feature2D> itsFeatureDetector;
    std::vector<cv::Ptr<cv::DescriptorMatcher> > itsMatcher; // one matcher per core
    std::vector<TrainData> itsTrainData;
};

