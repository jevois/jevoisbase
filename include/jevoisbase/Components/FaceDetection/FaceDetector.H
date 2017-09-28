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

#include <opencv2/objdetect/objdetect.hpp>

namespace facedetector
{
  static jevois::ParameterCategory const ParamCateg("FaceDetector Options");

  //! Parameter \relates FaceDetector
  JEVOIS_DECLARE_PARAMETER(face_cascade, std::string, "File name of the face cascade",
                           JEVOIS_SHARE_PATH "/facedetector/haarcascade_frontalface_alt.xml", ParamCateg);

  //! Parameter \relates FaceDetector
  JEVOIS_DECLARE_PARAMETER(eye_cascade, std::string, "File name of the eye cascade, or empty to not detect eyes",
                           JEVOIS_SHARE_PATH "/facedetector/haarcascade_eye_tree_eyeglasses.xml", ParamCateg);
}

//! Face detection using OpenCV
/*! See tutorial at http://docs.opencv.org/trunk/d7/d8b/tutorial_py_face_detection.html
    \ingroup components */
class FaceDetector : public jevois::Component,
                     public jevois::Parameter<facedetector::face_cascade, facedetector::eye_cascade>
{
  public:
    //! Constructor, loads the cascade classifiers from disk
    FaceDetector(std::string const & instance);
    
    //! Destructor
    ~FaceDetector();
    
    //! Process an image, results are held in our data members
    /*! The given image should be grayscale and with histogram equalized, for example by doing:
        \code
        // Given a color image in 'frame'
        cv::Mat frame_gray;
        cv::cvtColor(frame, frame_gray, CV_BGR2GRAY);
        cv::equalizeHist(frame_gray, frame_gray);
        // Then send frame_gray to FaceDetector::process()
        \endcode 

        If detect_eyes is true, throws if an eye cascade was not successfully loaded. Even if we do not detect eyes, the
        eyes vector will be resized to one (possibly empty) vector of eyes per face, so that it is safe to assume that
        one entry exists in eyes for each entry in faces. */
    void process(cv::Mat const & img, std::vector<cv::Rect> & faces, std::vector<std::vector<cv::Rect> > & eyes,
                 bool detect_eyes = false);
    
  protected:
    void postInit() override;
    std::shared_ptr<cv::CascadeClassifier> itsFaceCascade;
    std::shared_ptr<cv::CascadeClassifier> itsEyesCascade;
};
