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
#include <jevois/Types/Enum.H>
#include <opencv2/aruco.hpp>
namespace jevois { class StdModule; }

namespace aruco
{
  static jevois::ParameterCategory const ParamCateg("ArUco Options");

  //! Parameter \relates ArUco
  JEVOIS_DECLARE_PARAMETER(camparams, std::string, "File stem of camera parameters, or empty. Camera resolution "
                           "will be appended, as well as a .yaml extension. For example, specifying 'calibration' "
                           "here and running the camera sensor at 320x240 will attempt to load "
                           "calibration320x240.yaml from within directory " JEVOIS_SHARE_PATH "/camera/",
                           "calibration", ParamCateg);

  //! Parameter \relates ArUco
  JEVOIS_DECLARE_PARAMETER(detparams, std::string, "Filename of detector parameters, or empty", "", ParamCateg);

  //! Enum for parameter \relates ArUco
  JEVOIS_DEFINE_ENUM_CLASS(Dict, (Original) (D4X4_50) (D4X4_100) (D4X4_250) (D4X4_1000) (D5X5_50) (D5X5_100)
                           (D5X5_250) (D5X5_1000) (D6X6_50) (D6X6_100) (D6X6_250) (D6X6_1000) (D7X7_50)
                           (D7X7_100) (D7X7_250) (D7X7_1000) );
  
  //! Parameter \relates ArUco
  JEVOIS_DECLARE_PARAMETER(dictionary, Dict, "Symbol dictionary to use", Dict::D4X4_50, ParamCateg);

  //! Parameter \relates ArUco
  JEVOIS_DECLARE_PARAMETER(dopose, bool, "Compute (and show) pose vectors, requires a valid camera calibration",
                           false, ParamCateg);

  //! Parameter \relates ArUco
  JEVOIS_DECLARE_PARAMETER(markerlen, float, "Marker side length (millimeters), used only for pose estimation",
                           100.0F, ParamCateg);

  //! Parameter \relates ArUco
  JEVOIS_DECLARE_PARAMETER(showcube, bool, "Show a 3D cube on top of each detected marker, when dopose is also true",
                           false, aruco::ParamCateg);
}

//! Simple wrapper class over the opencv_contrib ArUco augmented reality markers
/*! ArUco markers are small 2D barcodes. Each ArUco marker corresponds to a number, encoded into a small grid of black
    and white pixels. The ArUco decoding algorithm is capable of locating, decoding, and of estimating the pose
    (location and orientation in space) of any ArUco markers in the camera's field of view.

    ArUcos are very useful as tags for many robotics and augmented reality applications. For example, one may place an
    ArUco next to a robot's charging station, an elevator button, or an object that a robot should manipulate.

    For more information about ArUco, see https://www.uco.es/investiga/grupos/ava/node/26

    The implementation of ArUco used by JeVois is the one of OpenCV-Contrib, documented here:
    http://docs.opencv.org/3.2.0/d5/dae/tutorial_aruco_detection.html

    ArUco markers can be created with several standard dictionaries. Different disctionaries give rise to different
    numbers of pixels in the markers, and to different numbers of possible symbols that can be created using the
    dictionary. The default dictionary used by JeVois is 4x4 with 50 symbols. Other dictionaries are also supported by
    setting the appropriate parameter over serial port or in a config file, up to 7x7 with 1000 symbols.

    Creating and printing ArUco markers
    -----------------------------------

    To create some markers, have a look at the samples here:
    https://github.com/opencv/opencv_contrib/tree/master/modules/aruco

    To make sure that the files you compile are compatible with JeVois which uses OpenCV 3.2.0 release, get that exact
    release from GitHub as opposed to just the latest version of those files:
    \verbatim
    wget wget https://github.com/opencv/opencv_contrib/archive/3.2.0.tar.gz
    tar zxvf 3.2.0.tar.gz
    # The files referenced below are in: opencv_contrib-3.2.0/modules/aruco/samples/
    \endverbatim

    To compile the sample program that can generate some ArUco markers (e.g., to be printed on paper):
    \code
    g++ -I/usr/share/jevois-opencv-3.2.0/include -L/usr/share/jevois-opencv-3.2.0/lib create_marker.cpp \\
        -o create_marker -lopencv_core -lopencv_imgcodecs -lopencv_highgui -lopencv_aruco
    \endcode

    Then, to make images of the markers in dictionary 0 (4x4_50):
    \code
    for id in {0..49}; do ./create_marker -d=0 --id=${id} aruco${id}.png; done
    \endcode

    You can then print them and later detect them using this Component. Make sure you select the same dictionary in the
    component as you did when you generated the markers.

    Recovering 3D pose of markers
    -----------------------------

    To enable recovery of the 3D pose of a marker, you need to calibrate your camera. Again using the sample code:

    \code
    g++ -I/usr/share/jevois-opencv-3.2.0/include -L/usr/share/jevois-opencv-3.2.0/lib create_board_charuco.cpp \\
        -o create_board_charuco -lopencv_core -lopencv_imgcodecs -lopencv_highgui -lopencv_aruco -lopencv_imgproc -lopencv_videoio

    g++ -I/usr/share/jevois-opencv-3.2.0/include -L/usr/share/jevois-opencv-3.2.0/lib calibrate_camera_charuco.cpp \\
        -o calibrate_camera_charuco -lopencv_core -lopencv_imgcodecs -lopencv_highgui -lopencv_aruco -lopencv_imgproc -lopencv_videoio
    \endcode

    To create a ChArUco board that can be printed, using dictionary 0, and then derive the camera parameters from it
    (set the `--ml` and `--sl` parameters to what you measure on your printed board; below are what we measured after
    printing the 5x8 charuco board on US Letter paper with auto scaling/rotate to fit paper):

    \code
    ./create_board_charuco -d=0 -h=5 -w=8 --ml=200 --sl=350 charuco.png
    ./calibrate_camera_charuco -d=0 -h=5 -w=8 --ml=.0172 --sl=.0303 --rs --sc calibration.yaml
    \endcode

    \ingroup components */
class ArUco : public jevois::Component,
              public jevois::Parameter<aruco::camparams, aruco::detparams, aruco::dictionary,
                                       aruco::dopose, aruco::markerlen, aruco::showcube>
{
  public:
    //! Constructor
    using jevois::Component::Component;
    
    //! Destructor
    virtual ~ArUco();
    
    //! Initialize, create the detector and read the config files
    void postInit() override;
    
    //! Un-initialize, nuke allocated resources
    void postUninit() override;

    //! Detect markers
    void detectMarkers(cv::InputArray image, cv::OutputArray ids, cv::OutputArrayOfArrays corners);

    //! Estimate pose of individual markers
    void estimatePoseSingleMarkers(cv::InputArrayOfArrays corners, cv::OutputArray rvecs, cv::OutputArray tvecs);

    //! Draw any markers previously detected by detectMarkers()
    /*! If txtx,txty are positive, also print a text string there */
    void drawDetections(jevois::RawImage & outimg, int txtx, int txty, std::vector<int> ids,
                        std::vector<std::vector<cv::Point2f> > corners, std::vector<cv::Vec3d> const & rvecs,
                        std::vector<cv::Vec3d> const & tvecs);
    
    //! Send serial messages about detections
    /*! The module given should be the owner of this component, we will use it to actually send each serial message
        using some variant of jevois::Module::sendSerial(). */
    void sendSerial(jevois::StdModule * mod, std::vector<int> ids, std::vector<std::vector<cv::Point2f> > corners,
                    unsigned int w, unsigned int h, std::vector<cv::Vec3d> const & rvecs,
                    std::vector<cv::Vec3d> const & tvecs);
    
    //! Our current camera matrix
    cv::Mat itsCamMatrix;

    //! Our current distortion coefficients
    cv::Mat itsDistCoeffs;

  protected:
    cv::Ptr<cv::aruco::DetectorParameters> itsDetectorParams;
    cv::Ptr<cv::aruco::Dictionary> itsDictionary;
};

