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
// Contact information: Shixian Wen - 3641 Watt Way, HNB-10A - Los Angeles, BA 90089-2520 - USA.
// Tel: +1 213 740 3527 - shixianw@usc.edu - http://iLab.usc.edu - http://jevois.org
// ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/*! \file */

#pragma once

#include <jevois/Component/Component.H>
#include <jevois/Types/Enum.H>
#include <jevois/GPU/GUIhelper.H>

#include <AR/ar.h>
#include <AR/video.h>
#include <AR/config.h>
#include <AR/param.h>
#include <jevoisbase/Components/ARtoolkit/ARMarkerSquare.h>

// I need to define diferent camera_para.dat according to different camera size because of the limitation of the
// original ARtoolkit library or change the original ARtoolkit library and I think its not a good way to go, makes the
// package hard to distribute and update
namespace jevois { class StdModule; }

namespace artoolkit
{
  static jevois::ParameterCategory const ParamCateg("ARtoolkit Options");

  //! Parameter \relates ARtoolkit
  JEVOIS_DECLARE_PARAMETER(camparams, std::string, "File stem of camera parameters, or empty. Camera resolution "
                           "will be appended, as well as a .dat extension. For example, specifying 'camera_para' "
                           "here and running the camera sensor at 320x240 will attempt to load "
                           "camera_para320x240.dat from within the module's directory (if relative stem) or "
                           "from the specified absolute location (if absolute stem).",
                           JEVOIS_SHARE_PATH "/camera/camera_para", ParamCateg);

  //! Parameter \relates ARtoolkit
  JEVOIS_DECLARE_PARAMETER(contpose, bool, "Use continuous pose estimation in good quality video",
                           true, ParamCateg);

  //! Parameter \relates ARtoolkit
  JEVOIS_DECLARE_PARAMETER(msg3d, bool, "Send 3D serial messages if true, otherwise 2D",
                           false, ParamCateg);

  //! Enum for parameter \relates ARtoolkit
  /*! dictionary to use according to the ARtoolkit 
      AR_MATRIX_CODE_3x3 support 64 markers
      AR_MATRIX_CODE_3x3_HAMMING63 support 7 markers
      AR_MATRIX_CODE_3x3_PARITY65 support 32 markers
      the smaller the # of marker => larger distance between markers => better performance */
  JEVOIS_DEFINE_ENUM_CLASS(Dict, (AR_MATRIX_CODE_3x3) (AR_MATRIX_CODE_3x3_HAMMING63) (AR_MATRIX_CODE_3x3_PARITY65));

  //! ThreshMode to overcome different lighting situation  \relates ARtoolkit
  JEVOIS_DEFINE_ENUM_CLASS(DictThreshMode, (AR_LABELING_THRESH_MODE_MANUAL) (AR_LABELING_THRESH_MODE_AUTO_MEDIAN)
                           (AR_LABELING_THRESH_MODE_AUTO_OTSU) (AR_LABELING_THRESH_MODE_AUTO_ADAPTIVE)
                           (AR_LABELING_THRESH_MODE_AUTO_BRACKETING));

  //! Parameter \relates ARtoolkit
  JEVOIS_DECLARE_PARAMETER(threshmode, DictThreshMode, "Dictionary threshold mode to use, affects "
                           "robustness to varying lighting conditions",
                           DictThreshMode::AR_LABELING_THRESH_MODE_AUTO_OTSU, DictThreshMode_Values, ParamCateg);

  //! Parameter \relates ARtoolkit
  JEVOIS_DECLARE_PARAMETER(dictionary, Dict, "Symbol dictionary to use",
                           Dict::AR_MATRIX_CODE_3x3_PARITY65, Dict_Values, ParamCateg);

  //! Parameter \relates ARtoolkit
  JEVOIS_DECLARE_PARAMETER(confthresh, double, "Minimum confidence threshold required for valid detections",
                           0.7, jevois::Range<double>(0.0, 1.0), ParamCateg)
}

//! Augmented reality markers using ARtoolkit
/*! Detect and decode patterns known as ARtoolkit markers, which are small 2D barcodes often used in augmented
    reality and robotics.

    \ingroup Components */
class ARtoolkit : public jevois::Component,
				  public jevois::Parameter<artoolkit::camparams, artoolkit::contpose, artoolkit::dictionary,
                                           artoolkit::msg3d, artoolkit::confthresh, artoolkit::threshmode>
{
  public:
	//! Constructor
	using jevois::Component::Component;
    
    //! Core initialization
    /*! Note that initialization is only partial and will complete once the frame size and format is known at the first
        call to detectMarkers() */
    void postInit() override;
    
    // Un-init in preparation for destruction
    void postUninit() override;
    
	//! Destructor
	~ARtoolkit();

    //! Detect markers in an image
    /*! Probably the fastest if you have not already converted your input image to cv::Mat */
    void detectMarkers(jevois::RawImage const & image);

    //! Detect markers in an image
    void detectMarkers(cv::Mat const & image);

    //! Draw any markers previously detected by detectMarkers()
    /*! If txtx,txty are positive, also print a text string there */
    void drawDetections(jevois::RawImage & outimg, int txtx = -1, int txty = -1);

#ifdef JEVOIS_PRO
    //! Draw any markers previously detected by detectMarkers() using ImGui
    /*! Detections are drawn to scale on the last image drawn by the helper. */
    void drawDetections(jevois::GUIhelper & helper);
#endif
    
    //! Send serial messages about detections
    /*! The module given should be the owner of this component, we will use it to actually send each serial message
        using some variant of jevois::Module::sendSetial(). */
    void sendSerial(jevois::StdModule * mod);
    
    struct arresults
    {
        int id; //!< Marker ID
        double width; //!< size in mm
        double height; //!< size in mm
        double q[4]; //!< quaternion
        double pos[3]; //!< 3d position
        int p2d[2]; //!< 2d position in image coords
        std::vector<cv::Point> corners; //!< corners in standardized image coords
    };

    std::vector<arresults> itsResults; //!< Results of the last detection round
    
  protected:
    //! Manual initialization which should be run on first frame once its size is known
    /*! pixformat should be AR_PIXEL_FORMAT_BGR, AR_PIXEL_FORMAT_RGB_565, etc */
    void manualinit(int w, int h, AR_PIXEL_FORMAT pixformat);
    
    void detectInternal(unsigned char const * data);
    
	// some other not important parameters related to the initialization of the ARtoolkit 
    
	// initialization required
	ARHandle * arHandle = nullptr;
	ARPattHandle  *arPattHandle;
	int gARPattDetectionMode;
	AR3DHandle * ar3DHandle;
	ARParamLT * gCparamLT = nullptr;
    ARMarkerSquare * markersSquare = nullptr;
    int markersSquareCount = 0;
    unsigned int itsW, itsH;
};
