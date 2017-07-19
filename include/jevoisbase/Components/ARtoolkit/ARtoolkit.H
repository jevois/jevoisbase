#pragma once
#include <jevois/Component/Component.H>
#include <jevois/Types/Enum.H>
#include "opencv2/objdetect.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
//#include <iostream>

using namespace std;
using namespace cv;

//*******************************************
//ARtoolkit area
#include <AR/ar.h>
#include <AR/video.h>
#include <AR/config.h>
#include <AR/param.h>
#include "ARMarkerSquare.h"
// I need to define diferent camera_para.dat according to different camera size because of the limitation of the original ARtoolkit library
// or change the original ARtoolkit library and I think its not a good way to go, makes the package hard to distribute and update
//#define             CPARA_NAME       "camera_para.dat"
//*******************************************

namespace artoolkit{
	static jevois::ParameterCategory const ParamCateg("ARtoolkit Options");
	//! Parameter \relates ARtoolkit
	JEVOIS_DECLARE_PARAMETER(useContPoseEstimation, bool, "Use Continues pose Estimation in a good quality of video", true, ParamCateg);
	//! Parameter \relates ARtoolkit
	JEVOIS_DECLARE_PARAMETER(xsize, int, "the length of x axis of the video,", 640, ParamCateg);
	//! Parameter \relates ARtoolkit
	JEVOIS_DECLARE_PARAMETER(ysize, int, "the length of y axis of the video,", 480, ParamCateg);

	//! Enum for parameter \relates ArUco
	//dictionary to use according to the ARtoolkit 
	//AR_MATRIX_CODE_3x3 support 64 markers
	//AR_MATRIX_CODE_3x3_HAMMING63 support 7 markers
	//AR_MATRIX_CODE_3x3_PARITY65 support 32 markers
	//the smaller the # of marker => larger distance between markers => better performance
  	JEVOIS_DEFINE_ENUM_CLASS(Dict, (AR_MATRIX_CODE_3x3) (AR_MATRIX_CODE_3x3_HAMMING63) (AR_MATRIX_CODE_3x3_PARITY65));
  	JEVOIS_DECLARE_PARAMETER(dictionary, Dict, "Symbol dictionary to use", Dict::AR_MATRIX_CODE_3x3_PARITY65, ParamCateg);

}



class ARtoolkit : public jevois::Component,
				  public jevois::Parameter<artoolkit::useContPoseEstimation, artoolkit::xsize, artoolkit::ysize, artoolkit::dictionary>
{
	public:

	//!Constructor
	using jevois::Component::Component;
	//!Destructor
	~ARtoolkit();

	//! Initialize, create the detector and read the config files
    void postInit() override;
    
    //! Un-initialize, nuke allocated resources
    void postUninit() override;
	
    void manualinit();
	void detectMarkers(cv::Mat image);

	//some other not important parameters related to the initialization of the ARtoolkit 

	//initilization required
	ARHandle           *arHandle;
	ARPattHandle       *arPattHandle;
	int         gARPattDetectionMode;
	// Transformation matrix retrieval.
	AR3DHandle         *ar3DHandle;
	ARParamLT          *gCparamLT = NULL;
	// Markers.
    ARMarkerSquare *markersSquare = nullptr;
    int markersSquareCount = 0;

};
