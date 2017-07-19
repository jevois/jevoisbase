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

#include <jevoisbase/Components/ARtoolkit/ARtoolkit.H>
#include <jevois/Image/RawImageOps.H>

// ##############################################################################################################
ARtoolkit::~ARtoolkit()
{ }

// ##############################################################################################################
void ARtoolkit::postInit()
{
  // Defer reading camera parameters to first processed frame, so we know the resolution:
  camparams::freeze();

}

// ##############################################################################################################
void ARtoolkit::postUninit()
{
  camparams::unFreeze();
}

// ##############################################################################################################
void ARtoolkit::manualinit(int w, int h, AR_PIXEL_FORMAT pixformat)
{
  std::string markerConfigDataFilename;
  
  switch (artoolkit::dictionary::get())
  {
  case artoolkit::Dict::AR_MATRIX_CODE_3x3: markerConfigDataFilename = "markers0.dat"; break;
  case artoolkit::Dict::AR_MATRIX_CODE_3x3_HAMMING63: markerConfigDataFilename = "markers1.dat"; break;
  case artoolkit::Dict::AR_MATRIX_CODE_3x3_PARITY65: markerConfigDataFilename = "markers2.dat"; break;
  default: markerConfigDataFilename = "markers2.dat";
  }

  ARParam cparam;

  arParamChangeSize(&cparam, w, h, &cparam);

  std::string const CPARA_NAME =
    absolutePath(camparams::get() + std::to_string(w) + 'x' + std::to_string(h) + ".dat");

  if (arParamLoad(absolutePath(CPARA_NAME).c_str(), 1, &cparam) < 0)
    LFATAL("Failed to load camera parameters " << CPARA_NAME);

  if ((gCparamLT = arParamLTCreate(&cparam, AR_PARAM_LT_DEFAULT_OFFSET)) == nullptr) LFATAL("Error in arParamLTCreate");

  if ((arHandle = arCreateHandle(gCparamLT)) == nullptr) LFATAL("Error in arCreateHandle");

  if ((ar3DHandle = ar3DCreateHandle(&cparam)) == nullptr) LFATAL("Error in ar3DCreateHandle");

  if (arSetPixelFormat(arHandle, pixformat) < 0) LFATAL("Error in arSetPixelFormat");

  if ((arPattHandle = arPattCreateHandle()) == nullptr) LFATAL("Error in arPattCreateHandle");

  newMarkers(absolutePath(markerConfigDataFilename).c_str(), arPattHandle, &markersSquare,
             &markersSquareCount, &gARPattDetectionMode);

  arPattAttach(arHandle, arPattHandle);

  arSetPatternDetectionMode(arHandle, AR_MATRIX_CODE_DETECTION);

  switch(dictionary::get())
  {
  case artoolkit::Dict::AR_MATRIX_CODE_3x3:
    arSetMatrixCodeType(arHandle, AR_MATRIX_CODE_3x3); break;
  case artoolkit::Dict::AR_MATRIX_CODE_3x3_HAMMING63:
    arSetMatrixCodeType(arHandle, AR_MATRIX_CODE_3x3_HAMMING63); break;
  case artoolkit::Dict::AR_MATRIX_CODE_3x3_PARITY65:
    arSetMatrixCodeType(arHandle, AR_MATRIX_CODE_3x3_PARITY65); break;
  default: arSetMatrixCodeType(arHandle, AR_MATRIX_CODE_3x3_PARITY65);
  }
}

// ##############################################################################################################
void ARtoolkit::detectMarkers(jevois::RawImage const & image)
{
  // Finalize initialization now that image format and size is known:
  if (arHandle == nullptr)
  {
    switch (image.fmt)
    {
    case V4L2_PIX_FMT_YUYV: manualinit(image.width, image.height, AR_PIXEL_FORMAT_yuvs); break; 
    case V4L2_PIX_FMT_GREY: manualinit(image.width, image.height, AR_PIXEL_FORMAT_MONO); break;
    case V4L2_PIX_FMT_RGB565: manualinit(image.width, image.height, AR_PIXEL_FORMAT_RGB_565); break;
    case V4L2_PIX_FMT_BGR24: manualinit(image.width, image.height, AR_PIXEL_FORMAT_BGR); break;
    default: LFATAL("Unsupported image format, should be V4L2_PIX_FMT_YUYV, V4L2_PIX_FMT_GREY, "
                    "V4L2_PIX_FMT_RGB565, or V4L2_PIX_FMT_BGR24");
    }
  }

  // Not sure why arDetectMarker() needs write access to the pixels? input pixels should be const...
  if (arDetectMarker(arHandle, const_cast<unsigned char *>(image.pixels<unsigned char>())) < 0) itsNumDetected = 0;
  else itsNumDetected = arGetMarkerNum(arHandle);
}

// ##############################################################################################################
void ARtoolkit::detectMarkers(cv::Mat const & image)
{
  // Finalize initialization now that image format and size is known:
  if (arHandle == nullptr)
  {
    switch (image.type())
    {
    case CV_8UC3: manualinit(image.cols, image.rows, AR_PIXEL_FORMAT_BGR); break;
    case CV_8UC1: manualinit(image.cols, image.rows, AR_PIXEL_FORMAT_MONO); break;
    default: LFATAL("Unsupported image format, should be CV_8UC3 for BGR or CV_8UC1 for gray");
    }
  }

  // Not sure why arDetectMarker() needs write access to the pixels? input pixels should be const...
  if (arDetectMarker(arHandle, const_cast<unsigned char *>(image.data)) < 0) itsNumDetected = 0;
  else itsNumDetected = arGetMarkerNum(arHandle);
}

// ##############################################################################################################
void ARtoolkit::drawDetections(jevois::RawImage & outimg, int txtx, int txty)
{
  int useContPoseEstimation = artoolkit::useContPoseEstimation::get();
  ARdouble  err;

  // Show number of detections:
  if (txtx >= 0 && txty >= 0)
    jevois::rawimage::writeText(outimg, "Detected " + std::to_string(itsNumDetected) + " ARtoolkit markers.",
                                txtx, txty, jevois::yuyv::White);

  // If we have no detection, we are done here:
  if (itsNumDetected == 0) return;

  // Draw each detection:
  ARMarkerInfo * markerInfo = arGetMarker(arHandle);

  for (int i = 0; i < markersSquareCount; ++i)
  {
    markersSquare[i].validPrev = markersSquare[i].valid;
    int k = -1;
    if (markersSquare[i].patt_type == AR_PATTERN_TYPE_MATRIX)
    {
      for (int j = 0; j < itsNumDetected; ++j)
      {
        if (markersSquare[i].patt_id == markerInfo[j].id)
        {
          if (k == -1)
          {
            if (markerInfo[j].cfPatt >= markersSquare[i].matchingThreshold) k = j; // First marker detected.
          }
          else if (markerInfo[j].cfPatt > markerInfo[k].cfPatt) k = j; // Higher confidence marker detected.
        }
      }
    } 
    if (k != -1)
    {
      markersSquare[i].valid = TRUE;
      // get the transformation matrix from the markers to camera in camera coordinate
      // rotation matrix + translation matrix
      if (markersSquare[i].validPrev && useContPoseEstimation)
        err = arGetTransMatSquareCont(ar3DHandle, &(markerInfo[k]), markersSquare[i].trans,
                                      markersSquare[i].marker_width, markersSquare[i].trans);
      else
        err = arGetTransMatSquare(ar3DHandle, &(markerInfo[k]), markersSquare[i].marker_width, markersSquare[i].trans);

      jevois::rawimage::drawCircle(outimg, markerInfo[k].pos[0], markerInfo[k].pos[1], 3, 2, jevois::yuyv::LightPink);
      
      for (int i1 = 0; i1 < 4; ++i1)
      {
        auto const & v1 = markerInfo[k].vertex[i1];
        auto const & v2 = markerInfo[k].vertex[(i1 + 1) % 4];
        jevois::rawimage::drawLine(outimg, v1[0], v1[1], v2[0], v2[1], 2, jevois::yuyv::LightPink);
      }
      jevois::rawimage::writeText(outimg, "AR=" + std::to_string(markerInfo[k].id),
                                  markerInfo[k].pos[0] + 5, markerInfo[k].pos[1] + 5, jevois::yuyv::LightPink);
    }
    else
    {
      markersSquare[i].valid = FALSE;					
    }
  }
}
