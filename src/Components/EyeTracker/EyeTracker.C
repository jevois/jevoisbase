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

#include <jevoisbase/src/Components/EyeTracker/EyeTracker.H>

// This code adpated from (see Contrib directory for full source):

/*
 *
 * cvEyeTracker is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * cvEyeTracker is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with cvEyeTracker; if not, write to the Free Software
 * Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
 *
 *
 * cvEyeTracker - Version 1.2.5
 * Part of the openEyes ToolKit -- http://hcvl.hci.iastate.edu/openEyes
 * Release Date:
 * Authors : Dongheng Li <dhli@iastate.edu>
 *           Derrick Parkhurst <derrick.parkhurst@hcvl.hci.iastate.edu>
 *           Jason Babcock <babcock@nyu.edu>
 *           David Winfield <dwinfiel@iastate.edu>
 * Copyright (c) 2004-2006
 * All Rights Reserved.
 *
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <jevoisbase/Contrib/cvEyeTracker-1.2.5/remove_corneal_reflection.h>
#include <jevoisbase/Contrib/cvEyeTracker-1.2.5/svd.h>

#include <opencv/cv.h>

#define UINT8 unsigned char

#define FIX_UINT8(x) ( (x)<0 ? 0 : ((x)>255 ? 255:(x)) )

void Draw_Cross(IplImage *image, int centerx, int centery, int x_cross_length, int y_cross_length, double color)
{
  CvPoint pt1,pt2,pt3,pt4;

  pt1.x = centerx - x_cross_length;
  pt1.y = centery;
  pt2.x = centerx + x_cross_length;
  pt2.y = centery;

  pt3.x = centerx;
  pt3.y = centery - y_cross_length;
  pt4.x = centerx;
  pt4.y = centery + y_cross_length;

  cvLine(image,pt1,pt2,color,1,8);
  cvLine(image,pt3,pt4,color,1,8);
}

void Normalize_Line_Histogram(IplImage *in_image) 
{
 unsigned char *s=(unsigned char *)in_image->imageData;
 int x,y;
 int linesum;
 double factor=0;
 int subsample=10;
 double hwidth = (100.0f * ((double)in_image->width) / (double)subsample);

 for (y=0;y<in_image->height;y++) {
   linesum=1; 
   for (x=0;x<in_image->width;x+=subsample) {
     linesum+=*s;
     s+=subsample;
   }
   s-=in_image->width;
   factor=hwidth/((double)linesum);
   for (x=0;x<in_image->width;x++) {
     *s=(unsigned char)(((double)*s)*factor);
     s++;
   }
 }
}


void Calculate_Avg_Intensity_Hori(IplImage* in_image, double * avg_intensity_hori)
{
  UINT8 *pixel = (UINT8*)in_image->imageData;
  int sum;
  for (int j = 0; j < in_image->height; ++j)
  {
    sum = 0; for (int i = 0; i < in_image->width; ++i) sum += *pixel++;
    avg_intensity_hori[j] = double(sum) / in_image->width;
  }
}

void Reduce_Line_Noise(IplImage* in_image, double * avg_intensity_hori, double * intensity_factor_hori)
{
  const double beta = 0.2;	// hysteresis factor for noise reduction

  UINT8 *pixel = (UINT8*)in_image->imageData;
  double beta2 = 1 - beta;
  int adjustment;

  Calculate_Avg_Intensity_Hori(in_image, avg_intensity_hori);

  for (int j = 0; j < in_image->height; ++j)
  {
    intensity_factor_hori[j] = avg_intensity_hori[j] * beta + intensity_factor_hori[j] * beta2;

    adjustment = (int)(intensity_factor_hori[j] - avg_intensity_hori[j]);

    for (int i = 0; i < in_image->width; ++i) { *pixel = FIX_UINT8(*pixel + adjustment); ++pixel; }
  }
}

// ##############################################################################################################
EyeTracker::EyeTracker(std::string const & instance) :
    jevois::Component(instance), currWidth(0), currHeight(0), intensity_factor_hori(nullptr),
    avg_intensity_hori(nullptr)
{ }

// ##############################################################################################################
EyeTracker::~EyeTracker()
{
  if (intensity_factor_hori) free(intensity_factor_hori);
  if (avg_intensity_hori) free(avg_intensity_hori);
}
  
// ##############################################################################################################
void EyeTracker::process(cv::Mat & eyeimg, double pupell[5], bool debugdraw)
{
  int *inliers_index;
  CvSize ellipse_axis;
  CvPoint gaze_point;
  static int lost_frame_num = 0;

  // Initialize avg intensity if needed:
  if (currWidth != eyeimg.cols || currHeight != eyeimg.rows)
  {
    // New image size, reset a bunch of things:
    currWidth = eyeimg.cols; currHeight = eyeimg.rows;
    if (intensity_factor_hori) { free(intensity_factor_hori); intensity_factor_hori = nullptr; }
    if (avg_intensity_hori) { free(avg_intensity_hori); avg_intensity_hori = nullptr; }
    start_point.x = currWidth / 2; start_point.y = currHeight / 2;
    for (int k = 0; k < 5; ++k) pupil_param[k] = 0.0;
  }

  // Make the eye image (in monochrome):
  IplImage * eye_image = cvCreateImageHeader(cvSize(eyeimg.cols, eyeimg.rows), 8, 1);
  eye_image->imageData = reinterpret_cast<char *>(eyeimg.data);

  // Make the threshold image (in monochrome):
  IplImage * threshold_image = cvCloneImage(eye_image);

  if (avg_intensity_hori == nullptr)
  {
    avg_intensity_hori = (double *)malloc(currHeight * sizeof(double));
    Calculate_Avg_Intensity_Hori(eye_image, avg_intensity_hori);
  }

  if (intensity_factor_hori == nullptr)
  {
    intensity_factor_hori = (double *)malloc(currHeight * sizeof(double));
    memcpy(intensity_factor_hori, avg_intensity_hori, currHeight * sizeof(double));    
  }
  
  cvSmooth(eye_image, eye_image, CV_GAUSSIAN, 3, 3); // JEVOIS: was 5x5
  
  Reduce_Line_Noise(eye_image, avg_intensity_hori, intensity_factor_hori);  
  
  // corneal reflection
  int crwin = eyetracker::corneal::get() & 1; // enforce odd
  if (crwin >= eyeimg.cols) crwin = (eyeimg.cols - 1) & 1;
  if (crwin >= eyeimg.rows) crwin = (eyeimg.rows - 1) & 1;
  
  int corneal_reflection_r = 0;       //the radius of corneal reflection
  CvPoint corneal_reflection = { 0, 0 }; //coordinates of corneal reflection in tracker coordinate system

  remove_corneal_reflection(eye_image, threshold_image, (int)start_point.x, (int)start_point.y,
                            crwin, (int)eye_image->height/10, corneal_reflection.x,
                            corneal_reflection.y, corneal_reflection_r);  
  /////printf("corneal reflection: (%d, %d)\n", corneal_reflection.x, corneal_reflection.y);

  //starburst pupil contour detection
  starburst_pupil_contour_detection((UINT8*)eye_image->imageData, eye_image->width, eye_image->height,
                                    eyetracker::edgethresh::get(), eyetracker::numrays::get(),
                                    eyetracker::mincand::get(), start_point, edge_point);
  
  int inliers_num = 0;
  CvPoint pupil = { 0, 0 }; // coordinates of pupil in tracker coordinate system

  inliers_index = pupil_fitting_inliers((UINT8*)eye_image->imageData, eye_image->width, eye_image->height,
                                        inliers_num, pupil_param, edge_point);

  ellipse_axis.width = (int)pupil_param[0];
  ellipse_axis.height = (int)pupil_param[1];
  pupil.x = (int)pupil_param[2];
  pupil.y = (int)pupil_param[3];

  // JEVOIS only draw line if valid
  if (debugdraw && corneal_reflection.x > 3 && corneal_reflection.y > 3 && pupil.x > 3 && pupil.y > 3)
    cvLine(eye_image, pupil, corneal_reflection, 100, 4, 8);

  /*** JEVOIS comment out
  printf("ellipse a:%lf; b:%lf, cx:%lf, cy:%lf, theta:%lf; inliers_num:%d\n\n", 
         pupil_param[0], pupil_param[1], pupil_param[2], pupil_param[3], pupil_param[4], inliers_num);
  */
  for (int k = 0; k < 5; ++k) pupell[k] = pupil_param[k]; // send data to caller

  if (debugdraw)
  {
    bool is_inliers = 0;
    for (int i = 0; i < int(edge_point.size()); i++)
    {
      is_inliers = 0;
      for (int j = 0; j < inliers_num; j++) if (i == inliers_index[j]) is_inliers = 1;
      stuDPoint const & edge = edge_point.at(i);
      if (is_inliers) Draw_Cross(eye_image, (int)edge.x,(int)edge.y, 5, 5, 200);
      else Draw_Cross(eye_image, (int)edge.x, (int)edge.y, 3, 3, 100);
    }
  }
  free(inliers_index);

  if (ellipse_axis.width > 0 && ellipse_axis.height > 0)
  {
    start_point.x = pupil.x;
    start_point.y = pupil.y;
    //printf("start_point: %d,%d\n", start_point.x, start_point.y);
    Draw_Cross(eye_image, pupil.x, pupil.y, 10, 10, 255);
    cvEllipse(eye_image, pupil, ellipse_axis, -pupil_param[4]*180/M_PI, 0, 360, 255, 2);

    lost_frame_num = 0;    
  } else {
    lost_frame_num++;
  }
  if (lost_frame_num > 5) {
    start_point.x = eyeimg.cols / 2;
    start_point.y = eyeimg.rows / 2;
  }

  Draw_Cross(eye_image, (int)start_point.x, (int)start_point.y, 7, 7, 128);

  // Cleanup:
  cvReleaseImageHeader(&eye_image);
  cvReleaseImage(&threshold_image);
}

