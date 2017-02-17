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

#ifndef _RANSAC_ELLIPSE_H
#define _RANSAC_ELLIPSE_H

#define UINT8 unsigned char

#include <vector>

// CvPoint is 'integer', I need 'double'
typedef struct stuDPoint
{
  double x;
  double y;
} stuDPoint;

void get_5_random_num(int max_num, int* rand_num);

bool solve_ellipse(double* conic_param, double* ellipse_param);

int* pupil_fitting_inliers(UINT8* pupil_image, int, int, int &return_max_inliers, double pupil_param[5],
                           std::vector<stuDPoint> const & edge_point);

stuDPoint* normalize_edge_point(double &dis_scale, stuDPoint &nor_center, int ep_num,
                                std::vector<stuDPoint> const & edge_point);

void denormalize_ellipse_param(double* par, double* normailized_par, double dis_scale, stuDPoint nor_center);


void starburst_pupil_contour_detection(UINT8* pupil_image, int width, int height, int edge_thresh, int N,
                                       int minimum_cadidate_features, stuDPoint const & start_point,
                                       std::vector<stuDPoint> & edge_point);

void locate_edge_points(UINT8* image, int width, int height, double cx, double cy, int dis, double angle_step,
                        double angle_normal, double angle_spread, int edge_thresh,
                        std::vector<int> & edge_intensity_diff, std::vector<stuDPoint> & edge_point);

stuDPoint get_edge_mean(std::vector<stuDPoint> const & edge_point);

stuDPoint* normalize_point_set(stuDPoint* point_set, double &dis_scale, stuDPoint &nor_center, int num);

#endif
