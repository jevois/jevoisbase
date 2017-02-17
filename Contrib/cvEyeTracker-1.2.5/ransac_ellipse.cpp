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


#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "ransac_ellipse.h"
#include "svd.h"
#include <string.h>

using namespace std;

//------------ Starburst pupil edge detection -----------//

// Input
// pupile_image: input image
// width, height: size of the input image
// cx,cy: central start point of the feature detection process
// pupil_edge_threshold: best guess for the pupil contour threshold 
// N: number of rays 
// minimum_candidate_features: must return this many features or error
void starburst_pupil_contour_detection(UINT8* pupil_image, int width, int height, int edge_thresh, int N,
                                       int minimum_cadidate_features, stuDPoint const & start_point,
                                       std::vector<stuDPoint> & edge_point)
{
  int dis = 7;
  double angle_spread = 100*M_PI/180;
  int loop_count = 0;
  double angle_step = 2*M_PI/N;
  double new_angle_step;
  stuDPoint *edge, edge_mean;
  double angle_normal;
  double cx = start_point.x;
  double cy = start_point.y;
  int first_ep_num;
  vector <int> edge_intensity_diff;

  while (edge_thresh > 5 && loop_count <= 10)
  {
    edge_intensity_diff.clear();
    edge_point.clear();

    while (int(edge_point.size()) < minimum_cadidate_features && edge_thresh > 5)
    {
      edge_intensity_diff.clear();
      edge_point.clear();

      locate_edge_points(pupil_image, width, height, cx, cy, dis, angle_step, 0, 2*M_PI,
                         edge_thresh, edge_intensity_diff, edge_point);

      if (int(edge_point.size()) < minimum_cadidate_features) {
        edge_thresh -= 1;
      }
    }
    if (edge_thresh <= 5) {
      break;
    }
    
    first_ep_num = edge_point.size();
    for (int i = 0; i < first_ep_num; i++) {
      edge = &edge_point.at(i);
      angle_normal = atan2(cy-edge->y, cx-edge->x);
      new_angle_step = angle_step*(edge_thresh*1.0/edge_intensity_diff.at(i));
      locate_edge_points(pupil_image, width, height, edge->x, edge->y, dis, new_angle_step, angle_normal, 
                         angle_spread, edge_thresh, edge_intensity_diff, edge_point);
    }

    loop_count += 1;
    edge_mean = get_edge_mean(edge_point);
    if (fabs(edge_mean.x-cx) + fabs(edge_mean.y-cy) < 10)
      break;

    cx = edge_mean.x;
    cy = edge_mean.y;
  }

  if (loop_count > 10) {
    edge_point.clear();
    ////printf("Error! edge points did not converge in %d iterations!\n", loop_count);
    return;
  }

  if (edge_thresh <= 5) {
    edge_point.clear();
    ////printf("Error! Adaptive threshold is too low!\n");
    return;
  }
}

void locate_edge_points(UINT8* image, int width, int height, double cx, double cy, int dis, double angle_step,
                        double angle_normal, double angle_spread, int edge_thresh,
                        std::vector<int> & edge_intensity_diff, std::vector<stuDPoint> & edge_point)
{
  double angle;
  stuDPoint p, *edge;
  double dis_cos, dis_sin;
  int pixel_value1, pixel_value2;
  for (angle = angle_normal-angle_spread/2+0.0001; angle < angle_normal+angle_spread/2; angle += angle_step)
  {
    dis_cos = dis * cos(angle);
    dis_sin = dis * sin(angle);
    p.x = cx + dis_cos;
    p.y = cy + dis_sin;

    // JEVOIS 
    if (p.x < 20) p.x = 20; else if (p.x >= width-20) p.x = width - 21;
    if (p.y < 20) p.y = 20; else if (p.y >= height-20) p.y = height - 21;
    
    pixel_value1 = image[(int)(p.y)*width+(int)(p.x)];
    while (1)
    {
      p.x += dis_cos;
      p.y += dis_sin;
      if (p.x < 0 || p.x >= width || p.y < 0 || p.y >= height) break;

      pixel_value2 = image[(int)(p.y)*width+(int)(p.x)];
      if (pixel_value2 - pixel_value1 > edge_thresh)
      {
        edge_point.push_back( { p.x - dis_cos/2, p.y - dis_sin/2 } );
        edge_intensity_diff.push_back(pixel_value2 - pixel_value1);
        break;
      }
      pixel_value1 = pixel_value2;
    }
  }
}

stuDPoint get_edge_mean(std::vector<stuDPoint> const & edge_point)
{
  double sumx = 0.0, sumy = 0.0;

  if (edge_point.empty()) return { -1.0, -1.0 };
  
  for (stuDPoint const & edge : edge_point) { sumx += edge.x; sumy += edge.y; }
  return { sumx / edge_point.size(), sumy / edge_point.size() };
}

//------------ Ransac ellipse fitting -----------//
// Randomly select 5 indeics
void get_5_random_num(int max_num, int* rand_num)
{
  int rand_index = 0;
  int r;
  int i;
  bool is_new = 1;

  if (max_num == 4) {
    for (i = 0; i < 5; i++) {
      rand_num[i] = i;
    }
    return;
  }

  while (rand_index < 5) {
    is_new = 1;
    r = (int)((rand()*1.0/RAND_MAX) * max_num);
    for (i = 0; i < rand_index; i++) {
      if (r == rand_num[i]) {
        is_new = 0;
        break;
      }
    }
    if (is_new) {
      rand_num[rand_index] = r;
      rand_index++;
    }
  }
}


// solve_ellipse
// conic_param[6] is the parameters of a conic {a, b, c, d, e, f}; conic equation: ax^2 + bxy + cy^2 + dx + ey + f = 0;
// ellipse_param[5] is the parameters of an ellipse {ellipse_a, ellipse_b, cx, cy, theta}; a & b is the major or minor axis; 
// cx & cy is the ellipse center; theta is the ellipse orientation
bool solve_ellipse(double* conic_param, double* ellipse_param)
{
  double a = conic_param[0];
  double b = conic_param[1];
  double c = conic_param[2];
  double d = conic_param[3];
  double e = conic_param[4];
  double f = conic_param[5];
  //get ellipse orientation
  double theta = atan2(b, a-c)/2;

  //get scaled major/minor axes
  double ct = cos(theta);
  double st = sin(theta);
  double ap = a*ct*ct + b*ct*st + c*st*st;
  double cp = a*st*st - b*ct*st + c*ct*ct;

  //get translations
  double cx = (2*c*d - b*e) / (b*b - 4*a*c);
  double cy = (2*a*e - b*d) / (b*b - 4*a*c);

  //get scale factor
  double val = a*cx*cx + b*cx*cy + c*cy*cy;
  double scale_inv = val - f;

  if (scale_inv/ap <= 0 || scale_inv/cp <= 0) {
    //printf("Error! ellipse parameters are imaginary a=sqrt(%lf), b=sqrt(%lf)\n", scale_inv/ap, scale_inv/cp);
    for (int k = 0; k < 5; ++k) ellipse_param[k] = 0.0;
    return 0;
  }

  ellipse_param[0] = sqrt(scale_inv / ap);
  ellipse_param[1] = sqrt(scale_inv / cp);
  ellipse_param[2] = cx;
  ellipse_param[3] = cy;
  ellipse_param[4] = theta;
  return 1;
}

stuDPoint* normalize_point_set(stuDPoint* point_set, double &dis_scale, stuDPoint &nor_center, int num)
{
  double sumx = 0, sumy = 0;
  double sumdis = 0;
  stuDPoint *edge = point_set;
  int i;
  for (i = 0; i < num; i++) {
    sumx += edge->x;
    sumy += edge->y;
    sumdis += sqrt((double)(edge->x*edge->x + edge->y*edge->y));
    edge++;
  }

  dis_scale = sqrt((double)2)*num/sumdis;
  nor_center.x = sumx*1.0/num;
  nor_center.y = sumy*1.0/num;
  stuDPoint *edge_point_nor = (stuDPoint*)malloc(sizeof(stuDPoint)*num);
  edge = point_set;
  for (i = 0; i < num; i++) {
    edge_point_nor[i].x = (edge->x - nor_center.x)*dis_scale;
    edge_point_nor[i].y = (edge->y - nor_center.y)*dis_scale;
    edge++;
  }
  return edge_point_nor;
}

stuDPoint* normalize_edge_point(double &dis_scale, stuDPoint &nor_center, int ep_num,
                                std::vector<stuDPoint> const & edge_point)
{
  double sumx = 0, sumy = 0, sumdis = 0;

  for (stuDPoint const & edge : edge_point)
  { sumx += edge.x; sumy += edge.y; sumdis += sqrt((double)(edge.x*edge.x + edge.y*edge.y)); }

  dis_scale = sqrt((double)2)*ep_num/sumdis;
  nor_center.x = sumx*1.0/ep_num;
  nor_center.y = sumy*1.0/ep_num;
  stuDPoint *edge_point_nor = (stuDPoint*)malloc(sizeof(stuDPoint)*ep_num);

  for (int i = 0; i < ep_num; ++i)
  {
    edge_point_nor[i].x = (edge_point[i].x - nor_center.x)*dis_scale;
    edge_point_nor[i].y = (edge_point[i].y - nor_center.y)*dis_scale;
  }
  return edge_point_nor;
}

void denormalize_ellipse_param(double* par, double* normailized_par, double dis_scale, stuDPoint nor_center)
{
    par[0] = normailized_par[0] / dis_scale;	//major or minor axis
    par[1] = normailized_par[1] / dis_scale;
    par[2] = normailized_par[2] / dis_scale + nor_center.x;	//ellipse center
    par[3] = normailized_par[3] / dis_scale + nor_center.y;
}

int* pupil_fitting_inliers(UINT8* JEVOIS_UNUSED_PARAM(pupil_image), int width, int height,  int &return_max_inliers_num,
                           double pupil_param[5], std::vector<stuDPoint> const & edge_point)
{
  int i;
  int ep_num = edge_point.size();   //ep stands for edge point
  stuDPoint nor_center;
  double dis_scale;

  int ellipse_point_num = 5;	//number of point that needed to fit an ellipse
  if (ep_num < ellipse_point_num) {
    ////printf("Error! %d points are not enough to fit ellipse\n", ep_num);
    return_max_inliers_num = 0;
    return NULL;
  }

  //Normalization
  stuDPoint *edge_point_nor = normalize_edge_point(dis_scale, nor_center, ep_num, edge_point);

  //Ransac
  int *inliers_index = (int*)malloc(sizeof(int)*ep_num);
  int *max_inliers_index = (int*)malloc(sizeof(int)*ep_num);
  int ninliers = 0;
  int max_inliers = 0;
  int sample_num = 1000;	//number of sample
  int ransac_count = 0;
  double dis_threshold = sqrt(3.84)*dis_scale;
  double dis_error;
  
  memset(inliers_index, int(0), sizeof(int)*ep_num);
  memset(max_inliers_index, int(0), sizeof(int)*ep_num);
  int rand_index[5];
  double A[6][6];
  int M = 6, N = 6; //M is row; N is column
  for (i = 0; i < N; i++) {
    A[i][5] = 1;
    A[5][i] = 0;
  }
  double **ppa = (double**)malloc(sizeof(double*)*M);
  double **ppu = (double**)malloc(sizeof(double*)*M);
  double **ppv = (double**)malloc(sizeof(double*)*N);
  for (i = 0; i < M; i++) {
    ppa[i] = A[i];
    ppu[i] = (double*)malloc(sizeof(double)*N);
  }
  for (i = 0; i < N; i++) {
    ppv[i] = (double*)malloc(sizeof(double)*N);
  }
  double pd[6]; 
  int min_d_index;
  double conic_par[6] = {0};
  double ellipse_par[5] = {0};
  double best_ellipse_par[5] = {0};
  double ratio;
  while (sample_num > ransac_count) {
    get_5_random_num((ep_num-1), rand_index);
	
    //svd decomposition to solve the ellipse parameter
    for (i = 0; i < 5; i++) {
      A[i][0] = edge_point_nor[rand_index[i]].x * edge_point_nor[rand_index[i]].x;
      A[i][1] = edge_point_nor[rand_index[i]].x * edge_point_nor[rand_index[i]].y;
      A[i][2] = edge_point_nor[rand_index[i]].y * edge_point_nor[rand_index[i]].y;
      A[i][3] = edge_point_nor[rand_index[i]].x;
      A[i][4] = edge_point_nor[rand_index[i]].y;
    }

    svd(M, N, ppa, ppu, pd, ppv);
    min_d_index = 0;
    for (i = 1; i < N; i++) {
      if (pd[i] < pd[min_d_index])
        min_d_index = i;
    }

    for (i = 0; i < N; i++)
      conic_par[i] = ppv[i][min_d_index];	//the column of v that corresponds to the smallest singular value, 
                                                //which is the solution of the equations
    ninliers = 0;
    memset(inliers_index, 0, sizeof(int)*ep_num);
    for (i = 0; i < ep_num; i++) {
      dis_error = conic_par[0]*edge_point_nor[i].x*edge_point_nor[i].x + 
                  conic_par[1]*edge_point_nor[i].x*edge_point_nor[i].y +
                  conic_par[2]*edge_point_nor[i].y*edge_point_nor[i].y + 
                  conic_par[3]*edge_point_nor[i].x + conic_par[4]*edge_point_nor[i].y + conic_par[5];
      if (fabs(dis_error) < dis_threshold) {
        inliers_index[ninliers] = i;
        ninliers++;
      }
    }

    if (ninliers > max_inliers) {
      if (solve_ellipse(conic_par, ellipse_par)) {
        denormalize_ellipse_param(ellipse_par, ellipse_par, dis_scale, nor_center);
        ratio = ellipse_par[0] / ellipse_par[1];
        if (ellipse_par[2] > 0 && ellipse_par[2] <= width-1 && ellipse_par[3] > 0 && ellipse_par[3] <= height-1 &&
            ratio > 0.5 && ratio < 2) {
          memcpy(max_inliers_index, inliers_index, sizeof(int)*ep_num);
          for (i = 0; i < 5; i++) {
            best_ellipse_par[i] = ellipse_par[i];
          }
          max_inliers = ninliers;
          sample_num = (int)(log((double)(1-0.99))/log(1.0-pow(ninliers*1.0/ep_num, 5)));
        }
      }
    }
    ransac_count++;
    if (ransac_count > 1500) {
      ////printf("Error! ransac_count exceed! ransac break! sample_num=%d, ransac_count=%d\n", sample_num, ransac_count);
      break;
    }
  }
  //INFO("ransc end\n");
  if (best_ellipse_par[0] > 0 && best_ellipse_par[1] > 0) {
    for (i = 0; i < 5; i++) {
      pupil_param[i] = best_ellipse_par[i];
    }
  } else {
    for (int k = 0; k < 5; ++k) pupil_param[k] = 0.0;
    max_inliers = 0;
    free(max_inliers_index);
    max_inliers_index = NULL;
  }

  for (i = 0; i < M; i++) {
    free(ppu[i]);
    free(ppv[i]);
  }
  free(ppu);
  free(ppv);
  free(ppa);

  free(edge_point_nor);
  free(inliers_index);
  return_max_inliers_num = max_inliers;
  return max_inliers_index;
}
