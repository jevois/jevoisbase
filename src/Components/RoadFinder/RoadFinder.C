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

// This code adapted from:

// //////////////////////////////////////////////////////////////////// //
// The iLab Neuromorphic Vision C++ Toolkit - Copyright (C) 2001 by the //
// University of Southern California (USC) and the iLab at USC.         //
// See http://iLab.usc.edu for information about this project.          //
// //////////////////////////////////////////////////////////////////// //
// Major portions of the iLab Neuromorphic Vision Toolkit are protected //
// under the U.S. patent ``Computation of Intrinsic Perceptual Saliency //
// in Visual Environments, and Applications'' by Christof Koch and      //
// Laurent Itti, California Institute of Technology, 2001 (patent       //
// pending; application number 09/912,225 filed July 23, 2001; see      //
// http://pair.uspto.gov/cgi-bin/final/home.pl for current status).     //
// //////////////////////////////////////////////////////////////////// //
// This file is part of the iLab Neuromorphic Vision C++ Toolkit.       //
//                                                                      //
// The iLab Neuromorphic Vision C++ Toolkit is free software; you can   //
// redistribute it and/or modify it under the terms of the GNU General  //
// Public License as published by the Free Software Foundation; either  //
// version 2 of the License, or (at your option) any later version.     //
//                                                                      //
// The iLab Neuromorphic Vision C++ Toolkit is distributed in the hope  //
// that it will be useful, but WITHOUT ANY WARRANTY; without even the   //
// implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR      //
// PURPOSE.  See the GNU General Public License for more details.       //
//                                                                      //
// You should have received a copy of the GNU General Public License    //
// along with the iLab Neuromorphic Vision C++ Toolkit; if not, write   //
// to the Free Software Foundation, Inc., 59 Temple Place, Suite 330,   //
// Boston, MA 02111-1307 USA.                                           //
// //////////////////////////////////////////////////////////////////// //

#include <jevoisbase/src/Components/RoadFinder/RoadFinder.H>
#include <jevois/Debug/Log.H>
#include <jevois/Debug/Profiler.H>
#include <opencv2/imgproc/imgproc.hpp> // for Canny
#include <opencv2/imgproc/imgproc_c.h> // for cvFitLine
#include <jevois/Image/RawImageOps.H>
#include <future>

// heading difference per unit pixel, it's measured 27 degrees per half image of 160 pixels
#define HEADING_DIFFERENCE_PER_PIXEL  27.0/160.0*M_PI/180.0  // in radians 

// Beobot 2.0 pixel to cm road bottom conversion ratio, that is, 1 pic = 9.525mm
#define BEOBOT2_PIXEL_TO_MM_ROAD_BOTTOM_RATIO 9.525

namespace
{
  // ######################################################################
  static inline bool coordsOk(int const x, int const y, int const w, int const h)
  { return (x >= 0 && x < w && y >= 0 && y < h); }

  // ######################################################################
  Point2D<float> intersectPoint(Point2D<float> const & p1, Point2D<float> const & p2,
                                Point2D<float> const & p3, Point2D<float> const & p4)
  {
    //Find intersection point Algorithm can be find here :
    //http://paulbourke.net/geometry/lineline2d/

    double mua,mub;
    double denom,numera,numerb;
    double x,y;
    double EPS = 0.0001;//Epsilon : a small number to enough to be insignificant
    
    denom  = (p4.j-p3.j) * (p2.i-p1.i) - (p4.i-p3.i) * (p2.j-p1.j);
    numera = (p4.i-p3.i) * (p1.j-p3.j) - (p4.j-p3.j) * (p1.i-p3.i);
    numerb = (p2.i-p1.i) * (p1.j-p3.j) - (p2.j-p1.j) * (p1.i-p3.i);
    
    /* Are the lines coincident? */
    if (fabs(numera) < EPS && fabs(numerb) < EPS && fabs(denom) < EPS) 
    {
      x = (p1.i + p2.i) / 2;
      y = (p1.j + p2.j) / 2;
      return Point2D<float>(x,y);
    }
    
    /* Are the lines parallel */
    if (fabs(denom) < EPS) {
      x = 0;
      y = 0;
      return Point2D<float>(x,y);
    }
    
    /* Is the intersection along the the segments */
    mua = numera / denom;
    mub = numerb / denom;
    if (mua < 0 || mua > 1 || mub < 0 || mub > 1) {
      x = 0;
      y = 0;
      
    }
    x = p1.i + mua * (p2.i - p1.i);
    y = p1.j + mua * (p2.j - p1.j);
    
    return Point2D<float>(x,y);
  }

  // ######################################################################
  template<typename T> 
  T side(Point2D<T> const & pt1, Point2D<T> const & pt2, Point2D<T> const & pt)
  {
    T i = pt.i;
    T j = pt.j;
    
    T Ax = pt1.i;  T Ay = pt1.j;
    T Bx = pt2.i;  T By = pt2.j;
    
    // 0 means on the line
    // positive value means to the left 
    // negative value means to the right
    return (Bx-Ax)*(j-Ay) - (By-Ay)*(i-Ax);
  }

  // ######################################################################
  template<typename T> 
  double distance(Point2D<T> const & pt1, Point2D<T> const & pt2, Point2D<T> const & pt, Point2D<T> & midPt)
  {
    double x1 = pt1.i;
    double y1 = pt1.j;
    
    double x2 = pt2.i;
    double y2 = pt2.j;
    
    double x3 = pt.i;
    double y3 = pt.j;
    
    double xi;  double yi;
    
    if     (x1 == x2){ xi = x1; yi = y3; }
    else if(y1 == y2){ xi = x3; yi = y1; }
    else
    {
      double x2m1 = x2 - x1;
      double y2m1 = y2 - y1;
      
      double x3m1 = x3 - x1;
      double y3m1 = y3 - y1;
      
      double distSq = x2m1*x2m1+y2m1*y2m1;
      double u = (x3m1*x2m1 + y3m1*y2m1)/distSq;
      
      xi = x1 + u*x2m1;
      yi = y1 + u*y2m1;
    }
    
    midPt = Point2D<T>(xi,yi);
    
    double dx = xi - x3;
    double dy = yi - y3;
    return pow(dx*dx+dy*dy, .5F);
  }

  // ######################################################################
  template<typename T> 
  double distance(Point2D<T> const & pt1, Point2D<T> const & pt2, Point2D<T> const & pt)
  {
    Point2D<T> midPt;
    return distance(pt1, pt2, pt, midPt);
  }
  
} // namespace

// ######################################################################
RoadFinder::RoadFinder(std::string const & instance) :
    jevois::Component(instance), itsTPXfilter(2, 1, 0), itsKalmanNeedInit(true)
{
  itsVanishingPoint           = Point2D<int>  (-1,-1);
  itsCenterPoint              = Point2D<float>(-1,-1);
  itsTargetPoint              = Point2D<float>(-1,-1);
  itsVanishingPointConfidence = 0.1;
  
  // currently not processing tracker
  itsTrackingFlag = false;
  
  // current accumulated trajectory
  itsAccumulatedTrajectory.i = 0.0F;
  itsAccumulatedTrajectory.j = 0.0F;
  
  // indicate how many unique lines have been identified NOTE: never reset
  itsNumIdentifiedLines = 0;

  // init kalman filter
  itsTPXfilter.transitionMatrix = (cv::Mat_<float>(2, 2) << 1, 1, 0, 1);
  cv::setIdentity(itsTPXfilter.measurementMatrix);
  cv::setIdentity(itsTPXfilter.processNoiseCov, cv::Scalar::all(0.1F * 0.1F));
  cv::setIdentity(itsTPXfilter.measurementNoiseCov, cv::Scalar::all(2.0F * 2.0F));
  cv::setIdentity(itsTPXfilter.errorCovPost, cv::Scalar::all(0.1F * 0.1F));
  itsTPXfilter.statePost.at<float>(1) = 0.0F;
}

// ######################################################################
RoadFinder::~RoadFinder()
{ }

// ######################################################################
void RoadFinder::postInit()
{
  roadfinder::horizon::freeze();
  roadfinder::support::freeze();
  roadfinder::spacing::freeze();
}

// ######################################################################
void RoadFinder::preUninit()
{
  roadfinder::horizon::unFreeze();
  roadfinder::support::unFreeze();
  roadfinder::spacing::unFreeze();
}

// ######################################################################
std::pair<Point2D<int>, float> RoadFinder::getCurrVanishingPoint() const
{ return std::make_pair(itsVanishingPoint, itsVanishingPointConfidence); }

// ######################################################################
Point2D<float> RoadFinder::getCurrCenterPoint() const
{ return itsCenterPoint; }

// ######################################################################
Point2D<float> RoadFinder::getCurrTargetPoint() const
{ return itsTargetPoint; }

// ######################################################################
float RoadFinder::getFilteredTargetX() const
{ return itsFilteredTPX; }

// ######################################################################
void RoadFinder::resetRoadModel()
{
  std::lock_guard<std::mutex> _(itsRoadMtx);
  itsRoadModel.lines.clear();
  itsRoadModel.lastSeenHorizonPoint.clear();
  itsRoadModel.lastSeenLocation.clear();
  itsRoadModel.lastActiveIndex.clear();
  itsRoadModel.numMatches.clear();      
}

// ######################################################################
void RoadFinder::process(cv::Mat const & img, jevois::RawImage & visual)
{
  static jevois::Profiler profiler("RoadFinder", 100, LOG_DEBUG);
  static int currRequestID = 0;
  ++currRequestID; ///FIXME
  
  profiler.start();

  // Set initial kalman state now that we know image width:
  if (itsKalmanNeedInit)
  {
    itsTPXfilter.statePost.at<float>(0) = img.cols / 2;
    itsKalmanNeedInit = false;
  }
  
  // If we are just starting, initialize the vanishing point locations:
  if (itsVanishingPoints.empty())
  {
    int const spacing = roadfinder::spacing::get();
    int const hline   = roadfinder::horizon::get();
    for (int i = -4 * spacing; i <= img.cols + 4 * spacing; i += spacing)
      itsVanishingPoints.push_back(VanishingPoint(Point2D<int>(i, hline), 0.0F));
  }

  // Code from updateMessage() in original code:

  // Compute Canny edges:
  int const sobelApertureSize = 7;
  int const highThreshold = 400 * sobelApertureSize * sobelApertureSize;
  int const lowThreshold  = int(highThreshold * 0.4F);
  cv::Mat cvEdgeMap;
  cv::Canny(img, cvEdgeMap, lowThreshold, highThreshold, sobelApertureSize);

  profiler.checkpoint("Canny done");
  
  // Get prior vp while we compute the new one:
  Point2D<int> prior_vp = itsVanishingPoint;

  // Track lines in a thread if we have any. Do not access itsCurrentLines or destroy cvEdgeMap until done:
  std::future<void> track_fut;
  if (itsCurrentLines.empty() == false)
    track_fut = std::async(std::launch::async, [&]() {
        // Track the vanishing lines:
        trackVanishingLines(cvEdgeMap, itsCurrentLines, visual);
        
        // Compute the vanishing point, center point, target point:
        float confidence; Point2D<int> vp(-1, -1); Point2D<float> cp(-1, -1);
        Point2D<float> tp = computeRoadCenterPoint(cvEdgeMap, itsCurrentLines, vp, cp, confidence);
        
        // update road model:
        updateRoadModel(itsCurrentLines, currRequestID);
        
        itsVanishingPoint           = vp;
        itsCenterPoint              = cp;
        itsTargetPoint              = tp;
        itsVanishingPointConfidence = confidence;
      });
  else
  {
    itsVanishingPoint           = Point2D<int>  (-1,-1);
    itsCenterPoint              = Point2D<float>(-1,-1);
    itsTargetPoint              = Point2D<float>(-1,-1);
    itsVanishingPointConfidence = 0.1;
  }

  profiler.checkpoint("Tracker launched");
  
  // Code from evolve() in original code:
  computeHoughSegments(cvEdgeMap); 

  profiler.checkpoint("Hough done");
  
  std::vector<Line> newLines = computeVanishingLines(cvEdgeMap, prior_vp, visual);

  profiler.checkpoint("Vanishing lines done");
  
  // wait until the tracking thread is done, get the trackers and 'disable' it during project forward
  if (track_fut.valid()) track_fut.get();

  profiler.checkpoint("Tracker done");

  // integrate the two sets of lines
  itsCurrentLines = combine(newLines, itsCurrentLines, cvEdgeMap.cols, cvEdgeMap.rows);

  profiler.checkpoint("Combine done");
  
  // Do some demo visualization if desired:
  if (visual.valid())
  {
    // find the most likely vanishing point location
    size_t max_il  = 0; float max_l  = itsVanishingPoints[max_il].likelihood;
    size_t max_ip = 0; float max_p  = itsVanishingPoints[max_ip].posterior;
    for (size_t i = 0; i < itsVanishingPoints.size(); ++i)
    {
      float const likelihood = itsVanishingPoints[i].likelihood;
      float const posterior = itsVanishingPoints[i].posterior;
      if (max_l < likelihood) { max_l  = likelihood; max_il = i; }
      if (max_p < posterior)  { max_p  = posterior;  max_ip = i; }
    }

    // draw the vanishing point likelihoods
    for (size_t i = 0; i < itsVanishingPoints.size(); ++i)
    {
      float const likelihood = itsVanishingPoints[i].likelihood;
      float const posterior  = itsVanishingPoints[i].posterior;
      Point2D<int> const & vp  = itsVanishingPoints[i].vp;
      int l_size = likelihood / max_l * 10;
      int p_size = posterior / max_l * 10;
    
      if (l_size < 2) l_size = 2;
      if (p_size < 2) p_size = 2;
    
      Point2D<int> pt = vp;
      if (i == max_il) jevois::rawimage::drawDisk(visual, pt.i, pt.j, l_size, jevois::yuyv::LightPink); // orange
      else jevois::rawimage::drawDisk(visual, pt.i, pt.j, l_size, jevois::yuyv::DarkPink);
    
      if (i == max_ip) jevois::rawimage::drawDisk(visual, pt.i, pt.j, p_size, jevois::yuyv::LightGreen);
      else jevois::rawimage::drawDisk(visual, pt.i, pt.j, p_size, jevois::yuyv::DarkGreen);
    }
  
    // Draw all the segments found:
    for (Segment const & s : itsCurrentSegments)
      jevois::rawimage::drawLine(visual, s.p1.i, s.p1.j, s.p2.i, s.p2.j, 0, jevois::yuyv::DarkGrey);
  
    // Draw the supporting segments
    for (Segment const & s : itsVanishingPoints[max_ip].supportingSegments)
      jevois::rawimage::drawLine(visual, s.p1.i, s.p1.j, s.p2.i, s.p2.j, 1, jevois::yuyv::LightGrey);

    // Draw current tracked lines
    for (Line const & line : itsCurrentLines)
    {
      unsigned short color = jevois::yuyv::DarkGreen;
      if (line.start_scores.size() > 0) color = jevois::yuyv::LightGreen;
      else if (line.scores.size() > 0) color = jevois::yuyv::MedGreen;

      jevois::rawimage::drawLine(visual, line.horizonPoint.i, line.horizonPoint.j,
                                 line.roadBottomPoint.i, line.roadBottomPoint.j, 0, color);
      jevois::rawimage::drawLine(visual, line.onScreenHorizonSupportPoint.i, line.onScreenHorizonSupportPoint.j,
                                 line.onScreenRoadBottomPoint.i, line.onScreenRoadBottomPoint.j, 0, color);
      
      // Highlight the segment points:
      for (Point2D<int> const & p : line.points) jevois::rawimage::drawDisk(visual, p.i, p.j, 1, jevois::yuyv::White);
    }
    int slack = 0;
    
    // Lateral position information
    //Point2D<int>   vp = itsVanishingPoint;
    Point2D<float> cp = itsCenterPoint;
    Point2D<float> tp = itsTargetPoint;
  
    // draw the lateral position point
    Point2D<int> cp_i(cp.i+slack, cp.j); 
    Point2D<int> tp_i(tp.i+slack, tp.j);   
    Point2D<int> cp_i0(cp.i+slack, cvEdgeMap.rows-20);
    Point2D<int> tp_i0(tp.i+slack, cvEdgeMap.rows-20); 
    if (cp_i.isValid())
      jevois::rawimage::drawLine(visual, cp_i0.i, cp_i0.j, cp_i.i, cp_i.j, 2, jevois::yuyv::LightGreen);
    if (tp_i.isValid())
      jevois::rawimage::drawLine(visual, tp_i0.i, tp_i0.j, tp_i.i, tp_i.j, 2, jevois::yuyv::LightPink);
  }
  
  // Filter the target point: Predict:
  cv::Mat prediction = itsTPXfilter.predict();
  
  // Update:
  if (itsTargetPoint.isValid())
  {
      cv::Mat measurement = (cv::Mat_<float>(1, 1) << itsTargetPoint.i);
      cv::Mat estimated = itsTPXfilter.correct(measurement);
      itsFilteredTPX = estimated.at<float>(0);
  }
  else itsFilteredTPX = prediction.at<float>(0);
    
  profiler.stop();
}

//######################################################################
Point2D<float> RoadFinder::computeRoadCenterPoint(cv::Mat const & edgeMap, std::vector<Line> & lines,
                                                  Point2D<int> & vanishing_point,
                                                  Point2D<float> & road_center_point, float & confidence)
{
  itsRoadMtx.lock(); 
  Point2D<int>      prev_vanishing_point    = itsVanishingPoint;
  std::vector<bool> vanishingPointStability = itsVanishingPointStability;
  itsRoadMtx.unlock(); 

  Point2D<float> target_point;

  size_t num_healthy_lines = 0;
  size_t num_new_lines     = 0;
  size_t num_noisy_lines   = 0;
  size_t num_healthy_active = 0;
  int const width = edgeMap.cols; int const height = edgeMap.rows;
  int const horiline = roadfinder::horizon::get();
  
  for (Line const & line : lines)
  {
    if (line.scores.size() > 0) ++num_noisy_lines;
    else if (line.start_scores.size() > 0) ++num_new_lines;
    else { ++num_healthy_lines; if (line.isActive) ++num_healthy_active; }
  }

  if (num_healthy_lines == 0 && num_new_lines == 0 && num_healthy_lines == 0) 
  {
    vanishing_point   = prev_vanishing_point; 
    road_center_point = Point2D<float>(width/2, height-1);
    target_point      = road_center_point;
    confidence        = .4;
  }
  else 
  {
    std::vector<Line> temp_lines; float weight = 0.0;    
    if (num_healthy_lines > 0)
    {
      for (Line const & line : lines)
        if (line.scores.size() == 0 && line.start_scores.size() == 0) temp_lines.push_back(line);
      vanishing_point = getVanishingPoint(temp_lines, weight);
    }
    else if (num_new_lines > 0)
    {
      for (Line const & line : lines) if (line.start_scores.size() > 0) temp_lines.push_back(line);
      vanishing_point = getVanishingPoint(temp_lines, weight);
      weight -= .1;
    }
    else if (num_noisy_lines > 0)
    {
      for (Line const & line : lines) if (line.scores.size() > 0) temp_lines.push_back(line);
      vanishing_point = getVanishingPoint(temp_lines, weight);
      weight -= .2;
    }
    
    road_center_point = Point2D<float>(width/2, height-1);
    target_point      = road_center_point;
    confidence        = weight;
  }

  if (num_healthy_lines > 0 && num_healthy_active == 0)
  {
    // reset the angle to the center lines 
    for (Line & line : lines)
    {
      if (line.scores.size() == 0 && line.start_scores.size() == 0)
      {
        line.isActive      = true;              
        line.angleToCenter = M_PI/2 - line.angle;
        line.pointToServo  = line.onScreenRoadBottomPoint; 
        line.offset        = 0.0f;
      }
    }
  }

  // get the average active center angle
  float total_weight = 0.0f;
  float total_angle  = 0.0f; 
  int   num_a_line   = 0;
  
  float total_curr_offset  = 0.0f;
  for (Line const & line : lines)
    if (line.scores.size() == 0 && line.start_scores.size() == 0 && line.isActive)
    {
      num_a_line++;
      
      // compute angle to road center
      float angle  = line.angle;
      float cangle = line.angleToCenter;
      float weight = line.score;
      
      float ccangle = cangle + angle; 
      
      total_angle  += ccangle * weight;
      total_weight += weight;
      
      // compute center point
      float dist = height - horiline;
      float tpi = 0.0;  
      float cos_ang = cos(M_PI/2 - ccangle);
      if (cos_ang != 0.0F) tpi = dist / cos_ang * sin(M_PI/2 - ccangle);
      tpi += vanishing_point.i;
      
      // compute offset from point to servo
      float c_offset = 0.0f;
      
      Point2D<float> c_pt  = line.onScreenRoadBottomPoint; 
      Point2D<float> pt_ts = line.pointToServo;
      Point2D<float> h_pt  = line.onScreenHorizonPoint;
      float offset = line.offset;
      if (fabs(pt_ts.i) < 0.05F || fabs(pt_ts.i - width) < 0.05F)
      {
        // figure out where the point would be for the specified 'j' component
        Point2D<float> h1(0    , pt_ts.j);
        Point2D<float> h2(width, pt_ts.j);
        c_pt = intersectPoint(h_pt, c_pt, h1, h2); 
      }
      c_offset = c_pt.i - pt_ts.i;
      total_curr_offset += (c_offset + offset) * weight;
    }

  // calculate confidence
  float avg_weight = 0.0F;
  if (num_a_line > 0) avg_weight = total_weight / num_a_line;
  confidence = avg_weight;
  if (avg_weight == 0) return road_center_point;

  // angle-based road center estimation
  float avg_angle  = 0.0F;
  if (total_weight > 0) avg_angle = total_angle / total_weight;
  float dist = height - horiline;
  float tpi = 0.0;  
  float cos_ang = cos(M_PI/2 - avg_angle);
  if (fabs(cos_ang) > 0.001F) tpi = dist / cos_ang * sin(M_PI/2 - avg_angle);
  tpi += float(width) / 2.0F;
  target_point = Point2D<float>(tpi, height-1);

  // offset point-based road center estimation
  float avg_offset = 0.0F; 
  if (total_weight > 0) avg_offset = total_curr_offset / total_weight;
  float cpi = float(width) / 2.0F + avg_offset;
  road_center_point = Point2D<float>(cpi, height-1);

  // if at least one line is active activate all the healthy but inactive lines
 for (Line & line : lines)
  {
    if (avg_weight > 0.0 && line.scores.size() == 0 && line.start_scores.size() == 0 && !line.isActive)
    {
      // calculate angle                  
      float angle = line.angle;
      float ccangle = avg_angle - angle;
      line.angleToCenter = ccangle;
      line.pointToServo  = line.onScreenRoadBottomPoint;
      line.offset        = avg_offset;
      line.isActive      = true; 
    }
  }
  
  return road_center_point;
}

//######################################################################
void RoadFinder::updateRoadModel(std::vector<Line> & lines, int index)
{
  size_t n_road_lines = itsRoadModel.lines.size();
  size_t n_in_lines   = lines.size(); 
  
  // update only on the healthy input lines
  std::vector<bool> is_healthy(n_in_lines);
  for (size_t i = 0; i < n_in_lines; ++i)
    is_healthy[i] = (lines[i].scores.size() == 0 && lines[i].start_scores.size() == 0);
  
  std::vector<int> road_match_index(n_road_lines, -1);
  std::vector<int> in_match_index(n_in_lines, -1);
  
  std::vector<std::vector<float> > match_dists;
  for (size_t i = 0; i < n_in_lines; ++i) match_dists.push_back(std::vector<float>(n_road_lines));

  // go through each input and road line combination to get the match score which is a simple closest point proximity
  for (size_t i = 0; i < n_in_lines; ++i)
  {
    if (!is_healthy[i]) continue;
    
    Point2D<float> ipt = lines[i].onScreenRoadBottomPoint; 
    Point2D<float> hpt = lines[i].horizonPoint;
    
    for (size_t j = 0; j < n_road_lines; ++j)
    {
      Point2D<float> lshpt = itsRoadModel.lastSeenHorizonPoint[j];
      Point2D<float> lsl   = itsRoadModel.lastSeenLocation[j];
      
      float dist  = lsl.distance(ipt);
      float hdist = hpt.distance(lshpt);
      
      if (hdist > 50) match_dists[i][j] = dist + hdist; 
      else match_dists[i][j] = dist;
    }
  }
  
  // calculate the best match and add it
  for (size_t i = 0; i < n_in_lines; ++i)
  {
    if (!is_healthy[i]) continue;
    
    // get the (best and second best) match and scores
    int   m1_index = -1;      float m1_dist = -1.0; 
    int   m2_index = -1;      float m2_dist = -1.0; 
    for (size_t j = 0; j < n_road_lines; ++j)
    {
      if (road_match_index[j] != -1) continue;
      
      float dist = match_dists[i][j];
      if (m1_index == -1 || dist < m1_dist)
      {
        m2_index = m1_index; m2_dist  = m1_dist;
        m1_index = j       ; m1_dist  = dist; 
      }
      else if (m2_index == -1 || dist < m2_dist)
      {
        m2_index = j       ; m2_dist  = dist; 
      }
    }
    
    // get the best match for road with many evidences
    int   ml1_index = -1;      float ml1_dist = -1.0; 
    int   ml2_index = -1;      float ml2_dist = -1.0; 
    for (size_t j = 0; j < n_road_lines; ++j)
    {
      int nmatch = itsRoadModel.numMatches[j];
      if (road_match_index[j] != -1 || nmatch < 10) continue;
      
      float dist = match_dists[i][j];
      if (ml1_index == -1 || dist < ml1_dist)
      {
        ml2_index = ml1_index; ml2_dist  = ml1_dist;
        ml1_index = j        ; ml1_dist  = dist; 
      }
      else if (ml2_index == -1 || dist < ml2_dist)
      {
        ml2_index = j       ; ml2_dist  = dist; 
      }
    }
    
    // if there first 
    int j = -1;
    
    // check the large matches first
    if (ml1_dist != -1.0F)
      if (ml1_dist < 15.0F || ((ml2_dist != -1.0 || ml1_dist/ml2_dist < .1) && ml1_dist < 30.0F))
        j = ml1_index;      
    
    // then the smaller matches
    if (j == -1 && m1_dist != -1.0F)
      if (m1_dist < 5.0F || ((m2_dist != -1.0 || m1_dist/m2_dist < .1) && m1_dist < 20.0F)) 
        j = m1_index;      
    
    if (j != -1)
    {
      road_match_index[j] = i;              
      in_match_index[i]   = j;
      
      // use the model parameters
      lines[i].angleToCenter = itsRoadModel.lines[j].angleToCenter;
      lines[i].pointToServo  = itsRoadModel.lines[j].pointToServo; 
      lines[i].offset        = itsRoadModel.lines[j].offset;
      lines[i].index         = itsRoadModel.lines[j].index;
      lines[i].isActive = true;
      
      // update the road model 
      Point2D<float> hpt = lines[i].horizonPoint; 
      Point2D<float> ipt = lines[i].onScreenRoadBottomPoint; 
      
      itsRoadModel.lastSeenHorizonPoint[j] = hpt;
      itsRoadModel.lastSeenLocation[j]     = ipt;
      itsRoadModel.lastActiveIndex[j]      = index;
      
      int nmatch = itsRoadModel.numMatches[j];
      itsRoadModel.numMatches[j]   = nmatch+1;
    }
  }

  // delete inactive road model lines
  std::vector<Line>::iterator l_itr = itsRoadModel.lines.begin();
  std::vector<Point2D<float> >::iterator h_itr = itsRoadModel.lastSeenHorizonPoint.begin();
  std::vector<Point2D<float> >::iterator p_itr = itsRoadModel.lastSeenLocation.begin();
  std::vector<int>::iterator i_itr = itsRoadModel.lastActiveIndex.begin();
  std::vector<int>::iterator n_itr = itsRoadModel.numMatches.begin();

  while (l_itr != itsRoadModel.lines.end())
  {
    int lindex = *i_itr;
    if (index > lindex+300) 
    {
      l_itr = itsRoadModel.lines.erase(l_itr);
      h_itr = itsRoadModel.lastSeenHorizonPoint.erase(h_itr);
      p_itr = itsRoadModel.lastSeenLocation.erase(p_itr);
      i_itr = itsRoadModel.lastActiveIndex.erase(i_itr);
      n_itr = itsRoadModel.numMatches.erase(n_itr);
    }
    else { l_itr++; p_itr++; i_itr++; n_itr++; }
  }
  
  n_road_lines = itsRoadModel.lines.size();

  // add all the lines not yet added to the road model  
  for (size_t i = 0; i < n_in_lines; ++i)
  {
    if (!is_healthy[i]) continue;
    if (in_match_index[i] != -1) continue;
    
    lines[i].index         = itsNumIdentifiedLines++; 
    
    Point2D<float> hpt = lines[i].horizonPoint; 
    Point2D<float> ipt = lines[i].onScreenRoadBottomPoint; 
    
    // update the road model
    itsRoadModel.lines.push_back(lines[i]);
    itsRoadModel.lastSeenHorizonPoint.push_back(hpt);
    itsRoadModel.lastSeenLocation.push_back(ipt);
    itsRoadModel.lastActiveIndex.push_back(index);
    itsRoadModel.numMatches.push_back(1);
  }
}

//######################################################################
Point2D<int> RoadFinder::getVanishingPoint(std::vector<Line> const & lines, float & confidence)
{
  // get the horizon points, do a weighted average
  float total_weight = 0.0F;
  float total_hi     = 0.0F;
  int   num_hi       = 0;
  for (Line const & l : lines)
  {
    float hi     = l.horizonPoint.i;
    float weight = l.score;
    
    total_hi     += hi*weight;
    total_weight += weight;
    num_hi++;
  }

  float wavg_hi    = 0.0F;
  float avg_weight = 0.0F;
  if (num_hi > 0) 
  {
    wavg_hi    = total_hi / total_weight;
    avg_weight = total_weight / num_hi;
  }
  confidence = avg_weight;

  return Point2D<int>(wavg_hi, roadfinder::horizon::get());
}

//######################################################################
void RoadFinder::computeHoughSegments(cv::Mat const & cvImage)
{
  int const threshold = 10, minLineLength = 5, maxGap =  2;
  std::vector<cv::Vec4i> segments;
  cv::HoughLinesP(cvImage, segments, 1, CV_PI/180, threshold, minLineLength, maxGap);
  
  itsCurrentSegments.clear();

  for (cv::Vec4i const & seg : segments)
  {
    Point2D<int> pt1(seg[0], seg[1]);
    Point2D<int> pt2(seg[2], seg[3]);
    
    int dx = pt2.i - pt1.i;
    int dy = pt2.j - pt1.j;
    
    float length = pow(dx * dx + dy * dy, .5);
    float angle  = atan2(dy, dx) * 180.0F /M_PI;
    
    int horizon_y   = roadfinder::horizon::get();
    int horizon_s_y = horizon_y + roadfinder::support::get();
    bool good_horizon_support =
      (pt1.j > horizon_y && pt2.j > horizon_y) && (pt1.j > horizon_s_y || pt2.j > horizon_s_y);
    
    bool non_vertical = !((angle > 80.0F && angle < 100.0F) ||  (angle < -80.0F && angle > -100.0F));
    
    if (length > 5.0F && good_horizon_support && non_vertical)
      itsCurrentSegments.push_back(Segment(pt1, pt2, angle, length));
  }
}

//######################################################################
std::vector<Line>
RoadFinder::computeVanishingLines(cv::Mat const & edgeMap, Point2D<int> const & vanishingPoint,
                                  jevois::RawImage & visual)
{
  int const horiline = roadfinder::horizon::get();
  int const vpdt = roadfinder::distthresh::get();
  
  Point2D<float> h1(0, horiline);
  Point2D<float> h2(edgeMap.cols, horiline);

  uint num_vp = itsVanishingPoints.size();
  std::vector<float> curr_vp_likelihood(num_vp, 0.0F);
  std::vector<std::vector<uint> > curr_vp_support(num_vp);

  for(size_t j = 0; j < itsCurrentSegments.size(); j++)
  {
    Segment const & s = itsCurrentSegments[j];
    Point2D<float> p1(s.p1);
    Point2D<float> p2(s.p2);
    if (p2.j > p1.j) 
    { 
      p1 = Point2D<float>(s.p2.i, s.p2.j); 
      p2 = Point2D<float>(s.p1.i, s.p1.j); 
    }
    
    float length = s.length;      
    
    // compute intersection to vanishing point vertical          
    Point2D<float> p_int = intersectPoint(p1, p2, h1, h2);

    // for each vanishing point
    for (size_t i = 0; i < itsVanishingPoints.size(); ++i)
    {
      Point2D<int> const & vp = itsVanishingPoints[i].vp;
      int p_int_i = int(p_int.i);
      if (!((p1.i <= p2.i && p2.i <= p_int_i && p_int_i <= vp.i+10) || 
           (p1.i >= p2.i && p2.i >= p_int_i && p_int_i >= vp.i-10)   ))
        continue;
      
      float dist  = p_int.distance(Point2D<float>(vp.i, vp.j));
      float d_val = 1.0 - dist / vpdt;
      if (d_val > 0.0F) 
      {
        curr_vp_support[i].push_back(j);
        
        // accumulate likelihood values
        curr_vp_likelihood[i] += d_val*length;
      }
    }
  }
  
  // integrate with previous values: FIXXX
  for (size_t i = 0; i < itsVanishingPoints.size(); ++i)
  {
    Point2D<int> const & vp = itsVanishingPoints[i].vp;
    
    float likelihood = curr_vp_likelihood[i];
    itsVanishingPoints[i].likelihood = likelihood;
    
    // compute prior
    float prior = 0.1;
    if (!(vanishingPoint.i == -1 && vanishingPoint.j == -1))
    {
      float di = fabs(vp.i - vanishingPoint.i);
      prior = 1.0 - di / (edgeMap.cols / 4); 
      if (prior < .1) prior = .1;
    }
    
    itsVanishingPoints[i].prior      = prior;
    itsVanishingPoints[i].posterior  = prior*likelihood;
    
    itsVanishingPoints[i].supportingSegments.clear();
    
    for (uint j = 0; j < curr_vp_support[i].size(); ++j)
      itsVanishingPoints[i].supportingSegments.push_back(itsCurrentSegments[curr_vp_support[i][j]]);
  }
  
  uint max_i = 0;
  float max_p = itsVanishingPoints[max_i].posterior;  
  for (uint i = 0; i < itsVanishingPoints.size(); ++i)
  {
    float posterior = itsVanishingPoints[i].posterior;
    if (max_p < posterior) { max_p = posterior; max_i = i; }
  }
  
  // create vanishing lines
  
  // sort the supporting segments on length
  std::list<Segment> supporting_segments;
  uint n_segments = itsVanishingPoints[max_i].supportingSegments.size();
  for (uint i = 0; i < n_segments; ++i) supporting_segments.push_back(itsVanishingPoints[max_i].supportingSegments[i]);
  supporting_segments.sort();
  supporting_segments.reverse();
  
  std::vector<Line> current_lines;
  std::vector<bool> is_used(n_segments, false);
  
  // create lines
  std::list<Segment>::iterator itr = supporting_segments.begin(), stop = supporting_segments.end();
  uint index = 0;
  while (itr != stop)
  {
    Segment const & s2 = *itr;
    itr++; 
    
    if (is_used[index]){ index++; continue; }
    is_used[index] = true;
    index++;
    
    // find other segments with this angle
    float total_length = 0.0F; uint  num_segments = 0;
    Line l = findLine2(s2, edgeMap, supporting_segments, is_used, total_length, num_segments);

    // check for line fitness
    Point2D<float> const & oshsp = l.onScreenHorizonSupportPoint;
    Point2D<float> const & osrbp = l.onScreenRoadBottomPoint;

    Point2D<int> hpt(oshsp + .5);
    Point2D<int> rpt(osrbp + .5);
    
    float score = getLineFitness(hpt, rpt, edgeMap, visual);
    l.score = score;
    l.start_scores.push_back(score);
    if (score >= .5) current_lines.push_back(l);          
  }
  
  // save the vanishing point
  itsRoadMtx.lock(); 
  itsVanishingPoint = itsVanishingPoints[max_i].vp;  
  itsRoadMtx.unlock(); 
  
  return current_lines;
}

// ######################################################################
std::vector<Point2D<int> >  
RoadFinder::getPixels(Point2D<int> const & p1, Point2D<int> const & p2, cv::Mat const & edgeMap)
{
  std::vector<uint> startIndexes;
  return getPixels(p1, p2, edgeMap, startIndexes);
}

// ######################################################################
std::vector<Point2D<int> >  
RoadFinder::getPixels(Point2D<int> const & p1, Point2D<int> const & p2, cv::Mat const & edgeMap,
                      std::vector<uint> & startIndexes)
{
  std::vector<Point2D<int> > points;
  
  // from Graphics Gems / Paul Heckbert
  const int w = edgeMap.cols, h = edgeMap.rows;
  int dx = p2.i - p1.i, ax = abs(dx) << 1, sx = dx < 0 ? -1 : 1;
  int dy = p2.j - p1.j, ay = abs(dy) << 1, sy = dy < 0 ? -1 : 1;
  int x = p1.i, y = p1.j;
  
  // flag to start new segment for the next hit
  bool start_segment = true; startIndexes.clear();
  
  if (ax > ay)
  {
    int d = ay - (ax >> 1);
    for (;;)
    {
      bool adding = false;
      
      if (coordsOk(x, y, w, h))
      {
        if (edgeMap.at<byte>(y, x) > 0)
        { points.push_back(Point2D<int>(x,y)); adding = true; }
        else if (points.size() > 0)
        {
          uint size = points.size();
          Point2D<int> ppt = points[size-1];
          
          // get points that are neighbors of the previous point
          for (int di = 0; di <= 1; di++)
            for (int dj = 0; dj <= 1; dj++)
            {
              if (di == 0 && dj == 0) continue;
              if (!coordsOk(x+di, y+dj, w, h)) continue;
              
              if (edgeMap.at<byte>(y+dj, x+di) && abs((x+di)-ppt.i) <= 1 && abs((y+dj)-ppt.j) <= 1)
              { points.push_back(Point2D<int>(x+di, y+dj)); adding = true; }
            }
        }
        
        if (start_segment && adding) { startIndexes.push_back(points.size()-1); start_segment = false; }
        else if (!start_segment && !adding && Point2D<int>(x, y).distance(points[points.size()-1]) > 1.5)
          start_segment = true;
      }
      
      if (x == p2.i) break;
      if (d >= 0) { y += sy; d -= ax; }
      x += sx; d += ay;
    }
  }
  else
  {
    int d = ax - (ay >> 1);
    for (;;)
    {
      bool adding = false;
      
      if (x >= 0 && x < w && y >= 0 && y < h)
      {
        if (edgeMap.at<byte>(y, x) > 0)
        { points.push_back(Point2D<int>(x,y)); adding = true; }
        else if (points.size() > 0)
        {
          uint size = points.size();
          Point2D<int> ppt = points[size-1];
          
          // get points that are neighbors of the previous point
          for (int di = 0; di <= 1; di++)
            for (int dj = 0; dj <= 1; dj++)
            {
              if (di == 0 && dj == 0) continue;
              if (!coordsOk(x+di, y+dj, w, h)) continue;
              
              if (edgeMap.at<byte>(y+dj, x+di) && abs((x+di)-ppt.i) <= 1 && abs((y+dj)-ppt.j) <= 1)
              { points.push_back(Point2D<int>(x+di,y+dj)); adding = true; }
            }
        }
        
        if (start_segment && adding) { startIndexes.push_back(points.size()-1); start_segment = false; }
        else if (!start_segment && !adding && Point2D<int>(x, y).distance(points[points.size()-1]) > 1.5)
          start_segment = true;
      }
      
      if (y == p2.j) break;
      if (d >= 0) { x += sx; d -= ay; }
      y += sy; d += ax;
    }
  }
  return points;
}

// ######################################################################
std::vector<Point2D<int> >  
RoadFinder::getPixelsQuick(Point2D<int> const & p1, Point2D<int> const & p2, cv::Mat const & edgeMap)
{
  std::vector<Point2D<int> > points;

  // from Graphics Gems / Paul Heckbert
  const int w = edgeMap.cols, h = edgeMap.rows;
  int dx = p2.i - p1.i, ax = abs(dx) << 1, sx = dx < 0 ? -1 : 1;
  int dy = p2.j - p1.j, ay = abs(dy) << 1, sy = dy < 0 ? -1 : 1;
  int x = p1.i, y = p1.j;
  
  // flag to start new segment for the next hit
  if (ax > ay)
  {
    int d = ay - (ax >> 1);
    for (;;)
    {
      if (x >= 0 && x < w && y >= 0 && y < h && edgeMap.at<byte>(y, x) > 0) points.push_back(Point2D<int>(x, y)); 
      if (x == p2.i) break; else if (d >= 0) { y += sy; d -= ax; }
      x += sx; d += ay;
    }
  }
  else
  {
    int d = ax - (ay >> 1);
    for (;;)
    {
      if (x >= 0 && x < w && y >= 0 && y < h && edgeMap.at<byte>(y, x) > 0) points.push_back(Point2D<int>(x, y)); 
      if (y == p2.j) break; else if (d >= 0) { x += sx; d -= ay; }
      y += sy; d += ax;
    }
  }
  return points;
}


// ######################################################################
Line RoadFinder::findLine2(Segment const & s, cv::Mat const & edgeMap, std::list<Segment> const & supportingSegments,
                           std::vector<bool> & is_used, float & totalLength, uint & numSegments)
{
  Point2D<int> const & p1 = s.p1; Point2D<int> const & p2 = s.p2;

  Line l; l.segments.push_back(s); l.length = s.length;
  std::vector<Point2D<int> > points = getPixels(p1, p2, edgeMap);

  float const distance_threshold  = 7.0F; float const distance_threshold2 = 5.0F;

  // find points within distance
  size_t index = 0; totalLength = s.length; numSegments = 1;
  
  std::list<Segment>::const_iterator itr = supportingSegments.begin();
  for (size_t i = 0; i < is_used.size(); ++i)
  { 
    Segment const & s2 = (*itr);
    Point2D<int> const & p2_1 = s2.p1; Point2D<int> const & p2_2 = s2.p2; float const length = s2.length;
    ++itr; ++index;
    
    if (is_used[i]) continue; 
    
    int mid_left_count = 0, mid_right_count = 0;
    bool is_inline = true, is_close_inline = true;
    std::vector<Point2D<int> > curr_points = getPixels(p2_1, p2_2, edgeMap);

    for (size_t j = 0; j < curr_points.size(); ++j)
    {
      float const dist = distance(p1, p2, curr_points[j]);
      int   const wsid = side(p1, p2, curr_points[j]);
      
      if (wsid <= 0) ++mid_left_count;
      if (wsid >= 0) ++mid_right_count;
      if (dist > distance_threshold2) { is_close_inline = false; }
      if (dist > distance_threshold) { is_inline = false; j = curr_points.size(); }
    }
    
    // include 
    if (is_close_inline || (is_inline && mid_left_count >= 2 && mid_right_count >= 2))
    {
      for (size_t j = 0; j < curr_points.size(); ++j) points.push_back(curr_points[j]);
      is_used[i] = true; totalLength += length; ++numSegments;
      
      std::vector<Point2D<int> > curr_points2 = getPixels(p2_1, p2_2, edgeMap);
      for (size_t j = 0; j < curr_points2.size(); ++j) points.push_back(curr_points2[j]);
    }
  }
  
  updateLine(l, points, totalLength, edgeMap.cols, edgeMap.rows);
  Point2D<float> const & point1 = l.onScreenHorizonSupportPoint;
  Point2D<float> const & point2 = l.onScreenRoadBottomPoint;
  float dist = point1.distance(point2);
  float score = l.score / dist; // FIXME div by zero??
  l.score = score;
  return l;
}

// ######################################################################
void RoadFinder::updateLine(Line & l, std::vector<Point2D<int> > const & points, float score,
                            int const width, int const height)
{
  if (points.empty()) { l.score = -1.0F; return; }

  int const horiline = roadfinder::horizon::get();
  int const horisupp = horiline + roadfinder::support::get();
  
  // fit a line using all the points
  Point2D<float> lp1, lp2; fitLine(points, lp1, lp2, width, height);
  l.points = points;
  l.score  = score;
  
  Point2D<float> tr1 = intersectPoint(lp1, lp2, Point2D<float>(0, horiline), Point2D<float>(width, horiline));
  Point2D<float> tr2 = intersectPoint(lp1, lp2, Point2D<float>(0, height-1), Point2D<float>(width, height-1)); 
  Point2D<float> tr3 = intersectPoint(lp1, lp2, Point2D<float>(0, horisupp), Point2D<float>(width, horisupp));
  
  l.horizonPoint            = tr1;
  l.horizonSupportPoint     = tr3;
  l.roadBottomPoint         = tr2;

  if (tr2.i >= 0 && tr2.i <= width) l.onScreenRoadBottomPoint = tr2;
  else if (tr2.i < 0)
    l.onScreenRoadBottomPoint = intersectPoint(lp1, lp2, Point2D<float>(0, 0), Point2D<float>(0,height));
  else if (tr2.i > width)
    l.onScreenRoadBottomPoint = intersectPoint(lp1, lp2, Point2D<float>(width, 0), Point2D<float>(width,height));
  
  if (tr1.i >= 0 && tr1.i <= width) l.onScreenHorizonPoint = tr1;
  else if (tr1.i < 0)
    l.onScreenHorizonPoint = intersectPoint(lp1, lp2, Point2D<float>(0, 0), Point2D<float>(0,height));
  else if (tr1.i > width)
    l.onScreenHorizonPoint = intersectPoint(lp1, lp2, Point2D<float>(width, 0), Point2D<float>(width,height));
  
  if (tr3.i >= 0 && tr3.i <= width) l.onScreenHorizonSupportPoint = tr3;
  else if (tr3.i < 0)
    l.onScreenHorizonSupportPoint = intersectPoint(lp1, lp2, Point2D<float>(0, 0), Point2D<float>(0,height));
  else if (tr3.i > width)
    l.onScreenHorizonSupportPoint = intersectPoint(lp1, lp2, Point2D<float>(width, 0), Point2D<float>(width,height));
  
  Point2D<float> const & p1 = l.horizonPoint;
  Point2D<float> const & p2 = l.roadBottomPoint;

  float dy = p2.j - p1.j;
  float dx = p2.i - p1.i;

  // set it to 0 to M_PI
  float angle = atan2(dy, dx); 
  if (angle < 0.0) angle = M_PI + angle;
  
  l.angle = angle;
}

// ######################################################################
void RoadFinder::fitLine(std::vector<Point2D<int> > const & points, Point2D<float> & p1,Point2D<float> & p2,
                         int const width, int const height)
{
  float line[4];

  CvPoint* cvPoints = (CvPoint*)malloc(points.size() * sizeof(Point2D<int>));
  for (size_t i = 0; i < points.size(); ++i) { cvPoints[i].x = points[i].i; cvPoints[i].y = points[i].j; }

  {
    CvMat point_mat = cvMat(1, points.size(), CV_32SC2, cvPoints);
    cvFitLine(&point_mat, CV_DIST_L2, 0, 0.01, 0.01, line);
  }
  
  free(cvPoints);
  
  float const d = sqrtf(line[0]*line[0] + line[1]*line[1]);  
  line[0] /= d; line[1] /= d;  

  float const t = width + height;
  p1.i = line[2] - line[0] * t;  
  p1.j = line[3] - line[1] * t;  
  p2.i = line[2] + line[0] * t;  
  p2.j = line[3] + line[1] * t;  
}

// ######################################################################
float RoadFinder::getLineFitness(Point2D<int> const & horizonPoint, Point2D<int> const & roadBottomPoint,
                                 cv::Mat const & edgeMap, jevois::RawImage & visual)
{
  std::vector<Point2D<int> > points;
  return getLineFitness(horizonPoint, roadBottomPoint, edgeMap, points, visual);
}

// ######################################################################
float RoadFinder::getLineFitness(Point2D<int> const & horizonPoint, Point2D<int> const & roadBottomPoint,
                                 cv::Mat const & edgeMap, std::vector<Point2D<int> > & points,
                                 jevois::RawImage & visual)
{
  float score = 0;
  int min_effective_segment_size = 5;

  // go through the points in the line
  Point2D<int> p1 = horizonPoint;
  Point2D<int> p2 = roadBottomPoint;
  float dist  = p1.distance(p2);

  points.clear();
  std::vector<uint> start_indexes;
  points = getPixels(p1, p2, edgeMap, start_indexes);

  int sp = 4;
  std::vector<Point2D<int> > lpoints = getPixelsQuick(p1+Point2D<int>(-sp,0), p2+Point2D<int>(-sp,0), edgeMap);
  std::vector<Point2D<int> > rpoints = getPixelsQuick(p1+Point2D<int>(sp,0), p2+Point2D<int>(sp,0), edgeMap);
  
  uint num_segments = start_indexes.size();
  float max_length  = 0.0F;
  int total         = 0; int max = 0;
  float eff_length  = 0;
  for (uint i = 1; i < num_segments; ++i)
  {
    int size = start_indexes[i] - start_indexes[i-1];
    if (max < size) max = size;
    total += size;
    
    uint i1 = start_indexes[i-1];
    uint i2 = start_indexes[i  ]-1;
    Point2D<int> pt1 = points[i1]; 
    Point2D<int> pt2 = points[i2];
    float length = pt1.distance(pt2);
    if (max_length < length) max_length = length;
    
    if (size >= min_effective_segment_size) eff_length+= length;
  }
  
  if (num_segments > 0) 
  {
    int size = int(points.size()) - int(start_indexes[num_segments-1]);
    
    if (max < size) max = size;
    total += size;
    
    uint i1 = start_indexes[num_segments-1 ];
    uint i2 = points.size()-1;
    
    Point2D<int> pt1 = points[i1];
    Point2D<int> pt2 = points[i2]; 
    float length = pt1.distance(pt2);
    
    if (max_length < length) max_length = length;
    
    if (size >= min_effective_segment_size) eff_length+= length;
  }
  
  if (max_length > 0.0)
  {
    uint lsize = lpoints.size();
    uint rsize = rpoints.size();

    // can't be bigger than 15 degrees or bad point
    if (dist <= 50.0 || points.size() < 2*(lsize+rsize)) score = 0.0;
    else score = eff_length/dist;
  }
  
  if (visual.valid())
  {
    jevois::rawimage::drawLine(visual, p1.i, p1.j, p2.i, p2.j, 0, 0x80ff);
    for (Point2D<int> const & p : points) jevois::rawimage::drawDisk(visual, p.i, p.j, 1, jevois::yuyv::LightTeal);
    for (Point2D<int> const & p : lpoints) jevois::rawimage::drawDisk(visual, p.i, p.j, 1, jevois::yuyv::LightTeal);
    for (Point2D<int> const & p : rpoints) jevois::rawimage::drawDisk(visual, p.i, p.j, 1, jevois::yuyv::LightTeal);
  }
  return score;
}

// ######################################################################
void RoadFinder::trackVanishingLines(cv::Mat const & edgeMap, std::vector<Line> & currentLines,
                                     jevois::RawImage & visual)
{
  for (Line & line : currentLines)
  {
    Point2D<int> pi1(line.onScreenHorizonSupportPoint + 0.5F);
    Point2D<int> pi2(line.onScreenRoadBottomPoint + 0.5F);
    
    float max_score = 0.0F;
    std::vector<Point2D<int> > max_points;

    for (int di1 = -10; di1 <= 10; di1 += 2)
      for (int di2 = -10; di2 <= 10; di2 += 2) 
      {
        Point2D<int> pn1(pi1.i + di1, pi1.j);
        Point2D<int> pn2(pi2.i + di2, pi2.j);
        std::vector<Point2D<int> > points;
        
        float score = getLineFitness(pn1, pn2, edgeMap, points, visual);

        // Debug drawing:
        if (visual.valid())
        {
          Line l; updateLine(l, points, score, edgeMap.cols, edgeMap.rows);
          jevois::rawimage::drawLine(visual, pn1.i, pn1.j, pn2.i, pn2.j, 0, jevois::yuyv::MedPurple);
          for (Point2D<int> const & p : points)
            jevois::rawimage::drawDisk(visual, p.i, p.j, 1, jevois::yuyv::MedPurple);
        }

        if (score > max_score) { max_score = score; max_points = points; }
      }
    
    // update the vanishing line
    if (max_score > 0) updateLine(line, max_points, max_score, edgeMap.cols, edgeMap.rows); else line.score = max_score;
    line.segments.clear();
  }
  
  // check for start values 
  std::vector<Line> temp_lines;
  for (Line & line : currentLines)
  {
    uint num_sscore = line.start_scores.size();
    
    // still in starting stage
    if (num_sscore > 0)
    {
      line.start_scores.push_back(line.score);
      ++num_sscore;
      
      uint num_high = 0;
      for (uint j = 0; j < num_sscore; ++j) if (line.start_scores[j] > .5) num_high++;
      
      if (num_high > 5) { line.start_scores.clear(); num_sscore = 0; }
      if (num_sscore < 7) temp_lines.push_back(line);
    }
    else temp_lines.push_back(line);
  }
  currentLines.clear();
  for (uint i = 0; i < temp_lines.size(); ++i) currentLines.push_back(temp_lines[i]);

  // check lines to see if any lines are below the threshold
  temp_lines.clear();
  for (Line & line : currentLines)
  {
    if (line.score < 0.3 || line.scores.size() > 0) line.scores.push_back(line.score);
    
    uint num_low = 0; int size = line.scores.size(); bool all_good_values = true;
    for (int j = 0; j < size; ++j) if (line.scores[j] < 0.3) { num_low++; all_good_values = false; }
    
    // keep until 5 of 7 bad values
    if (num_low < 5)
    {     
      // update the values
      if (all_good_values) line.scores.clear();
      else
      {
        std::vector<float> vals;
        if (size > 7)
        {
          for (int j = size-7; j < size; ++j) vals.push_back(line.scores[j]);
          line.scores = vals;
        }
      }
      temp_lines.push_back(line);          
    }
  }

  currentLines.clear();
  for (uint i = 0; i < temp_lines.size(); ++i) currentLines.push_back(temp_lines[i]);
}

// ######################################################################
void RoadFinder::projectForwardVanishingLines(std::vector<Line> & lines, std::vector<cv::Mat> const & edgeMaps,
                                              jevois::RawImage & visual)
{
  // project forward the lines using all the frames that are just passed
  for (cv::Mat const & em : edgeMaps) trackVanishingLines(em, lines, visual);
}

// ######################################################################
std::vector<Line> RoadFinder::combine(std::vector<Line> & prevLines, std::vector<Line> const & currentLines,
                                      int width, int height)
{
  std::vector<Line> combLines;
  std::vector<bool> cline_isadded(currentLines.size(), false);
  prevLines = discardDuplicates(prevLines);

  // integrate the two trackers  
  for (size_t j = 0; j < prevLines.size(); ++j)
  {
    Point2D<float> const & pp1 = prevLines[j].onScreenHorizonSupportPoint;
    Point2D<float> const & pp2 = prevLines[j].onScreenRoadBottomPoint;         
    float const score_pl = prevLines[j].score;
    
    float min_dist = 1.0e30F; int min_i = -1;
    std::vector<size_t> match_index;
    for (size_t i = 0; i < currentLines.size(); ++i)
    {
      Point2D<float> const & cp1 = currentLines[i].onScreenHorizonSupportPoint;
      Point2D<float> const & cp2 = currentLines[i].onScreenRoadBottomPoint;
      
      // check the two ends of the vanishing points
      float const dist = cp1.distance(pp1) + cp2.distance(pp2);
    
      // if the lines are close enough
      if (dist < 7.0F) { match_index.push_back(i); if (dist < min_dist) { min_dist = dist; min_i = i; } }
    }
    
    // combine lines if there are duplicates
    if (match_index.size() > 0)
    {
      // if matched with more than 1 pick the closest one and use the one with the higher score
      float score_mcl = currentLines[min_i].score;   
      Line l;
      if (score_pl > score_mcl)
        combLines.push_back(prevLines[j]);
      else
      {
        updateLine(l, currentLines[min_i].points, score_mcl, width, height);
        
        l.start_scores = prevLines[j].start_scores;               
        size_t ss_size = l.start_scores.size(); 
        if (ss_size > 0) l.start_scores[ss_size-1] = score_mcl;
        
        l.scores = prevLines[j].start_scores; 
        size_t s_size = l.scores.size(); 
        if (s_size > 0) l.scores[s_size-1] = score_mcl;
        
        l.isActive      = prevLines[j].isActive;
        l.angleToCenter = prevLines[j].angleToCenter;
        l.pointToServo  = prevLines[j].pointToServo;
        l.offset        = prevLines[j].offset;
        l.index         = prevLines[j].index;
        combLines.push_back(l);
      }
      
      // but all the other lines are discarded
      for (uint i = 0; i < match_index.size(); ++i) cline_isadded[match_index[i]] = true;
    }
    else combLines.push_back(prevLines[j]);
  }
  
  for (uint i = 0; i < cline_isadded.size(); ++i) if (!cline_isadded[i]) combLines.push_back(currentLines[i]);
  
  return combLines;
}

// ######################################################################
std::vector<Line> RoadFinder::discardDuplicates(std::vector<Line> const & lines)
{
  std::vector<Line> newLines; std::vector<bool> line_isadded(lines.size(), false);
  
  for (size_t j = 0; j < lines.size(); ++j)
  {
    if (line_isadded[j]) continue;
    
    Line line_added = lines[j];
    Point2D<float> const & pp1 = line_added.onScreenHorizonSupportPoint;
    Point2D<float> const & pp2 = line_added.onScreenRoadBottomPoint;         
    
    for (size_t i = j + 1; i < lines.size(); ++i)
    {
      if (line_isadded[i]) continue;
      
      Point2D<float> const & cp1 = lines[i].onScreenHorizonSupportPoint;
      Point2D<float> const & cp2 = lines[i].onScreenRoadBottomPoint;
      float const score_cl2 = lines[i].score;
      
      // check the two ends of the vanishing points
      float const dist = cp1.distance(pp1) + cp2.distance(pp2);
      
      // if the lines are close enough
      if (dist < 3.0F)
      {
        line_isadded[i] = true;
        if (line_added.score < score_cl2) line_added = lines[i];
      }
    }
    newLines.push_back(line_added);
  }
  return newLines;
}
