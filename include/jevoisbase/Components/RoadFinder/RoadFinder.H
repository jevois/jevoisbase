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

#pragma once

#include <jevois/Component/Component.H>
#include <jevois/Image/RawImage.H>
#include <opencv2/core/core.hpp>
#define INVT_TYPEDEF_INT64
#define INVT_TYPEDEF_UINT64
#include <jevoisbase/Components/RoadFinder/Point2D.H>
#include <opencv2/video/tracking.hpp> // for kalman filter

// ######################################################################

//! a segment is defined by the two end-points
/*! \relates RoadFinder*/
struct Segment
{
    Segment(Point2D<int> in_p1, Point2D<int> in_p2, float in_angle, float in_length) :
        p1(in_p1), p2(in_p2), angle(in_angle), length(in_length)
    { }
    
    Point2D<int> p1;
    Point2D<int> p2;
    float angle;
    float length;
    
    bool operator<(const Segment & s) { return length < s.length; }
};

//! Keeps all the supporting information about a specific vanishing point
/*! \relates RoadFinder*/
struct VanishingPoint
{
    VanishingPoint(Point2D<int> in_vp, float in_likelihood) :
        vp(in_vp), likelihood(in_likelihood)
    { }
    
    Point2D<int>         vp;
    float                prior;
    float                likelihood;
    float                posterior;
    std::vector<Segment> supportingSegments;
};

//! keeps all the ready to use information of a supporting line as it pertains to describing the road
/*! \relates RoadFinder*/
struct Line
{
    Line() : length(0.0F), angle(0.0F), score(0.0F), isActive(false), index(-1)
    { }
    
    //! basic information to specify the line
    float length;
    float angle;
    float score;
    
    //! the points that are fit to the line
    std::vector<Point2D<int> > points;
    
    //! quick information for various locations with respect to the road
    Point2D<float> horizonPoint;
    Point2D<float> horizonSupportPoint;
    Point2D<float> roadBottomPoint;
    Point2D<float> onScreenRoadBottomPoint;
    Point2D<float> onScreenHorizonPoint;
    Point2D<float> onScreenHorizonSupportPoint;
    
    //! original supporting segments out of sync after initial frame
    std::vector<Segment> segments;
    
    //! tracking information to monitor health of the line
    std::vector<float> scores;
    std::vector<float> start_scores; 
    
    //! tracks whether the line can be used for finding the road center
    bool isActive;
    float angleToCenter;
    Point2D<float> pointToServo;
    float offset;
    
    int index;
};

//! Store information about the road
/*! \relates RoadFinder*/
struct RoadModel
{
    std::vector<Line>            lines;
    std::vector<int>             lastActiveIndex;
    std::vector<Point2D<float> > lastSeenHorizonPoint;
    std::vector<Point2D<float> > lastSeenLocation;
    std::vector<int>             numMatches;
};

namespace roadfinder
{
  static jevois::ParameterCategory const ParamCateg("RoadFinder Options");

  //! Parameter \relates RoadFinder
  JEVOIS_DECLARE_PARAMETER(horizon, int, "Estimated vertical (Y) position of the horizon (pixels, "
                           "with 0 at the top of the frame). Adjust this depending on the tilt angle of your camera "
                           "and video input resolution.",
                           70, ParamCateg);

  //! Parameter \relates RoadFinder
  JEVOIS_DECLARE_PARAMETER(support, int, "Offset (in pixels) between horizon line and horizon support line (positive "
                           "values means support line is below horizon line.",
                           20, ParamCateg);

  //! Parameter \relates RoadFinder
  JEVOIS_DECLARE_PARAMETER(spacing, unsigned int, "Spacing between vanishing point candidates (pixels).",
                           20, ParamCateg);

  //! Parameter \relates RoadFinder
  JEVOIS_DECLARE_PARAMETER(distthresh, unsigned int, "Vanishing point distance threshold (pixels).",
                           40, ParamCateg);
} // namespace roadfinder


//! Navigation by finding road
/*! This algorithm implements the algorithm described in: C.-K. Chang, C. Siagian, L. Itti, Mobile Robot Monocular
    Vision Navigation Based on Road Region and Boundary Estimation, In: Proc. IEEE/RSJ International Conference on
    Intelligent Robots and Systems (IROS), pp. 1043-1050, Oct 2012.

    See the research paper at http://ilab.usc.edu/publications/doc/Chang_etal12iros.pdf

    \ingroup components */
class RoadFinder : public jevois::Component,
                   public jevois::Parameter<roadfinder::horizon, roadfinder::support,
                                            roadfinder::spacing, roadfinder::distthresh>
{
  public:
    //! constructor
    RoadFinder(std::string const & instance);
    
    //! desctructor
    virtual ~RoadFinder();
    
    //! Compute the vanishing point location using the full blown algorithm
    /*! img should be greyscale. If visual is valid, it should be YUYV with same or larger dims, with the assumption
        that it contains a color copy of the input frame in the top-left corner (this is used for debug visualizations
        of the various things computed). */
    void process(cv::Mat const & img, jevois::RawImage & visual);

    //! Get the current vanishing point and confidence
    std::pair<Point2D<int>, float> getCurrVanishingPoint() const;

    //! Get the current road center point
    Point2D<float> getCurrCenterPoint() const;

    //! Get the current target point
    Point2D<float> getCurrTargetPoint() const;

    //! Get the kalman-fitered target X, can be used to set robot steering
    float getFilteredTargetX() const;
    
    //! Reset all tracker internals and start fresh (e.g., when changing goal direction)
    /*! Thread-safe, ok to call concurrently with process(). */
    void resetRoadModel();

  protected:
    //! This class has state and does not support some online param changes
    void postInit() override;

    //! This class has state and does not support some online param changes
    void preUninit() override;
    
    //! compute the hough segments in the image
    void computeHoughSegments(cv::Mat const & cvImage);
    
    //! main function to detect the road
    std::vector<Line> computeVanishingLines(cv::Mat const & edgeMap, Point2D<int> const & vanishingPoint,
                                            jevois::RawImage & visual);
    
    //! computes the road center point to servo to 
    Point2D<float> computeRoadCenterPoint(cv::Mat const & edgeMap, std::vector<Line> & lines,
                                          Point2D<int> & vanishing_point, 
                                          Point2D<float> & road_center_point, float & confidence);
    
    //! update road model and incoming lines NOTE: also change the line parameters to sync them this avoids drifts
    void updateRoadModel(std::vector<Line> & lines, int index);
    
    //! estimate the vanishing point from the tracked lines
    Point2D<int> getVanishingPoint(std::vector<Line> const & lines, float &confidence);
    
    //! track vanishing lines by to fit to the new, inputted, edgemap
    void trackVanishingLines(cv::Mat const & edgeMap, std::vector<Line> & currentLines, jevois::RawImage & visual);
    
    //! get pixels for segment defined by p1 and p2 have added complexity to search within 1.5 pixels of the line
    std::vector<Point2D<int> >  
    getPixels(Point2D<int> const & p1, Point2D<int> const & p2, cv::Mat const & edgeMap);
    
    //! get pixels for segment defined by p1 and p2 have added complexity to search within 1.5 pixels of the line
    std::vector<Point2D<int> >  
    getPixels(Point2D<int> const & p1, Point2D<int> const & p2, cv::Mat const & edgeMap,
              std::vector<uint>& startIndexes);
    
    //! get pixels that make up the segment defined by p1 and p2
    std::vector<Point2D<int> >  
    getPixelsQuick(Point2D<int> const & p1, Point2D<int> const & p2, cv::Mat const & edgeMap);
    
    //! find lines given the found supporting segments
    Line findLine2(Segment const & s, cv::Mat const & edgeMap, std::list<Segment> const & supportingSegments,
                   std::vector<bool> & is_used, float & totalLength, uint & numSegments);
    
    //! openCV wrapper function to fit a line to an input vector of points
    void fitLine(std::vector<Point2D<int> > const & points, Point2D<float> & p1,Point2D<float> & p2,
                 int const width, int const height);
    
    //! compute how well the line equation fit the edgels in edgemap
    float getLineFitness(Point2D<int> const & horizonPoint, Point2D<int> const & roadBottomPoint,
                         cv::Mat const & edgeMap, jevois::RawImage & visual);
    
    //! compute how well the line equation fit the edgels in edgemap
    float getLineFitness(Point2D<int> const & horizonPoint, Point2D<int> const & roadBottomPoint, 
                         cv::Mat const & edgeMap, std::vector<Point2D<int> > & points, jevois::RawImage & visual);
    
    //! update the information in by updating the input points, score and various handy coordinate locations
    void updateLine(Line & l, std::vector<Point2D<int> > const & points, float score,
                    int const width, int const height);
    
    //! update the lines with the inputted set of edgemaps
    void projectForwardVanishingLines(std::vector<Line> & lines, std::vector<cv::Mat> const & edgeMaps,
                                      jevois::RawImage & visual);
    
    //! combine two lines sets, discard duplicates and overlaps
    std::vector<Line> combine(std::vector<Line> & prevLines, std::vector<Line> const & currentLines,
                              int width, int height);
    
    //! discard duplicate lines in a set
    std::vector<Line> discardDuplicates(std::vector<Line> const & currentLines);
    
    //! the current road heading
    double itsRoadHeading;
    
    //! the accumulated trajectory
    Point2D<float> itsAccumulatedTrajectory;
    
    //! locking accunulated trajectory
    std::mutex itsAccTrajMtx;
    
    //! current segments found using CVHoughlines
    std::vector<Segment> itsCurrentSegments;
    
    std::mutex itsTrackMtx;         //!< locking line trackers vars
    
    //! indicate whether tracking
    bool itsTrackingFlag;                         
    
    //! the current lines being tracked
    std::vector<Line> itsCurrentLines;
    
    //! indicate how many unique lines have been identified NOTE: never reset
    uint itsNumIdentifiedLines;
    
    RoadModel itsRoadModel;
    
    //! vanishing points being considered
    std::vector<VanishingPoint> itsVanishingPoints;
    
    std::mutex itsRoadMtx;
    Point2D<int>      itsVanishingPoint;           //!< current vanishing point
    Point2D<float>    itsCenterPoint;              //!< current center of road point 
    Point2D<float>    itsTargetPoint;              //!< target servo point
    float             itsVanishingPointConfidence; //!< current vanishing point 
    std::vector<bool> itsVanishingPointStability;  //!< vanishing point score tracker
    
    //! for visualizer
    int itsCurrentMessageID;

    cv::KalmanFilter itsTPXfilter;
    float itsFilteredTPX;
    bool itsKalmanNeedInit;
};
