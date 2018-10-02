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

#include <jevois/Core/Module.H>
#include <jevois/Debug/Log.H>
#include <jevois/Util/Utils.H>
#include <jevois/Image/RawImageOps.H>
#include <jevois/Debug/Timer.H>
#include <jevois/Util/Coordinates.H>

#include <jevoisbase/Components/Tracking/Kalman1D.H>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/calib3d/calib3d.hpp>

#include <Eigen/Geometry> // for AngleAxis and Quaternion

// REMINDER: make sure you understand the viral nature and terms of the above license. If you are writing code derived
// from this file, you must offer your source under the GPL license too.

static jevois::ParameterCategory const ParamCateg("FirstVision Options");

//! Parameter \relates FirstVision
JEVOIS_DECLARE_PARAMETER_WITH_CALLBACK(hcue, unsigned char, "Initial cue for target hue (0=red/do not use because of "
				       "wraparound, 30=yellow, 45=light green, 60=green, 75=green cyan, 90=cyan, "
				       "105=light blue, 120=blue, 135=purple, 150=pink)",
				       45, jevois::Range<unsigned char>(0, 179), ParamCateg);

//! Parameter \relates FirstVision
JEVOIS_DECLARE_PARAMETER_WITH_CALLBACK(scue, unsigned char, "Initial cue for target saturation lower bound",
				       50, ParamCateg);

//! Parameter \relates FirstVision
JEVOIS_DECLARE_PARAMETER_WITH_CALLBACK(vcue, unsigned char, "Initial cue for target value (brightness) lower bound",
				       200, ParamCateg);

//! Parameter \relates FirstVision
JEVOIS_DECLARE_PARAMETER(maxnumobj, size_t, "Max number of objects to declare a clean image. If more blobs are "
			 "detected in a frame, we skip that frame before we even try to analyze shapes of the blobs",
                         100, ParamCateg);

//! Parameter \relates FirstVision
JEVOIS_DECLARE_PARAMETER(hullarea, jevois::Range<unsigned int>, "Range of object area (in pixels) to track. Use this "
			 "if you want to skip shape analysis of very large or very small blobs",
                         jevois::Range<unsigned int>(20*20, 300*300), ParamCateg);

//! Parameter \relates FirstVision
JEVOIS_DECLARE_PARAMETER(hullfill, int, "Max fill ratio of the convex hull (percent). Lower values mean your shape "
			 "occupies a smaller fraction of its convex hull. This parameter sets an upper bound, "
			 "fuller shapes will be rejected.",
                         50, jevois::Range<int>(1, 100), ParamCateg);

//! Parameter \relates FirstVision
JEVOIS_DECLARE_PARAMETER(erodesize, size_t, "Erosion structuring element size (pixels), or 0 for no erosion",
                         2, ParamCateg);

//! Parameter \relates FirstVision
JEVOIS_DECLARE_PARAMETER(dilatesize, size_t, "Dilation structuring element size (pixels), or 0 for no dilation",
                         4, ParamCateg);

//! Parameter \relates FirstVision
JEVOIS_DECLARE_PARAMETER(epsilon, double, "Shape smoothing factor (higher for smoother). Shape smoothing is applied "
			 "to remove small contour defects before the shape is analyzed.",
                         0.015, jevois::Range<double>(0.001, 0.999), ParamCateg);

//! Parameter \relates FirstVision
JEVOIS_DECLARE_PARAMETER(debug, bool, "Show contours of all object candidates if true",
                         false, ParamCateg);

//! Parameter \relates FirstVision
JEVOIS_DECLARE_PARAMETER(threads, size_t, "Number of parallel vision processing threads. Thread 0 uses the HSV values "
			 "provided by user parameters; thread 1 broadens that fixed range a bit; threads 2-3 use a "
			 "narrow and broader learned HSV window over time",
                         4, jevois::Range<size_t>(2, 4), ParamCateg);

//! Parameter \relates FirstVision
JEVOIS_DECLARE_PARAMETER(showthread, size_t, "Thread number that is used to display HSV-thresholded image",
                         0, jevois::Range<size_t>(0, 3), ParamCateg);

//! Parameter \relates FirstVision
JEVOIS_DECLARE_PARAMETER(ethresh, double, "Shape error threshold (lower is stricter for exact shape)",
                         900.0, jevois::Range<double>(0.01, 1000.0), ParamCateg);

//! Parameter \relates FirstVision
JEVOIS_DECLARE_PARAMETER(dopose, bool, "Compute (and show) 6D object pose, requires a valid camera calibration. "
			 "When dopose is true, 3D serial messages are sent out, otherwise 2D serial messages.",
			 true, ParamCateg);

//! Parameter \relates FirstVision
JEVOIS_DECLARE_PARAMETER(camparams, std::string, "File stem of camera parameters, or empty. Camera resolution "
			 "will be appended, as well as a .yaml extension. For example, specifying 'calibration' "
			 "here and running the camera sensor at 320x240 will attempt to load "
			 "calibration320x240.yaml from within directory " JEVOIS_SHARE_PATH "/camera/",
			 "calibration", ParamCateg);

//! Parameter \relates FirstVision
JEVOIS_DECLARE_PARAMETER(iou, double, "Intersection-over-union ratio over which duplicates are eliminated",
                         0.3, jevois::Range<double>(0.01, 0.99), ParamCateg);

//! Parameter \relates FirstVision
JEVOIS_DECLARE_PARAMETER(objsize, cv::Size_<float>, "Object size (in meters)",
                         cv::Size_<float>(0.28F, 0.175F), ParamCateg);

//! Parameter \relates FirstVision
JEVOIS_DECLARE_PARAMETER(margin, size_t, "Margin from from frame borders (pixels). If any corner of a detected shape "
			 "gets closer than the margin to the frame borders, the shape will be rejected. This is to "
			 "avoid possibly bogus 6D pose estimation when the shape starts getting truncated as it "
			 "partially exits the camera's field of view.",
                         5, ParamCateg);

//! Simple color-based detection of a U-shaped object for FIRST Robotics
/*! This module isolates pixels within a given HSV range (hue, saturation, and value of color pixels), does some
    cleanups, and extracts object contours. It is looking for a rectangular U shape of a specific size (set by parameter
    \p objsize). See screenshots for an example of shape. It sends information about detected objects over serial.

    This module usually works best with the camera sensor set to manual exposure, manual gain, manual color balance, etc
    so that HSV color values are reliable. See the file \b script.cfg file in this module's directory for an example of
    how to set the camera settings each time this module is loaded.

    This code was loosely inspired by the JeVois \jvmod{ObjectTracker} module. Also see \jvmod{FirstPython} for a
    simplified version of this module, written in Python.

    This module is provided for inspiration. It has no pretension of actually solving the FIRST Robotics vision problem
    in a complete and reliable way. It is released in the hope that FRC teams will try it out and get inspired to
    develop something much better for their own robot.

    General pipeline
    ----------------

    The basic idea of this module is the classic FIRST robotics vision pipeline: first, select a range of pixels in HSV
    color pixel space likely to include the object. Then, detect contours of all blobs in range. Then apply some tests
    on the shape of the detected blobs, their size, fill ratio (ratio of object area compared to its convex hull's
    area), etc. Finally, estimate the location and pose of the object in the world.

    In this module, we run up to 4 pipelines in parallel, using different settings for the range of HSV pixels
    considered:

    - Pipeline 0 uses the HSV values provided by user parameters;
    - Pipeline 1 broadens that fixed range a bit;
    - Pipelines 2-3 use a narrow and broader learned HSV window over time.

    Detections from all 4 pipelines are considered for overlap and quality (raggedness of their outlines), and only the
    cleanest of several overlapping detections is preserved. From those cleanest detections, pipelines 2-3 learn and
    adapt the HSV range for future video frames.

    Using this module
    -----------------

    Check out [this tutorial](http://jevois.org/tutorials/UserFirstVision.html).

    Detection and quality control steps
    -----------------------------------

    The following messages appear for each of the 4 pipelines, at the bottom of the demo video, to help users figure out
    why their object may not be detected:
    
    - T0 to T3: thread (pipeline) number
    - H=..., S=..., V=...: HSV range considered by that thread
    - N=...: number of raw blobs detected in that range
    - Because N blobs are considered in each thread from this point on, information about only the one that progressed
      the farthest through a series of tests is shown. One letter is added each time a test is passed:
      + H: the convex hull of the blob is quadrilateral (4 vertices)
      + A: hull area is within range specified by parameter \p hullarea
      + F: object to hull fill ratio is below the limit set by parameter \p hullfill (i.e., object is not a solid,
        filled quadrilateral shape)
      + S: the object has 8 vertices after shape smoothing to eliminate small shape defects (a U shape is
        indeed expected to have 8 vertices).
      + E: The shape discrepency between the original shape and the smoothed shape is acceptable per parameter 
        \p ethresh, i.e., the original contour did not have a lot of defects.
      + M: the shape is not too close to the borders of the image, per parameter \p margin, i.e., it is unlikely to 
        be truncated as the object partially exits the camera's field of view.
      + V: Vectors describing the shape as it related to its convex hull are non-zero, i.e., the centroid of the shape
        is not exactly coincident with the centroid of its convex hull, as we would expect for a U shape.
      + U: the shape is roughly upright; upside-down U shapes are rejected as likely spurious.
      + OK: this thread detected at least one shape that passed all the tests.

    The black and white picture at right shows the pixels in HSV range for the thread determined by parameter \p
    showthread (with value 0 by default).

    Serial Messages
    ---------------
 
    This module can send standardized serial messages as described in \ref UserSerialStyle. One message is issued on
    every video frame for each detected and good object. The \p id field in the messages simply is \b FIRST for all
    messages.

    When \p dopose is turned on, 3D messages will be sent, otherwise 2D messages.

    2D messages when \p dopose is off:

    - Serial message type: \b 2D
    - `id`: always `FIRST`
    - `x`, `y`, or vertices: standardized 2D coordinates of object center or corners
    - `w`, `h`: standardized marker size
    - `extra`: none (empty string)

    3D messages when \p dopose is on:

    - Serial message type: \b 3D
    - `id`: always `FIRST`
    - `x`, `y`, `z`, or vertices: 3D coordinates in millimeters of object center, or corners
    - `w`, `h`, `d`: object size in millimeters, a depth of 1mm is always used
    - `extra`: none (empty string)

    NOTE: 3D pose estimation from low-resolution 176x144 images at 120fps can be quite noisy. Make sure you tune your
    HSV ranges very well if you want to operate at 120fps (see below). To operate more reliably at very low resolutions,
    one may want to improve this module by adding subpixel shape refinement and tracking across frames.

    See \ref UserSerialStyle for more on standardized serial messages, and \ref coordhelpers for more info on
    standardized coordinates.

    Trying it out
    -------------

    The default parameter settings (which are set in \b script.cfg explained below) attempt to detect yellow-green
    objects. Present an object to the JeVois camera and see whether it is detected. When detected and good
    enough according to a number of quality control tests, the outline of the object is drawn.

    For further use of this module, you may want to check out the following tutorials:

    - [Using the sample FIRST Robotics vision module](http://jevois.org/tutorials/UserFirstVision.html)
    - [Tuning the color-based object tracker using a python graphical
      interface](http://jevois.org/tutorials/UserColorTracking.html)
    - [Making a motorized pan-tilt head for JeVois and tracking
      objects](http://jevois.org/tutorials/UserPanTilt.html)
    - \ref ArduinoTutorial

    Tuning
    ------

    You need to provide the exact width and height of your physical shape to parameter \p objsize for this module to
    work. It will look for a shape of that physical size (though at any distance and orientation from the camera). Be
    sure you edit \b script.cfg and set the parameter \p objsize in there to the true measured physical size of your
    shape.

    You should adjust parameters \p hcue, \p scue, and \p vcue to isolate the range of Hue, Saturation, and Value
    (respectively) that correspond to the objects you want to detect. Note that there is a \b script.cfg file in this
    module's directory that provides a range tuned to a light yellow-green object, as shown in the demo screenshot.

    Tuning the parameters is best done interactively by connecting to your JeVois camera while it is looking at some
    object of the desired color. Once you have achieved a tuning, you may want to set the hcue, scue, and vcue
    parameters in your \b script.cfg file for this module on the microSD card (see below).

    Typically, you would start by narrowing down on the hue, then the value, and finally the saturation. Make sure you
    also move your camera around and show it typical background clutter so check for false positives (detections of
    things which you are not interested, which can happen if your ranges are too wide).

    Config file
    -----------

    JeVois allows you to store parameter settings and commands in a file named \b script.cfg stored in the directory of
    a module. The file \b script.cfg may contain any sequence of commands as you would type them interactively in the
    JeVois command-line interface. For the \jvmod{FirstVision} module, a default script is provided that sets the camera
    to manual color, gain, and exposure mode (for more reliable color values), and other example parameter values.

    The \b script.cfg file for \jvmod{FirstVision} is stored on your microSD at
    <b>JEVOIS:/modules/JeVois/FirstVision/script.cfg</b> 

    @author Laurent Itti

    @videomapping YUYV 176 194 120.0 YUYV 176 144 120.0 JeVois FirstVision
    @videomapping YUYV 352 194 120.0 YUYV 176 144 120.0 JeVois FirstVision
    @videomapping YUYV 320 290 60.0 YUYV 320 240 60.0 JeVois FirstVision
    @videomapping YUYV 640 290 60.0 YUYV 320 240 60.0 JeVois FirstVision
    @videomapping NONE 0 0 0.0 YUYV 320 240 60.0 JeVois FirstVision
    @videomapping NONE 0 0 0.0 YUYV 176 144 120.0 JeVois FirstVision
    @email itti\@usc.edu
    @address University of Southern California, HNB-07A, 3641 Watt Way, Los Angeles, CA 90089-2520, USA
    @copyright Copyright (C) 2017 by Laurent Itti, iLab and the University of Southern California
    @mainurl http://jevois.org
    @supporturl http://jevois.org/doc
    @otherurl http://iLab.usc.edu
    @license GPL v3
    @distribution Unrestricted
    @restrictions None
    \ingroup modules */
class FirstVision : public jevois::StdModule,
		    public jevois::Parameter<hcue, scue, vcue, maxnumobj, hullarea, hullfill, erodesize,
					     dilatesize, epsilon, debug, threads, showthread, ethresh,
					     dopose, camparams, iou, objsize, margin>
{
  protected:
    cv::Mat itsCamMatrix; //!< Our camera matrix
    cv::Mat itsDistCoeffs; //!< Our camera distortion coefficients
    bool itsCueChanged = true; //!< True when users change ranges

    void onParamChange(hcue const & param, unsigned char const & newval) { itsCueChanged = true; }
    void onParamChange(scue const & param, unsigned char const & newval) { itsCueChanged = true; }
    void onParamChange(vcue const & param, unsigned char const & newval) { itsCueChanged = true; }
    
    // ####################################################################################################
    //! Helper struct for an HSV range triplet, where each range is specified as a mean and sigma:
    /*! Note that sigma is used differently for H, S, and V, under the assumption that we want to track a bright target:
        For H, the range is [mean-sigma .. mean+sigma]. For S and V, the range is [mean-sigma .. 255]. See rmin() and
        rmax() for details. */
    struct hsvcue
    {
	//! Constructor
	hsvcue(unsigned char h, unsigned char s, unsigned char v) : muh(h), sih(30), mus(s), sis(20), muv(v), siv(20)
	{ fix(); }
	
	//! Constructor
	hsvcue(unsigned char h, unsigned char hsig, unsigned char s, unsigned char ssig,
	       unsigned char v, unsigned char vsig) : muh(h), sih(hsig), mus(s), sis(ssig), muv(v), siv(vsig)
	{ fix(); }
	
	//! Fix ranges so they don't go out of bounds
	void fix()
	{
	  muh = std::min(179.0F, std::max(1.0F, muh)); sih = std::max(1.0F, std::min(sih, 360.0F));
	  mus = std::min(254.0F, std::max(1.0F, mus)); sis = std::max(1.0F, std::min(sis, 512.0F));
	  muv = std::min(254.0F, std::max(1.0F, muv)); siv = std::max(1.0F, std::min(siv, 512.0F));
	}
	
	//! Get minimum triplet for use by cv::inRange()
	cv::Scalar rmin() const
	{ return cv::Scalar(std::max(0.0F, muh - sih), std::max(0.0F, mus - sis), std::max(0.0F, muv - siv)); }

	//! Get maximum triplet for use by cv::inRange()
	cv::Scalar rmax() const
	{ return cv::Scalar(std::min(179.0F, muh + sih), 255, 255); }
	
	float muh, sih; //!< Mean and sigma for H
	float mus, sis; //!< Mean and sigma for S
	float muv, siv; //!< Mean and sigma for V
    };
    
    std::vector<hsvcue> itsHSV;

    // ####################################################################################################
    //! Helper struct for a detected object
    struct detection
    {
	std::vector<cv::Point> contour; //!< The full detailed contour
	std::vector<cv::Point> approx;  //!< Smoothed approximation of the contour
	std::vector<cv::Point> hull;    //!< Convex hull of the contour
	size_t threadnum;               //!< Thread number that detected this object
	float serr;                     //!< Shape error score (higher for rougher contours with defects)
    };

    //! Our detections, combined across all threads
    std::vector<detection> itsDetections;
    std::mutex itsDetMtx;

    //! Kalman filters to learn and adapt HSV windows over time
    std::shared_ptr<Kalman1D> itsKalH, itsKalS, itsKalV;

    //! Erosion and dilation kernels shared across all detect threads
    cv::Mat itsErodeElement, itsDilateElement;
    
    // ####################################################################################################
    //! ParallelLoopBody class for the parallelization of the single markers pose estimation
    /*! Derived from opencv_contrib ArUco module, it's just a simple solvePnP inside. */
    class SinglePoseEstimationParallel : public cv::ParallelLoopBody
    {
      public:
	SinglePoseEstimationParallel(cv::Mat & _objPoints, cv::InputArrayOfArrays _corners,
				     cv::InputArray _cameraMatrix, cv::InputArray _distCoeffs,
				     cv::Mat & _rvecs, cv::Mat & _tvecs) :
	    objPoints(_objPoints), corners(_corners), cameraMatrix(_cameraMatrix),
	    distCoeffs(_distCoeffs), rvecs(_rvecs), tvecs(_tvecs)
	{ }
	
	void operator()(cv::Range const & range) const
	{
	  int const begin = range.start;
	  int const end = range.end;
	  
	  for (int i = begin; i < end; ++i)
	    cv::solvePnP(objPoints, corners.getMat(i), cameraMatrix, distCoeffs,
			 rvecs.at<cv::Vec3d>(i), tvecs.at<cv::Vec3d>(i));
	}
	
      private:
	cv::Mat & objPoints;
	cv::InputArrayOfArrays corners;
	cv::InputArray cameraMatrix, distCoeffs;
	cv::Mat & rvecs, tvecs;
    };

    // ####################################################################################################
    // ####################################################################################################
    // ####################################################################################################

  public:
    // ####################################################################################################
    //! Constructor
    FirstVision(std::string const & instance) : jevois::StdModule(instance)
    {
      itsKalH = addSubComponent<Kalman1D>("kalH");
      itsKalS = addSubComponent<Kalman1D>("kalS");
      itsKalV = addSubComponent<Kalman1D>("kalV");
    }
    
    // ####################################################################################################
    //! Virtual destructor for safe inheritance
    virtual ~FirstVision() { }

    // ####################################################################################################
    //! Estimate 6D pose of detected objects, if dopose parameter is true, otherwise just 2D corners
    /*! Inspired from the ArUco module of opencv_contrib
      The corners array is always filled, but rvecs and tvecs only are if dopose is true */
    void estimatePose(std::vector<std::vector<cv::Point2f> > & corners, cv::OutputArray _rvecs,
		      cv::OutputArray _tvecs)
    {
      auto const osiz = objsize::get();
      
      // Get a vector of all our corners so we can map them to 3D and draw them:
      corners.clear();
      for (detection const & d : itsDetections)
      {
	corners.push_back(std::vector<cv::Point2f>());
	std::vector<cv::Point2f> & v = corners.back();
	for (auto const & p : d.hull) v.push_back(cv::Point2f(p));
      }

      if (dopose::get())
      {
	// set coordinate system in the middle of the object, with Z pointing out
	cv::Mat objPoints(4, 1, CV_32FC3);
	objPoints.ptr< cv::Vec3f >(0)[0] = cv::Vec3f(-osiz.width * 0.5F, -osiz.height * 0.5F, 0);
	objPoints.ptr< cv::Vec3f >(0)[1] = cv::Vec3f(-osiz.width * 0.5F, osiz.height * 0.5F, 0);
	objPoints.ptr< cv::Vec3f >(0)[2] = cv::Vec3f(osiz.width * 0.5F, osiz.height * 0.5F, 0);
	objPoints.ptr< cv::Vec3f >(0)[3] = cv::Vec3f(osiz.width * 0.5F, -osiz.height * 0.5F, 0);
	
	int nobj = (int)corners.size();
	_rvecs.create(nobj, 1, CV_64FC3); _tvecs.create(nobj, 1, CV_64FC3);
	cv::Mat rvecs = _rvecs.getMat(), tvecs = _tvecs.getMat();
	cv::parallel_for_(cv::Range(0, nobj), SinglePoseEstimationParallel(objPoints, corners, itsCamMatrix,
									   itsDistCoeffs, rvecs, tvecs));
      }
    }

    // ####################################################################################################
    //! Load camera calibration parameters
    void loadCameraCalibration(unsigned int w, unsigned int h)
    {
      camparams::freeze();
      
      std::string const cpf = std::string(JEVOIS_SHARE_PATH) + "/camera/" + camparams::get() +
	std::to_string(w) + 'x' + std::to_string(h) + ".yaml";
    
      cv::FileStorage fs(cpf, cv::FileStorage::READ);
      if (fs.isOpened())
      {
	fs["camera_matrix"] >> itsCamMatrix;
	fs["distortion_coefficients"] >> itsDistCoeffs;
	LINFO("Loaded camera calibration from " << cpf);
      }
      else LFATAL("Failed to read camera parameters from file [" << cpf << "]");
    }
    
    // ####################################################################################################
    //! HSV object detector, we run several of those in parallel with different hsvcue settings
    void detect(cv::Mat const & imghsv, size_t tnum, int dispx = 3, int dispy = 242, jevois::RawImage *outimg = nullptr)
    {
      // Threshold the HSV image to only keep pixels within the desired HSV range:
      cv::Mat imgth;
      hsvcue const & hsv = itsHSV[tnum]; cv::Scalar const rmin = hsv.rmin(), rmax = hsv.rmax();
      cv::inRange(imghsv, rmin, rmax, imgth);
      std::string str = jevois::sformat("T%zu: H=%03d-%03d S=%03d-%03d V=%03d-%03d ", tnum, int(rmin.val[0]),
					int(rmax.val[0]), int(rmin.val[1]), int(rmax.val[1]),
					int(rmin.val[2]), int(rmax.val[2]));
					
      // Apply morphological operations to cleanup the image noise:
      if (itsErodeElement.empty() == false) cv::erode(imgth, imgth, itsErodeElement);
      if (itsDilateElement.empty() == false) cv::dilate(imgth, imgth, itsDilateElement);

      // Detect objects by finding contours:
      std::vector<std::vector<cv::Point> > contours; std::vector<cv::Vec4i> hierarchy;
      cv::findContours(imgth, contours, hierarchy, cv::RETR_CCOMP, cv::CHAIN_APPROX_SIMPLE);
      str += jevois::sformat("N=%03d ", hierarchy.size());
      
      double const epsi = epsilon::get();
      int const m = margin::get();

      // Identify the "good" objects:
      std::string str2, beststr2;
      if (hierarchy.size() > 0 && hierarchy.size() <= maxnumobj::get())
      {
        for (int index = 0; index >= 0; index = hierarchy[index][0])
        {
	  // Keep track of our best detection so far:
	  if (str2.length() > beststr2.length()) beststr2 = str2;
	  str2.clear();

	  // Let's examine this contour:
	  std::vector<cv::Point> const & c = contours[index];
	  detection d;
	  
	  // Compute contour area:
          double const area = cv::contourArea(c, false);

	  // Compute convex hull:
	  std::vector<cv::Point> rawhull;
	  cv::convexHull(c, rawhull, true);
	  double const rawhullperi = cv::arcLength(rawhull, true);
	  cv::approxPolyDP(rawhull, d.hull, epsi * rawhullperi * 3.0, true);

	  // Is it the right shape?
	  if (d.hull.size() != 4) continue;  // 4 vertices for the rectangular convex outline (shows as a trapezoid)
	  str2 += "H"; // Hull is quadrilateral
	  
          double const huarea = cv::contourArea(d.hull, false);
	  if ( ! hullarea::get().contains(int(huarea + 0.4999))) continue;
	  str2 += "A"; // Hull area ok
	  
	  int const hufill = int(area / huarea * 100.0 + 0.4999);
	  if (hufill > hullfill::get()) continue;
	  str2 += "F"; // Fill is ok
	  
	  // Check object shape:
	  double const peri = cv::arcLength(c, true);
	  cv::approxPolyDP(c, d.approx, epsi * peri, true);
	  if (d.approx.size() < 7 || d.approx.size() > 9) continue;  // 8 vertices for a U shape
	  str2 += "S"; // Shape is ok

	  // Compute contour serr:
	  d.serr = 100.0 * cv::matchShapes(c, d.approx, cv::CONTOURS_MATCH_I1, 0.0);
	  if (d.serr > ethresh::get()) continue;
	  str2 += "E"; // Shape error is ok
	  
	  // Reject the shape if any of its vertices gets within the margin of the image bounds. This is to avoid
	  // getting grossly incorrect 6D pose estimates as the shape starts getting truncated as it partially exits the
	  // camera field of view:
	  bool reject = false;
	  for (int i = 0; i < c.size(); ++i)
	    if (c[i].x < m || c[i].x >= imghsv.cols - m || c[i].y < m || c[i].y >= imghsv.rows - m)
	    { reject = true; break; }
	  if (reject) continue;
	  str2 += "M"; // Margin ok
	  
	  // Re-order the 4 points in the hull if needed: In the pose estimation code, we will assume vertices ordered
	  // as follows:
	  //
	  //    0|        |3
	  //     |        |
	  //     |        |
	  //    1----------2

	  // v10+v23 should be pointing outward the U more than v03+v12 is:
	  std::complex<float> v10p23(float(d.hull[0].x - d.hull[1].x + d.hull[3].x - d.hull[2].x),
				     float(d.hull[0].y - d.hull[1].y + d.hull[3].y - d.hull[2].y));
	  float const len10p23 = std::abs(v10p23);
	  std::complex<float> v03p12(float(d.hull[3].x - d.hull[0].x + d.hull[2].x - d.hull[1].x),
				     float(d.hull[3].y - d.hull[0].y + d.hull[2].y - d.hull[1].y));
	  float const len03p12 = std::abs(v03p12);

	  // Vector from centroid of U shape to centroid of its hull should also point outward of the U:
	  cv::Moments const momC = cv::moments(c);
	  cv::Moments const momH = cv::moments(d.hull);
	  std::complex<float> vCH(momH.m10 / momH.m00 - momC.m10 / momC.m00, momH.m01 / momH.m00 - momC.m01 / momC.m00);
	  float const lenCH = std::abs(vCH);

	  if (len10p23 < 0.1F || len03p12 < 0.1F || lenCH < 0.1F) continue;
	  str2 += "V"; // Shape vectors ok

	  float const good = (v10p23.real() * vCH.real() + v10p23.imag() * vCH.imag()) / (len10p23 * lenCH);
	  float const bad = (v03p12.real() * vCH.real() + v03p12.imag() * vCH.imag()) / (len03p12 * lenCH);

	  // We reject upside-down detections as those are likely to be spurious:
	  if (vCH.imag() >= -2.0F) continue;
	  str2 += "U"; // U shape is upright
	
	  // Fixup the ordering of the vertices if needed:
	  if (bad > good) { d.hull.insert(d.hull.begin(), d.hull.back()); d.hull.pop_back(); }

	  // This detection is a keeper:
	  str2 += " OK";
	  d.contour = c;
	  std::lock_guard<std::mutex> _(itsDetMtx);
	  itsDetections.push_back(d);
	}
	if (str2.length() > beststr2.length()) beststr2 = str2;
      }

      // Display any results requested by the users:
      if (outimg && outimg->valid())
      {
	if (tnum == showthread::get() && outimg->width == 2 * imgth.cols)
	  jevois::rawimage::pasteGreyToYUYV(imgth, *outimg, imgth.cols, 0);
	jevois::rawimage::writeText(*outimg, str + beststr2, dispx, dispy + 12*tnum, jevois::yuyv::White);
      }
    }
    
    // ####################################################################################################
    //! Initialize (e.g., if user changes cue params) or update our HSV detection ranges
    void updateHSV(size_t nthreads)
    {
      float const spread = 0.2F;
      
      if (itsHSV.empty() || itsCueChanged)
      {
	// Initialize or reset because of user parameter change:
	itsHSV.clear(); itsCueChanged = false;
	for (size_t i = 0; i < nthreads; ++i)
	{
	  hsvcue cue(hcue::get(), scue::get(), vcue::get());
	  cue.sih *= (1.0F + spread * i); cue.sis *= (1.0F + spread * i); cue.siv *= (1.0F + spread * i); 
	  cue.fix();
	  itsHSV.push_back(cue);
	}
	if (nthreads > 2)
	{
	  itsKalH->set(hcue::get()); itsKalH->get();
	  itsKalS->set(scue::get()); itsKalS->get();
	  itsKalV->set(vcue::get()); itsKalV->get();
	}
      }
      else
      {
	// Kalman update:
	if (nthreads > 2)
	{
	  itsHSV[2].muh = itsKalH->get();
	  itsHSV[2].mus = itsKalS->get();
	  itsHSV[2].muv = itsKalV->get();
	  itsHSV[2].fix();
	  for (size_t i = 3; i < itsHSV.size(); ++i)
	  {
	    itsHSV[i] = itsHSV[2];
	    itsHSV[i].sih *= (1.0F + spread * i);
	    itsHSV[i].sis *= (1.0F + spread * i);
	    itsHSV[i].siv *= (1.0F + spread * i); 
	    itsHSV[i].fix();
	  }
	}
      }
    }

    // ####################################################################################################
    //! Clean up the detections by eliminating duplicates:
    void cleanupDetections()
    {
      bool keepgoing = true;
      double const iouth = iou::get();

      while (keepgoing)
      {
	// We will stop if we do not eliminate any more objects:
	keepgoing = false; int delidx = -1;

	// Loop over all pairs of objects:
	size_t const siz = itsDetections.size();
	for (size_t i = 0; i < siz; ++i)
	{
	  for (size_t j = 0; j < i; ++j)
	  {
	    std::vector<cv::Point> pts = itsDetections[i].hull;
	    for (cv::Point const & p : itsDetections[j].hull) pts.push_back(p);
	    std::vector<cv::Point> hull;
	    cv::convexHull(pts, hull); // FIXME should do a true union! this is just an approximation to it
	    double uarea = cv::contourArea(hull);
	    double iarea = cv::contourArea(itsDetections[i].hull) + cv::contourArea(itsDetections[j].hull) - uarea;
	    
	    // note: object detection code guarantees non-zero area:
	    double const inoun = iarea / uarea;
	    if (inoun >= iouth)
	    {
	      if (itsDetections[i].serr > itsDetections[j].serr) delidx = j; else delidx = i;
	      break;
	    }
	  }
	  if (delidx != -1) break;
	}
	if (delidx != -1) { itsDetections.erase(itsDetections.begin() + delidx); keepgoing = true; }
      }
    }
    
    // ####################################################################################################
    //! Learn and update our HSV ranges
    void learnHSV(size_t nthreads, cv::Mat const & imgbgr, jevois::RawImage *outimg = nullptr)
    {
      int const w = imgbgr.cols, h = imgbgr.rows;

      // Compute the median filtered BGR image in a thread:
      cv::Mat medimgbgr;
      auto median_fut = std::async(std::launch::async, [&](){ cv::medianBlur(imgbgr, medimgbgr, 3); } );
      
      // Get all the cleaned-up contours:
      std::vector<std::vector<cv::Point> > contours;
      for (detection const & d : itsDetections) contours.push_back(d.contour);
	
      // If desired, draw all contours:
      std::future<void> drawc_fut;
      if (debug::get() && outimg && outimg->valid())
	drawc_fut = std::async(std::launch::async, [&]() {
	    // We reinterpret the top portion of our YUYV output image as an opencv 8UC2 image:
	    cv::Mat outuc2(outimg->height, outimg->width, CV_8UC2, outimg->pixelsw<unsigned char>());
	    cv::drawContours(outuc2, contours, -1, jevois::yuyv::LightPink, 2);
	  } );

      // Draw all the filled contours into a binary mask image:
      cv::Mat mask(h, w, CV_8UC1, (unsigned char)0);
      cv::drawContours(mask, contours, -1, 255, -1); // last -1 is for filled
      
      // Wait until median filter is done:
      median_fut.get();
      
      // Compute mean and std BGR values inside objects:
      cv::Mat mean, std;
      cv::meanStdDev(medimgbgr, mean, std, mask);
      
      // Convert to HSV:
      cv::Mat bgrmean(2, 1, CV_8UC3); bgrmean.at<cv::Vec3b>(0, 0) = mean; bgrmean.at<cv::Vec3b>(1, 0) = std;
      cv::Mat hsvmean; cv::cvtColor(bgrmean, hsvmean, cv::COLOR_BGR2HSV);

      cv::Vec3b hsv = hsvmean.at<cv::Vec3b>(0, 0);
      int H = hsv.val[0], S = hsv.val[1], V = hsv.val[2];

      cv::Vec3b sighsv = hsvmean.at<cv::Vec3b>(1, 0);
      int sH = sighsv.val[0], sS = sighsv.val[1], sV = sighsv.val[2];

      // Set the new measurements:
      itsKalH->set(H); itsKalS->set(S); itsKalV->set(V);

      if (nthreads > 2)
      {
	float const eta = 0.4F;
	itsHSV[2].sih = (1.0F - eta) * itsHSV[2].sih + eta * sH;
	itsHSV[2].sis = (1.0F - eta) * itsHSV[2].sis + eta * sS;
	itsHSV[2].siv = (1.0F - eta) * itsHSV[2].siv + eta * sV;
	itsHSV[2].fix();
      }

      // note: drawc_fut may block us here until it is complete.
    }

    // ####################################################################################################
    //! Send serial messages about each detection:
    void sendAllSerial(int w, int h, std::vector<std::vector<cv::Point2f> > const & corners,
		       std::vector<cv::Vec3d> const & rvecs, std::vector<cv::Vec3d> const & tvecs)
    {
      if (rvecs.empty() == false)
      {
	// If we have rvecs and tvecs, we are doing 3D pose estimation, so send a 3D message:
	auto const osiz = objsize::get();
	for (size_t i = 0; i < corners.size(); ++i)
	{
	  std::vector<cv::Point2f> const & curr = corners[i];
	  cv::Vec3d const & rv = rvecs[i];
	  cv::Vec3d const & tv = tvecs[i];
	  
	  // Compute quaternion:
	  float theta = std::sqrt(rv[0] * rv[0] + rv[1] * rv[1] + rv[2] * rv[2]);
	  Eigen::Vector3f axis(rv[0], rv[1], rv[2]);
	  Eigen::Quaternion<float> q(Eigen::AngleAxis<float>(theta, axis));
	  
	  sendSerialStd3D(tv[0], tv[1], tv[2],             // position
			  osiz.width, osiz.height, 1.0F,   // size
			  q.w(), q.x(), q.y(), q.z(),      // pose
			  "FIRST");                        // FIRST robotics shape
	}
      }
      else
      {
	// Send one 2D message per object:
	for (size_t i = 0; i < corners.size(); ++i)
	  sendSerialContour2D(w, h, corners[i], "FIRST");
      }
    }

    // ####################################################################################################
    //! Update the morphology structuring elements if needed
    void updateStructuringElements()
    {
      int e = erodesize::get();
      if (e != itsErodeElement.cols)
      {
	if (e) itsErodeElement = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(e, e));
	else itsErodeElement.release();
      }

      int d = dilatesize::get();
      if (d != itsDilateElement.cols)
      {
	if (d) itsDilateElement = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(d, d));
	else itsDilateElement.release();
      }
    }
    
    // ####################################################################################################
    //! Processing function, no USB video output
    virtual void process(jevois::InputFrame && inframe) override
    {
      static jevois::Timer timer("processing");

      // Wait for next available camera image. Any resolution ok:
      jevois::RawImage inimg = inframe.get(); unsigned int const w = inimg.width, h = inimg.height;

      timer.start();

      // Load camera calibration if needed:
      if (itsCamMatrix.empty()) loadCameraCalibration(w, h);

      // Convert input image to BGR24, then to HSV:
      cv::Mat imgbgr = jevois::rawimage::convertToCvBGR(inimg);
      cv::Mat imghsv; cv::cvtColor(imgbgr, imghsv, cv::COLOR_BGR2HSV);
      size_t const nthreads = threads::get();
      
      // Make sure our HSV range parameters are up to date:
      updateHSV(nthreads);
      
      // Clear any old detections and get ready to parallelize the detection work:
      itsDetections.clear();
      updateStructuringElements();
      
      // Launch our workers: run nthreads-1 new threads, and last worker in our current thread:
      std::vector<std::future<void> > dfut;
      for (size_t i = 0; i < nthreads - 1; ++i)
	dfut.push_back(std::async(std::launch::async, [&](size_t tn) { detect(imghsv, tn, 3, h+2); }, i));
      detect(imghsv, nthreads - 1, 3, h+2);

      // Wait for all threads to complete:
      for (auto & f : dfut) try { f.get(); } catch (...) { jevois::warnAndIgnoreException(); }

      // Let camera know we are done processing the input image:
      inframe.done();

      // Clean up the detections by eliminating duplicates:
      cleanupDetections();

      // Learn the object's HSV value over time:
      auto learn_fut = std::async(std::launch::async, [&]() { learnHSV(nthreads, imgbgr); });

      // Map to 6D (inverse perspective):
      std::vector<std::vector<cv::Point2f> > corners; std::vector<cv::Vec3d> rvecs, tvecs;
      estimatePose(corners, rvecs, tvecs);

      // Send all serial messages:
      sendAllSerial(w, h, corners, rvecs, tvecs);
      
      // Wait for all threads:
      try { learn_fut.get(); } catch (...) { jevois::warnAndIgnoreException(); }
      
      // Show processing fps:
      timer.stop();
    }

    // ####################################################################################################
    //! Processing function, with USB video output
    virtual void process(jevois::InputFrame && inframe, jevois::OutputFrame && outframe) override
    {
      static jevois::Timer timer("processing");

      // Wait for next available camera image. Any resolution ok, but require YUYV since we assume it for drawings:
      jevois::RawImage inimg = inframe.get(); unsigned int const w = inimg.width, h = inimg.height;
      inimg.require("input", w, h, V4L2_PIX_FMT_YUYV);

      timer.start();

      // Load camera calibration if needed:
      if (itsCamMatrix.empty()) loadCameraCalibration(w, h);

      // While we process it, start a thread to wait for output frame and paste the input image into it:
      jevois::RawImage outimg; // main thread should not use outimg until paste thread is complete
      auto paste_fut = std::async(std::launch::async, [&]() {
          outimg = outframe.get();
          outimg.require("output", outimg.width, h + 50, inimg.fmt);
	  if (outimg.width != w && outimg.width != w * 2) LFATAL("Output image width should be 1x or 2x input width");
          jevois::rawimage::paste(inimg, outimg, 0, 0);
          jevois::rawimage::writeText(outimg, "JeVois FIRST Vision", 3, 3, jevois::yuyv::White);
          jevois::rawimage::drawFilledRect(outimg, 0, h, outimg.width, outimg.height-h, jevois::yuyv::Black);
        });

      // Convert input image to BGR24, then to HSV:
      cv::Mat imgbgr = jevois::rawimage::convertToCvBGR(inimg);
      cv::Mat imghsv; cv::cvtColor(imgbgr, imghsv, cv::COLOR_BGR2HSV);
      size_t const nthreads = threads::get();
      
      // Make sure our HSV range parameters are up to date:
      updateHSV(nthreads);
      
      // Clear any old detections and get ready to parallelize the detection work:
      itsDetections.clear();
      updateStructuringElements();
      
      // Launch our workers: run nthreads-1 new threads, and last worker in our current thread:
      std::vector<std::future<void> > dfut;
      for (size_t i = 0; i < nthreads - 1; ++i)
	dfut.push_back(std::async(std::launch::async, [&](size_t tn) { detect(imghsv, tn, 3, h+2, &outimg); }, i));
      detect(imghsv, nthreads - 1, 3, h+2, &outimg);

      // Wait for all threads to complete:
      for (auto & f : dfut) try { f.get(); } catch (...) { jevois::warnAndIgnoreException(); }

      // Wait for paste to finish up:
      paste_fut.get();
      
      // Let camera know we are done processing the input image:
      inframe.done();

      // Clean up the detections by eliminating duplicates:
      cleanupDetections();

      // Learn the object's HSV value over time:
      auto learn_fut = std::async(std::launch::async, [&]() { learnHSV(nthreads, imgbgr, &outimg); });

      // Map to 6D (inverse perspective):
      std::vector<std::vector<cv::Point2f> > corners; std::vector<cv::Vec3d> rvecs, tvecs;
      estimatePose(corners, rvecs, tvecs);

      // Send all serial messages:
      sendAllSerial(w, h, corners, rvecs, tvecs);
      
      // Draw all detections in 3D:
      drawDetections(outimg, corners, rvecs, tvecs);

      // Show number of detected objects:
      jevois::rawimage::writeText(outimg, "Detected " + std::to_string(itsDetections.size()) + " objects.",
				  w + 3, 3, jevois::yuyv::White);

      // Wait for all threads:
      try { learn_fut.get(); } catch (...) { jevois::warnAndIgnoreException(); }
      
      // Show processing fps:
      std::string const & fpscpu = timer.stop();
      jevois::rawimage::writeText(outimg, fpscpu, 3, h - 13, jevois::yuyv::White);
      
      // Send the output image with our processing results to the host over USB:
      outframe.send();
    }

    // ####################################################################################################
    void drawDetections(jevois::RawImage & outimg, std::vector<std::vector<cv::Point2f> > corners,
			std::vector<cv::Vec3d> const & rvecs, std::vector<cv::Vec3d> const & tvecs)
    {
      auto const osiz = objsize::get(); float const w = osiz.width, h = osiz.height;
      int nobj = int(corners.size());

      // This code is like drawDetectedMarkers() in cv::aruco, but for YUYV output image:
      if (rvecs.empty())
      {
	// We are not doing 3D pose estimation. Just draw object outlines in 2D:
	for (int i = 0; i < nobj; ++i)
	{
	  std::vector<cv::Point2f> const & obj = corners[i];
        
	  // draw marker sides:
	  for (int j = 0; j < 4; ++j)
	  {
	    cv::Point2f const & p0 = obj[j];
	    cv::Point2f const & p1 = obj[ (j+1) % 4 ];
	    jevois::rawimage::drawLine(outimg, int(p0.x + 0.5F), int(p0.y + 0.5F),
				       int(p1.x + 0.5F), int(p1.y + 0.5F), 1, jevois::yuyv::LightPink);
	    //jevois::rawimage::writeText(outimg, std::to_string(j),
	    //			      int(p0.x + 0.5F), int(p0.y + 0.5F), jevois::yuyv::White);
	  }
	}
      }
      else
      {
	// Show trihedron and parallelepiped centered on object:
	float const hw = w * 0.5F, hh = h * 0.5F, dd = -0.5F * std::max(w, h);

	for (int i = 0; i < nobj; ++i)
	{
	  // Project axis points:
	  std::vector<cv::Point3f> axisPoints;
	  axisPoints.push_back(cv::Point3f(0.0F, 0.0F, 0.0F));
	  axisPoints.push_back(cv::Point3f(hw, 0.0F, 0.0F));
	  axisPoints.push_back(cv::Point3f(0.0F, hh, 0.0F));
	  axisPoints.push_back(cv::Point3f(0.0F, 0.0F, dd));
	  
	  std::vector<cv::Point2f> imagePoints;
	  cv::projectPoints(axisPoints, rvecs[i], tvecs[i], itsCamMatrix, itsDistCoeffs, imagePoints);
	  
	  // Draw axis lines:
	  jevois::rawimage::drawLine(outimg, int(imagePoints[0].x + 0.5F), int(imagePoints[0].y + 0.5F),
				     int(imagePoints[1].x + 0.5F), int(imagePoints[1].y + 0.5F),
				     2, jevois::yuyv::MedPurple);
	  jevois::rawimage::drawLine(outimg, int(imagePoints[0].x + 0.5F), int(imagePoints[0].y + 0.5F),
				     int(imagePoints[2].x + 0.5F), int(imagePoints[2].y + 0.5F),
				     2, jevois::yuyv::MedGreen);
	  jevois::rawimage::drawLine(outimg, int(imagePoints[0].x + 0.5F), int(imagePoints[0].y + 0.5F),
				     int(imagePoints[3].x + 0.5F), int(imagePoints[3].y + 0.5F),
				     2, jevois::yuyv::MedGrey);
	  
	  // Also draw a parallelepiped:
	  std::vector<cv::Point3f> cubePoints;
	  cubePoints.push_back(cv::Point3f(-hw, -hh, 0.0F));
	  cubePoints.push_back(cv::Point3f(hw, -hh, 0.0F));
	  cubePoints.push_back(cv::Point3f(hw, hh, 0.0F));
	  cubePoints.push_back(cv::Point3f(-hw, hh, 0.0F));
	  cubePoints.push_back(cv::Point3f(-hw, -hh, dd));
	  cubePoints.push_back(cv::Point3f(hw, -hh, dd));
	  cubePoints.push_back(cv::Point3f(hw, hh, dd));
	  cubePoints.push_back(cv::Point3f(-hw, hh, dd));
	  
	  std::vector<cv::Point2f> cuf;
	  cv::projectPoints(cubePoints, rvecs[i], tvecs[i], itsCamMatrix, itsDistCoeffs, cuf);
	
	  // Round all the coordinates:
	  std::vector<cv::Point> cu;
	  for (auto const & p : cuf) cu.push_back(cv::Point(int(p.x + 0.5F), int(p.y + 0.5F)));
	  
	  // Draw parallelepiped lines:
	  jevois::rawimage::drawLine(outimg, cu[0].x, cu[0].y, cu[1].x, cu[1].y, 1, jevois::yuyv::LightGreen);
	  jevois::rawimage::drawLine(outimg, cu[1].x, cu[1].y, cu[2].x, cu[2].y, 1, jevois::yuyv::LightGreen);
	  jevois::rawimage::drawLine(outimg, cu[2].x, cu[2].y, cu[3].x, cu[3].y, 1, jevois::yuyv::LightGreen);
	  jevois::rawimage::drawLine(outimg, cu[3].x, cu[3].y, cu[0].x, cu[0].y, 1, jevois::yuyv::LightGreen);
	  jevois::rawimage::drawLine(outimg, cu[4].x, cu[4].y, cu[5].x, cu[5].y, 1, jevois::yuyv::LightGreen);
	  jevois::rawimage::drawLine(outimg, cu[5].x, cu[5].y, cu[6].x, cu[6].y, 1, jevois::yuyv::LightGreen);
	  jevois::rawimage::drawLine(outimg, cu[6].x, cu[6].y, cu[7].x, cu[7].y, 1, jevois::yuyv::LightGreen);
	  jevois::rawimage::drawLine(outimg, cu[7].x, cu[7].y, cu[4].x, cu[4].y, 1, jevois::yuyv::LightGreen);
	  jevois::rawimage::drawLine(outimg, cu[0].x, cu[0].y, cu[4].x, cu[4].y, 1, jevois::yuyv::LightGreen);
	  jevois::rawimage::drawLine(outimg, cu[1].x, cu[1].y, cu[5].x, cu[5].y, 1, jevois::yuyv::LightGreen);
	  jevois::rawimage::drawLine(outimg, cu[2].x, cu[2].y, cu[6].x, cu[6].y, 1, jevois::yuyv::LightGreen);
	  jevois::rawimage::drawLine(outimg, cu[3].x, cu[3].y, cu[7].x, cu[7].y, 1, jevois::yuyv::LightGreen);
	}
      }
    }
};

// Allow the module to be loaded as a shared object (.so) file:
JEVOIS_REGISTER_MODULE(FirstVision);
