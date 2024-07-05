// ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//
// JeVois Smart Embedded Machine Vision Toolkit - Copyright (C) 2024 by Laurent Itti, the University of Southern
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

// icon by flaticon

#include <jevois/Core/Module.H>
#include <jevois/Core/Engine.H>
#include <jevois/Debug/Timer.H>
#include <jevois/Image/RawImageOps.H>
#include <jevoisbase/Components/ArUco/ArUco.H> // for dictionary enum
#include <opencv2/core.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/objdetect/charuco_detector.hpp>
#include <chrono>

static jevois::ParameterCategory const ParamCateg("Camera Calibration Options");

//! Enum for parameter \relates CameraCalibration
JEVOIS_DEFINE_ENUM_CLASS(Pattern, (ChessBoard) (ChArUcoBoard) (CirclesGrid) (AsymmetricCirclesGrid) )

//! Parameter \relates CameraCalibration
JEVOIS_DECLARE_PARAMETER_WITH_CALLBACK(pattern, Pattern, "Type of calibration board pattern to use",
                         Pattern::ChessBoard, Pattern_Values, ParamCateg);

//! Parameter \relates CameraCalibration
JEVOIS_DECLARE_PARAMETER_WITH_CALLBACK(dictionary, aruco::Dict, "ArUco dictionary to use",
                                       aruco::Dict::D4X4_50, aruco::Dict_Values, ParamCateg);

//! Parameter \relates CameraCalibration
JEVOIS_DECLARE_PARAMETER(squareSize, float, "Size of each tile (check) in user-chosen units (e.g., mm, inch, etc). The "
                         "unit used here is the one that will be used once calibrated to report 3D coordinates "
                         "of objects relative to the camera",
                         23.0f, ParamCateg);

//! Parameter \relates CameraCalibration
JEVOIS_DECLARE_PARAMETER(markerSize, float, "ChArUco marker size in user-chosen units (e.g., mm, inch, "
                         "etc). The unit used here is the one that will be used once calibrated to report 3D "
                         "coordinates of objects relative to the camera",
                         27.0f, ParamCateg);

//! Parameter \relates CameraCalibration
JEVOIS_DECLARE_PARAMETER_WITH_CALLBACK(boardSize, cv::Size, "Board size [width height] in number of horizontal and "
                                       "vertical tiles/disks. (Note: for asymmetric circle grid, count the number of "
                                       "disks on each row, then the number of rows). The product width * height "
                                       "should be the total number of tiles/disks on the grid.)",
                                       { 11, 7 }, ParamCateg);
  
//! Parameter \relates CameraCalibration
JEVOIS_DECLARE_PARAMETER(aspectRatio, float, "Fixed aspect ratio value to use when non-zero, or auto when 0.0",
                         0.0F, ParamCateg);

//! Parameter \relates CameraCalibration
JEVOIS_DECLARE_PARAMETER(zeroTangentDist, bool, "Assume zero tangential distortion coefficients P1, P2 and do "
                         "not try to optimize them, i.e., assume board is exactly planar",
                         true, ParamCateg);

//! Parameter \relates CameraCalibration
JEVOIS_DECLARE_PARAMETER(fixPrincipalPoint, bool, "Fix principal point at center, otherwise find its location",
                         true, ParamCateg);

//! Parameter \relates CameraCalibration
JEVOIS_DECLARE_PARAMETER(fishEye, bool, "Use fisheye model. Should be true if using a wide-angle lens that "
                         "produces significant barrel distortion, otherwise false",
                         false, ParamCateg);

//! Parameter \relates CameraCalibration
JEVOIS_DECLARE_PARAMETER(winSize, unsigned char, "Half of search window size for sub-pixel corner refinement",
                         11, ParamCateg);

//! Parameter \relates CameraCalibration
JEVOIS_DECLARE_PARAMETER(fixK1, bool, "Fix (do not try to optimize) K1 radial distortion parameter",
                         false, ParamCateg);

//! Parameter \relates CameraCalibration
JEVOIS_DECLARE_PARAMETER(fixK2, bool, "Fix (do not try to optimize) K2 radial distortion parameter",
                         false, ParamCateg);

//! Parameter \relates CameraCalibration
JEVOIS_DECLARE_PARAMETER(fixK3, bool, "Fix (do not try to optimize) K3 radial distortion parameter",
                         false, ParamCateg);

//! Parameter \relates CameraCalibration
JEVOIS_DECLARE_PARAMETER(showUndistorted, bool, "Once calibrated, show undistorted image instead of original capture",
                         false, ParamCateg);

//! Parameter \relates CameraCalibration
JEVOIS_DECLARE_PARAMETER(grab, bool, "Grab one image, try to detect the calibration pattern, and add to list of "
                         "detected patterns so far. Click this for at least 5 images showing your calibration board "
                         "under various viewpoints. When enough boards have been successfully captured, click "
                         "'calibrate'",
                         false, ParamCateg);

//! Parameter \relates CameraCalibration
JEVOIS_DECLARE_PARAMETER(calibrate, bool, "Calibrate using all the grabbed boards so far. You first need to grab "
                         "enough good views of your calibration board before you can calibrate.",
                         false, ParamCateg);

//! Helper module to calibrate a given sensor+lens combo, which allows ArUco and other modules to do 3D pose estimation
/*! Just follow the on-screen prompts to calibrate your camera. The calibration results will be saved into
    /jevois[pro]/share/camera/ on microSD and will be automatically loaded when using a machine vision module that uses
    camera calibration, for example \jvmod{DemoArUco} or \jvmod{FirstVision}.

    The basic workflow is:

    - print a calibration board and affix is to a good planar surface (e.g., an acrylic board, aluminum board, etc). Or
      buy a board, for example from https://calib.io or similar.

    - decide which sensor, resolution, and lens you want to calibrate. The sensor should be detected at boot time (check
      \p engine::camerasens system parameter). The lens can be set manually through the \p engine::cameralens
      parameter. The default lens is called "standard". For the resolution, edit videomappings.cfg and add a mode for
      the CalibrateCamera module that will use the same resolution as the one you want to use later for machine
      vision. Several commented-out examples are already in videomappings.cfg, so likely you can just uncomment one of
      them. Then run the corresponding version of the CameraCalibration module.

    - set the parameters to match the board type, board width and height in number of tiles, tile size in the unit that
      you want to later use to estimate distance to detected objects (e.g., in millimeters, inches, etc), and possibly
      ChArUco marker size and dictionary.

    - point the camera towards the board so that the full board is in the camera view. Click 'grab'. Change the
      viewpoint and repeat at least 5 times. Use various 3D viewpoints.

    - click 'calibrate' and the calibration will be computed and saved to microSD for that sensor, resolution, and
      lens. It will be ready to be loaded by modules that want to use it.

    The default settings are for a 11x7 Chess Board that was created using the online generator at https://calib.io and
    which you can get from http://jevois.org/data/calib.io_checker_260x200_7x11_23.pdf -
    When you print the board, make sure you print at 100% scale. You should confirm that the checks in the printout are
    23mm x 23mm.

    An alternate chess board with fewer checks may be desirable at low resolutions, such as the 7x5 chess board at
    http://jevois.org/data/calib.io_checker_260x200_5x7_36.pdf - if you use it, set board size to '7 5', and square size
    to '36' to match it.
   
    If you are using a wide-angle fish-eye lens with a lot of distortion, turn on the \p fishEye parameter.

    You can also use a ChArUco board, though we have actually obtained worse reprojection errors with these boards. You
    can print the board at http://jevois.org/data/calib.io_charuco_260x200_5x7_36_27_DICT_4X4.pdf at 100% scale, and you
    should confirm that the checks in the printout are 36mm x 36mm, and the ArUco patterns within the white checks are
    27mm x 27mm.  If you use it, you need to set the board type, size, marker size, and square size parameters to match
    it. An alternate board with more ChArUcos is at
    http://jevois.org/data/calib.io_charuco_260x200_7x11_23_17_DICT_4X4.pdf


    @author Laurent Itti

    @displayname Calibrate Camera
    @videomapping YUYV 320 495 30.0 YUYV 320 240 30.0 JeVois DemoArUco
    @videomapping YUYV 640 975 20.0 YUYV 640 480 20.0 JeVois DemoArUco
    @email itti\@usc.edu
    @address University of Southern California, HNB-07A, 3641 Watt Way, Los Angeles, CA 90089-2520, USA
    @copyright Copyright (C) 2024 by Laurent Itti, iLab and the University of Southern California
    @mainurl http://jevois.org
    @supporturl http://jevois.org/doc
    @otherurl http://iLab.usc.edu
    @license GPL v3
    @distribution Unrestricted
    @restrictions None
    \ingroup modules */
class CalibrateCamera : public jevois::Module,
                        public jevois::Parameter<pattern, dictionary, squareSize, markerSize, boardSize,
                                                 aspectRatio, zeroTangentDist, fixPrincipalPoint, winSize,
                                                 fishEye, fixK1, fixK2, fixK3, showUndistorted, grab, calibrate>
{
  public:
    // ####################################################################################################
    //! Constructor
    // ####################################################################################################
    using jevois::Module::Module;
    
    // ####################################################################################################
    //! Virtual destructor for safe inheritance
    // ####################################################################################################
    virtual ~CalibrateCamera()
    { }

    // ####################################################################################################
    //! Process one captured image
    // ####################################################################################################
    void process_frame(cv::Mat const & view)
    {
      // Add one image, or run the whole calibration if we have enough images:
      if (grab::get())
      {
        grab::set(false);
        
        // Try to detect the board:
        std::vector<cv::Point2f> pointBuf;
        int chessBoardFlags = cv::CALIB_CB_ADAPTIVE_THRESH | cv::CALIB_CB_NORMALIZE_IMAGE;

        // fast check erroneously fails with high distortions like fisheye
        if (fishEye::get() == false) chessBoardFlags |= cv::CALIB_CB_FAST_CHECK;

        // Find feature points:
        cv::Size bs = boardSize::get();
        
        bool found = false;
        switch (pattern::get())
        {
        case Pattern::ChessBoard:
          found = cv::findChessboardCorners(view, cv::Size(bs.width - 1, bs.height - 1), pointBuf, chessBoardFlags);
          if (found)
          {
            // Improve the found corners' coordinate accuracy for chessboard:
            cv::Mat viewGray;
            cv::cvtColor(view, viewGray, cv::COLOR_BGR2GRAY);
            cv::cornerSubPix(viewGray, pointBuf, cv::Size(winSize::get(), winSize::get()), cv::Size(-1,-1),
                             cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 30, 0.0001));
          }
          break;
          
        case Pattern::ChArUcoBoard:
        {
          // May need to instantiate a detector
          if (! itsChArUcoDetector)
          {
            // Instantiate the dictionary:
            cv::aruco::Dictionary dico = ArUco::getDictionary(dictionary::get());
            cv::aruco::CharucoBoard ch_board({boardSize::get().width, boardSize::get().height}, squareSize::get(),
                                             markerSize::get(), dico);
            cv::aruco::DetectorParameters detector_params;
            detector_params.cornerRefinementMethod = int(cv::aruco::CORNER_REFINE_NONE); // recommended by docs
            
            cv::aruco::CharucoParameters charuco_params;
            charuco_params.tryRefineMarkers = true;
            charuco_params.cameraMatrix = itsCameraMatrix;
            charuco_params.distCoeffs = itsDistCoeffs;
            itsChArUcoDetector.reset(new cv::aruco::CharucoDetector(ch_board, charuco_params, detector_params));
          }

          // Detect the board:
          std::vector<int> markerIds, charucoIds;
          std::vector<std::vector<cv::Point2f> > markerCorners;
          itsChArUcoDetector->detectBoard(view, pointBuf, charucoIds, markerCorners, markerIds);
          size_t needed_num = (size_t)((bs.height - 1)*(bs.width - 1));
          LINFO("ChArUco: detected "<< pointBuf.size() << " out of " << needed_num << " corners");
          found = (pointBuf.size() == needed_num);

          // Special handling of charuco boards:
          if (found)
          {
            LINFO("ChArUco calibration board successfully detected");
          
            itsImagePoints.push_back(pointBuf);
          
            // Draw the corners into the image:
            itsLastGoodView = view.clone();
            itsLastGoodTime = std::chrono::steady_clock::now();
            if (markerIds.size() > 0)
              cv::aruco::drawDetectedMarkers(itsLastGoodView, markerCorners);
            if (charucoIds.size() > 0)
              cv::aruco::drawDetectedCornersCharuco(itsLastGoodView, pointBuf, charucoIds, cv::Scalar(255, 0, 0));
          }
          break;
        }
        
        case Pattern::CirclesGrid:
          found = findCirclesGrid(view, bs, pointBuf);
          break;
          
        case Pattern::AsymmetricCirclesGrid:
          found = findCirclesGrid(view, bs, pointBuf, cv::CALIB_CB_ASYMMETRIC_GRID);
          break;
        }

        // Common handling of non-charuco boards:
        if (pattern::get() != Pattern::ChArUcoBoard)
          if (found)
            {
              LINFO("Calibration board successfully detected");
              
              itsImagePoints.push_back(pointBuf);
              
              // Draw the corners into the image:
              itsLastGoodView = view.clone();
              itsLastGoodTime = std::chrono::steady_clock::now();
              drawChessboardCorners(itsLastGoodView, cv::Size(bs.width - 1, bs.height - 1), cv::Mat(pointBuf), found);
              
            }
            else engine()->reportError("Calibration board was not detected. Make sure it is in full view with no "
                                       "occlusions, reflections, strong shadows, etc.");
      }

      // Do we want to calibrate now?
      if (calibrate::get())
      {
        calibrate::set(false);

        if (itsImagePoints.size() < 5)
          engine()->reportError("Need at least 5 good board views before calibration");
        else
        {
          do_calibration();

          if (itsCalibrated == false)
          {
            engine()->reportError("Calibration failed. Let's try again.");
            restart();
          }
        }
      }
 
      // If we are calibrated, show either live view or undistorted view:
      if (itsCalibrated)
      {
        if (showUndistorted::get())
        {
          cv::Mat temp = view.clone();
          if (fishEye::get())
          {
            cv::Mat newCamMat;
            cv::fisheye::estimateNewCameraMatrixForUndistortRectify(itsCameraMatrix, itsDistCoeffs, itsImageSize,
                                                                    cv::Matx33d::eye(), newCamMat, 1);
            cv::fisheye::undistortImage(temp, itsLastGoodView, itsCameraMatrix, itsDistCoeffs, newCamMat);
          }
          else cv::undistort(temp, itsLastGoodView, itsCameraMatrix, itsDistCoeffs);
        }
        else itsLastGoodView = view;
      }
    }

    // ####################################################################################################
    void do_calibration()
    {
      itsCalibrated = false;
      
      // Gather the calibration flags:
      itsFlag = 0;
      if (fishEye::get())
      {
        itsFlag = cv::fisheye::CALIB_FIX_SKEW | cv::fisheye::CALIB_RECOMPUTE_EXTRINSIC;
        if (fixK1::get())             itsFlag |= cv::fisheye::CALIB_FIX_K1;
        if (fixK2::get())             itsFlag |= cv::fisheye::CALIB_FIX_K2;
        if (fixK3::get())             itsFlag |= cv::fisheye::CALIB_FIX_K3;
        if (fixPrincipalPoint::get()) itsFlag |= cv::fisheye::CALIB_FIX_PRINCIPAL_POINT;
      }
      else
      {
        if (fixPrincipalPoint::get()) itsFlag |= cv::CALIB_FIX_PRINCIPAL_POINT;
        if (zeroTangentDist::get())   itsFlag |= cv::CALIB_ZERO_TANGENT_DIST;
        if (aspectRatio::get())       itsFlag |= cv::CALIB_FIX_ASPECT_RATIO;
        if (fixK1::get())             itsFlag |= cv::CALIB_FIX_K1;
        if (fixK2::get())             itsFlag |= cv::CALIB_FIX_K2;
        if (fixK3::get())             itsFlag |= cv::CALIB_FIX_K3;
      }

      std::vector<cv::Mat> rvecs, tvecs;
      std::vector<float> reprojErrs;
      std::vector<cv::Point3f> newObjPoints;
      
      itsCalibrated = runCalibration(rvecs, tvecs, reprojErrs, itsAvgErr, newObjPoints);

      // If calibrated, save the calibration now:
      if (itsCalibrated)
      {
        jevois::CameraCalibration calib;
        calib.sensor = engine()->camerasens::get();
        calib.lens = engine()->cameralens::get();
        calib.w = itsImageSize.width;
        calib.h = itsImageSize.height;
        calib.camMatrix = itsCameraMatrix;
        calib.distCoeffs = itsDistCoeffs;
        calib.avgReprojErr = itsAvgErr;
        
        engine()->saveCameraCalibration(calib);
      }
    }

    // ####################################################################################################
    double computeReprojectionErrors(std::vector<std::vector<cv::Point3f>> const & objectPoints,
                                     std::vector<cv::Mat> const & rvecs, std::vector<cv::Mat> const & tvecs,
                                     std::vector<float> & perViewErrors)
    {
      std::vector<cv::Point2f> imagePoints2;
      size_t totalPoints = 0;
      double totalErr = 0, err;
      perViewErrors.resize(objectPoints.size());
      
      for (size_t i = 0; i < objectPoints.size(); ++i )
      {
        if (fishEye::get())
          cv::fisheye::projectPoints(objectPoints[i], imagePoints2, rvecs[i], tvecs[i], itsCameraMatrix, itsDistCoeffs);
        else
          cv::projectPoints(objectPoints[i], rvecs[i], tvecs[i], itsCameraMatrix, itsDistCoeffs, imagePoints2);

        err = cv::norm(itsImagePoints[i], imagePoints2, cv::NORM_L2);

        size_t n = objectPoints[i].size();
        perViewErrors[i] = (float) std::sqrt(err*err/n);
        totalErr        += err*err;
        totalPoints     += n;
      }

      return std::sqrt(totalErr / totalPoints);
    }

    // ####################################################################################################
    void calcBoardCornerPositions(cv::Size boardsiz, float squaresiz, std::vector<cv::Point3f> & corners)
    {
      corners.clear();

      switch (pattern::get())
      {
      case Pattern::ChessBoard:
      case Pattern::CirclesGrid:
        for (int i = 0; i < boardsiz.height - 1; ++i)
          for (int j = 0; j < boardsiz.width - 1; ++j)
            corners.push_back(cv::Point3f(j*squaresiz, i*squaresiz, 0));
        break;
        
      case Pattern::ChArUcoBoard:
        for (int i = 0; i < boardsiz.height - 1; ++i)
          for (int j = 0; j < boardsiz.width - 1; ++j)
            corners.push_back(cv::Point3f(j*squaresiz, i*squaresiz, 0));
        break;
        
      case Pattern::AsymmetricCirclesGrid:
        for (int i = 0; i < boardsiz.height - 1; i++)
          for (int j = 0; j < boardsiz.width - 1; j++)
            corners.push_back(cv::Point3f((2 * j + i % 2)*squaresiz, i*squaresiz, 0));
        break;
      }
    }
    
    // ####################################################################################################
    bool runCalibration(std::vector<cv::Mat> & rvecs, std::vector<cv::Mat> & tvecs, std::vector<float> & reprojErrs,
                        double & totalAvgErr, std::vector<cv::Point3f> & newObjPoints)
    {
      itsCameraMatrix = cv::Mat::eye(3, 3, CV_64F);
      if (fishEye::get() == false && itsFlag & cv::CALIB_FIX_ASPECT_RATIO)
        itsCameraMatrix.at<double>(0,0) = aspectRatio::get();
      if (fishEye::get()) itsDistCoeffs = cv::Mat::zeros(4, 1, CV_64F);
      else itsDistCoeffs = cv::Mat::zeros(8, 1, CV_64F);

      std::vector<std::vector<cv::Point3f> > objectPoints(1);
      calcBoardCornerPositions(boardSize::get(), squareSize::get(), objectPoints[0]);
      newObjPoints = objectPoints[0];
      
      objectPoints.resize(itsImagePoints.size(), objectPoints[0]);
      
      // Find intrinsic and extrinsic camera parameters
      double rms;
      
      if (fishEye::get())
      {
        cv::Mat _rvecs, _tvecs;
        rms = cv::fisheye::calibrate(objectPoints, itsImagePoints, itsImageSize, itsCameraMatrix, itsDistCoeffs,
                                     _rvecs, _tvecs, itsFlag);
        
        rvecs.reserve(_rvecs.rows);
        tvecs.reserve(_tvecs.rows);
        for (int i = 0; i < int(objectPoints.size()); i++)
        {
          rvecs.push_back(_rvecs.row(i));
          tvecs.push_back(_tvecs.row(i));
        }
      }
      else
      {
        int iFixedPoint = -1;
        rms = cv::calibrateCameraRO(objectPoints, itsImagePoints, itsImageSize, iFixedPoint,
                                    itsCameraMatrix, itsDistCoeffs, rvecs, tvecs, newObjPoints,
                                    itsFlag | cv::CALIB_USE_LU);
      }
      
      LINFO("Re-projection error reported by calibrateCamera: "<< rms);
      
      bool ok = cv::checkRange(itsCameraMatrix) && cv::checkRange(itsDistCoeffs);
      
      objectPoints.clear();
      objectPoints.resize(itsImagePoints.size(), newObjPoints);
      totalAvgErr = computeReprojectionErrors(objectPoints, rvecs, tvecs, reprojErrs);
      
      return ok;
    }

    // ####################################################################################################
    //! Processing function with video output to USB
    // ####################################################################################################
    virtual void process(jevois::InputFrame && inframe, jevois::OutputFrame && outframe) override
    {
      static jevois::Timer timer("processing", 100, LOG_DEBUG);
      
      // Wait for next available camera image:
      jevois::RawImage const inimg = inframe.get();

      timer.start();

      // We only handle YUYV input frames, calibration is the same for other formats; any resolution ok:
      unsigned int const w = inimg.width, h = inimg.height;
      inimg.require("input", w, h, V4L2_PIX_FMT_YUYV);

      // Update image size:
      itsImageSize.width = w; itsImageSize.height = h;
      
      // While we process it, start a thread to wait for out frame and paste the input into it:
      jevois::RawImage outimg;
      auto paste_fut = jevois::async([&]() {
          outimg = outframe.get();
          outimg.require("output", w, 2*h + 15, V4L2_PIX_FMT_YUYV);
          jevois::rawimage::paste(inimg, outimg, 0, 0);
          jevois::rawimage::writeText(outimg, "JeVois Camera Calibration", 3, 3, jevois::yuyv::White);
          jevois::rawimage::drawFilledRect(outimg, 0, h, w, outimg.height-h, jevois::yuyv::Black);
        });

      // Convert the image to BGR and process:
      cv::Mat cvimg = jevois::rawimage::convertToCvBGR(inimg);

      // Detect/calibrate, will update itsCalibrated and itsLastGoodView:
      process_frame(cvimg);
      
      // Wait for paste to finish up:
      paste_fut.get();

      // Let camera know we are done processing the input image:
      inframe.done();

      // Draw and guide the user:
      int y = 15, y2 = h + 3; // we use y to print on top image, y2 on bottom image

      // Draw our results if any:
      if (itsLastGoodView.empty() == false)
        jevois::rawimage::pasteBGRtoYUYV(itsLastGoodView, outimg, 0, h);

      // If we are now calibrated, show live view, possibly undistorted:
      if (itsCalibrated)
      {
        y2 = jevois::rawimage::itext(outimg, "Calibrated - avg err " + std::to_string(itsAvgErr), y2);
        if (showUndistorted::get())
          y2 = jevois::rawimage::itext(outimg, "Undistorted view (see showUndistorted parameter)", y2);
        else
          y2 = jevois::rawimage::itext(outimg, "Normal view (see showUndistorted parameter)", y2);
      }
      else
      {
        // Show some guiding messages:
        y = jevois::rawimage::itext(outimg, "Set parameters then point to board", y);
        y = jevois::rawimage::itext(outimg, "Click 'grab' param to grab a board", y);
        y = jevois::rawimage::itext(outimg, "Vary viewpoints between grabs", y);
        y = jevois::rawimage::itext(outimg, "Good views so far: " + std::to_string(itsImagePoints.size()), y);

        if (itsImagePoints.size() >= 5)
        {
          y = jevois::rawimage::itext(outimg, "Ok to calibrate", y);
          jevois::rawimage::itext(outimg, "Ok to calibrate", outimg.height - 13);
        }
        else
        {
          y = jevois::rawimage::itext(outimg, "Need to grab 5+ good views", y);
          jevois::rawimage::itext(outimg, "Need to grab 5+ good views", outimg.height - 13);
        }
        y2 = jevois::rawimage::itext(outimg, "Last grabbed board", y2);
      }
      
      // Show processing fps:
      std::string const & fpscpu = timer.stop();
      jevois::rawimage::writeText(outimg, fpscpu, 3, h - 13, jevois::yuyv::White);
    
      // Send the output image with our processing results to the host over USB:
      outframe.send();
    }

#ifdef JEVOIS_PRO
    // ####################################################################################################
    //! Processing function with zero-copy and GUI on JeVois-Pro
    // ####################################################################################################
    virtual void process(jevois::InputFrame && inframe, jevois::GUIhelper & helper) override
    {
      static jevois::Timer timer("processing", 100, LOG_DEBUG);
      
      // Start the GUI frame:
      unsigned short winw, winh;
      helper.startFrame(winw, winh);

      // Draw the full-resolution camera frame:
      int x = 0, y = 0; unsigned short iw = 0, ih = 0;
      helper.drawInputFrame("camera", inframe, x, y, iw, ih);
      helper.itext("JeVois-Pro Camera Calibration");

      // Wait for next available camera image at processing resolution, get it as RGB:
      cv::Mat cvimg = inframe.getCvRGBp();
      itsImageSize.width = cvimg.cols;
      itsImageSize.height = cvimg.rows;
      inframe.done();
      
      timer.start();

      // Detect/calibrate, will update itsCalibrated and itsLastGoodView:
      process_frame(cvimg); // fixme they want bgr!
      
      // If we are now calibrated, show live view, possibly undistorted:
      if (itsCalibrated)
      {
        helper.itext("Camera is now calibrated - average reprojection error = " + std::to_string(itsAvgErr));
        if (showUndistorted::get())
          helper.itext("Showing undistorted view. Turn off showUndistorted to see normal view");
        else
          helper.itext("Showing normal view. Turn on showUndistorted to see undistorted view");

        int vx = 0, vy = 0; unsigned short vw = 0, vh = 0;
        helper.drawImage("v", itsLastGoodView, true /*rgb*/, vx, vy, vw, vh, false, true);
      }
      else
      {
        // Show some guiding messages:
        helper.itext("Good calibration board views so far: " + std::to_string(itsImagePoints.size()));
        if (itsImagePoints.size() >= 5)
          helper.itext("Enough good board views acquired for calibration, you may grab more to improve accuracy,"
                       "  or calibrate now.");
        else
          helper.itext("Need at least 5 good board views for calibration, keep grabbing, use varied viewpoints");

        // Display last good view for a while, otherwise nothing (live video already drawn):
        std::chrono::duration<float> const elapsed = std::chrono::steady_clock::now() - itsLastGoodTime;
        if (elapsed.count() < 2.0F && itsLastGoodView.empty() == false)
        {
          helper.itext("Calibration board successfully detected");
          int vx = 0, vy = 0; unsigned short vw = 0, vh = 0;
          helper.drawImage("v", itsLastGoodView, true /*rgb*/, vx, vy, vw, vh, false, true);
        }
      }

      // Draw a user interface:

      // Set window size applied only on first use ever, otherwise from imgui.ini:
      ImGui::SetNextWindowPos(ImVec2(100, 100), ImGuiCond_FirstUseEver);
      ImGui::SetNextWindowSize(ImVec2(500, 400), ImGuiCond_FirstUseEver);

      // Light blue window background:
      ImGui::PushStyleColor(ImGuiCol_WindowBg, 0xf0ffe0e0);

      if (ImGui::Begin("Calibrate Camera Controls"))
      {
        int wrap = ImGui::GetWindowSize().x - ImGui::GetFontSize() * 2.0f;
        ImGui::PushTextWrapPos(wrap);

        if (itsCalibrated)
        {
          ImGui::TextUnformatted("Camera is now calibrated and calibration was saved to disk.");
          ImGui::TextUnformatted("");
          ImGui::Text("Average reprojection error: %f (lower is better)", itsAvgErr);
          ImGui::TextUnformatted("");
          ImGui::Separator();
          
          if (ImGui::Button("Grab more views")) itsCalibrated = false;
        }
        else if (itsReady == false)
        {
          if (itsImagePoints.empty())
          {
            ImGui::TextUnformatted("To get started, switch to the Parameters tab of the main window, "
                                   "and set some parameters.");
            ImGui::TextUnformatted("");
            ImGui::TextUnformatted("Critical parameters are:");
            ImGui::TextUnformatted("");
            ImGui::Bullet();
            ImGui::Text("pattern - choose a board type (chessboard, ChArUco, etc)");
            ImGui::Bullet();
            ImGui::Text("fishEye - whether you are using a fish-eye (wide angle) lens");
            ImGui::Bullet();
            ImGui::Text("squareSize - physical size in your chosen units (mm, inch, etc) of one board square");
            ImGui::Bullet();
            ImGui::Text("markerSize - if using ChArUco, physical size of markers in your units");
            ImGui::Bullet();
            ImGui::Text("boardSize - horizontal and vertical number of tiles");
            
            ImGui::TextUnformatted("");
            ImGui::Separator();
          }
          itsReady = ImGui::Button("Ready!");
        }
        else
        {
          ImGui::TextUnformatted("Point the camera so that you get a full view of the calibration board, "
                                 "then click Grab.");
          if (itsImagePoints.empty() == false)
          {
            ImGui::TextUnformatted("");
            ImGui::TextUnformatted("Move the camera around to get a variety of viewpoints.");
            ImGui::TextUnformatted("Need at least 5 valid viewpoints. The more the better.");
            ImGui::TextUnformatted("");
            ImGui::Text("Good board views so far: %zu", itsImagePoints.size());
          }
          ImGui::TextUnformatted("");
          ImGui::Separator();
          if (ImGui::Button("Grab")) grab::set(true);
          if (itsImagePoints.size() >= 5)
          {
            ImGui::SameLine(); ImGui::TextUnformatted("        "); ImGui::SameLine();
            if (ImGui::Button("Calibrate now")) calibrate::set(true);
          }
        }

        // Always show a start over button, except at the every beginning:
        if (itsReady)
        {
          ImGui::SameLine(); ImGui::TextUnformatted("     "); ImGui::SameLine();
          if (ImGui::Button("Start over")) restart();
        }
        
        ImGui::PopTextWrapPos();
      }
      
      ImGui::End();
      ImGui::PopStyleColor();
      
      // Show processing fps:
      std::string const & fpscpu = timer.stop();
      helper.iinfo(inframe, fpscpu, winw, winh);
      
      // Render the image and GUI:
      helper.endFrame();
    }
#endif

  protected:
    // Restart if some params changed:
    void restart()
    {
      itsChArUcoDetector.reset();
      itsReady = false;
      itsAvgErr = 0.0;
      itsCalibrated = false;
      itsImagePoints.clear();
    }
    
    //! Restart when some critical params are changed:
    virtual void onParamChange(pattern const & par, Pattern const & val) { restart(); }
    virtual void onParamChange(dictionary const & par, aruco::Dict const & val) { restart(); }
    virtual void onParamChange(boardSize const & par, cv::Size const & val) { restart(); }

  private:
    bool itsReady = false;
    bool itsCalibrated = false;
    int itsFlag = 0;
    cv::Size itsImageSize;
    std::vector<std::vector<cv::Point2f> > itsImagePoints;
    cv::Mat itsCameraMatrix = cv::Mat::eye(3, 3, CV_64F);
    cv::Mat itsDistCoeffs = cv::Mat::zeros(8, 1, CV_64F);
    cv::Mat itsLastGoodView;
    std::chrono::time_point<std::chrono::steady_clock> itsLastGoodTime;
    std::shared_ptr<cv::aruco::CharucoDetector> itsChArUcoDetector;
    double itsAvgErr = 0.0;
};

// Allow the module to be loaded as a shared object (.so) file:
JEVOIS_REGISTER_MODULE(CalibrateCamera);
