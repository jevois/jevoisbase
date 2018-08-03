// ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//
// JeVois Smart Embedded Machine Vision Toolkit - Copyright (C) 2018 by Laurent Itti, the University of Southern
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
#include <jevoisbase/Components/ObjectDetection/BlobDetector.H>
#include <jevoisbase/Components/ArUco/ArUco.H>
#include <jevois/Debug/Log.H>
#include <jevois/Util/Utils.H>
#include <jevois/Image/RawImageOps.H>
#include <jevois/Debug/Timer.H>
#include <jevois/Util/Coordinates.H>

#include <opencv2/imgproc/imgproc.hpp>

#include <map>

static jevois::ParameterCategory const ParamCateg("ArUcoBlob Options");

//! Parameter \relates ArUcoBlob
JEVOIS_DECLARE_PARAMETER(numtrack, size_t, "Number of parallel blob trackers to run. They will be named blob0, "
			 "blob1, etc for parameters and serial messages",
			   3, ParamCateg);
 
//! Combined ArUco marker + multiple color-based object detection
/*! This modules 1) detects ArUco markers (small black-and-white geometric patterns which can be used as tags for some
    objects), and, in parallel, 2) isolates pixels within multiple given HSV ranges (hue, saturation, and value of color
    pixels), does some cleanups, and extracts object contours.  It sends information about detected ArUco tags and color
    objects over serial.

    This module was developed to allow students to easily develop visually-guided robots that can at the same time
    detect ArUco markers placed in the environment to signal certain key objects (e.g., charging station, home base) and
    colored objects of different kinds (e.g., blue people, green trees, and yellow fires).

    This module usually works best with the camera sensor set to manual exposure, manual gain, manual color balance, etc
    so that HSV color values are reliable. See the file \b script.cfg file in this module's directory for an example of
    how to set the camera settings each time this module is loaded.

    Since this is a combination module, refer to:

    - \jvmod{DemoArUco} for the ArUco algorithm and messages
    - \jvmod{ObjectTracker} for the blob detection algorithm and messages

    The number of parallel blob trackers is determined by parameter \p numtrack, which should be set before the module
    is initialized, i.e., in the module's \b params.cfg file. It cannot be changed while the module is running.

    The module runs at about 50 frames/s with 3 parallel blob detectors plus ArUco, at 320x240 camera sensor
    resolution. Increasing to 10 parallel blob detectors will still get you about 25 frames/s (but finding robust
    non-overlapping HSV ranges for all those detectors will become challenging!)

    To configure parameters \p hrange, \p srange, and \p vrange for each detector in the module's \b scrip.cfg, we
    recommend that you do it one by one for each kind of colored object you want, using the \jvmod{ObjectTracker} module
    (which shares the same color blob detection code, just for one HSV range) and the tutorial on <A
    HREF="http://jevois.org/tutorials/UserColorTracking.html">Tuning the color-based object tracker using a python
    graphical interface</A>, or the sliders in JeVois Inventor. Just make sure that both modules have the same camera
    settings in their respective \b script.cfg files.

    Using the serial outputs
    ------------------------

    We recommend the following settings (to apply after you load the module, for example in the module's \b script.cfg
    file):
    \code{.py}
    setpar serout USB      # to send to serial-over-USB, or use Hard to send to 4-pin serial port
    setpar serstyle Normal # to get ID, center location, size for every detected color blob and ArUco tag
    setpar serstamp Frame  # to add video frame number to all messages
    \endcode

    With a scene as shown in this module's screenshots, you would then get outputs like:

    \verbatim
    ...
    1557 N2 U42 -328 -9 706 569
    1557 N2 U18 338 -241 613 444
    1557 N2 blob0 616 91 406 244
    1557 N2 blob1 28 584 881 331
    1557 N2 blob2 47 -553 469 206
    1558 N2 U42 -328 -9 706 569
    1558 N2 U18 338 -241 613 444
    1558 N2 blob0 547 113 519 275
    1558 N2 blob1 28 581 881 338
    1558 N2 blob2 47 -553 469 206
    1559 N2 U42 -331 -13 700 563
    1559 N2 U18 338 -244 613 450
    1559 N2 blob0 369 153 200 194
    1559 N2 blob0 616 94 381 250
    1559 N2 blob1 28 581 881 338
    1559 N2 blob2 47 -553 469 206
    ...
    \endverbatim

    which basically means that, on frame 1557, ArUco markers U42 and U18 were detected, then blob detector named "blob0"
    (configured for light blue objects in \b script.cfg) detected one blob, then "blob1" (configured for yellow) also
    detected one, and finally "blob2" (configured for green) found one too. That was all for frame 1157, and we then
    switch to frame 1158 (with essentially the same detections), then frame 1159 (note how blob0 detected 2 different
    blobs on that frame), and so on. See \ref UserSerialStyle for more info about these messages.

    See \ref UserSerialStyle for more on standardized serial messages, and \ref coordhelpers for more info on
    standardized coordinates.

    Running with no video output (standalone mode)
    ----------------------------------------------

    Try these settings in the global initialization script file of JeVois, which is executed when JeVois powers up, in
    <b>JEVOIS:/config/initscript.cfg</b>:

    \code{.py}
    setmapping2 YUYV 320 240 45.0 JeVois ArUcoBlob # to select this module upon power up
    setpar serout Hard     # to send detection messages to 4-pin serial port
    setpar serstyle Normal # to get ID, center location, size
    setpar serstamp Frame  # to add video frame number to all messages
    streamon               # start capturing and processing camera sensor data
    \endcode

    Make sure you do not have conflicting settings in the module's \b params.cfg or \b script.cfg file; as a reminder,
    the order of execution is: 1) \b initscript.cfg runs, which loads the module through the `setmapping2` command; 2)
    as part of the loading process and before the module is initialized, settings in \b params.cfg are applied; 3) the
    module is then initialized and commands in \b script.cfg are run; 4) the additional commands following `setmapping2`
    in \b initscript.cfg are finally run. Next time JeVois starts up, it will automatically load this module and start
    sending messages to the hardware 4-pin serial port, which you should then connect to an Arduino or other robot
    controller.

    @author Laurent Itti

    @displayname ArUco Blob
    @videomapping YUYV 320 266 30.0 YUYV 320 240 30.0 JeVois ArUcoBlob
    @videomapping NONE 0 0 0.0 YUYV 320 240 30.0 JeVois ArUcoBlob
    @email itti\@usc.edu
    @address University of Southern California, HNB-07A, 3641 Watt Way, Los Angeles, CA 90089-2520, USA
    @copyright Copyright (C) 2018 by Laurent Itti, iLab and the University of Southern California
    @mainurl http://jevois.org
    @supporturl http://jevois.org/doc
    @otherurl http://iLab.usc.edu
    @license GPL v3
    @distribution Unrestricted
    @restrictions None
    \ingroup modules */
class ArUcoBlob : public jevois::StdModule,
		  public jevois::Parameter<numtrack>
{
  public:
    // ####################################################################################################
    //! Constructor
    // ####################################################################################################
    ArUcoBlob(std::string const & instance) :
	jevois::StdModule(instance)
    {
      itsArUco = addSubComponent<ArUco>("aruco");
      // We instantiate the blob detectors in postInit() once their number is finalized
    }

    // ####################################################################################################
    //! Post-init: instantiate the blob detectors
    // ####################################################################################################
    void postInit() override
    {
      numtrack::freeze();
      
      for (int i = 0; i < numtrack::get(); ++i)
	itsBlobs.push_back(addSubComponent<BlobDetector>("blob" + std::to_string(i)));
    }

    // ####################################################################################################
    // Pre-unInit: release the blob detectors
    // ####################################################################################################
    void preUninit() override
    {
      for (auto & b : itsBlobs) removeSubComponent(b);
      itsBlobs.clear();
    }
    
    // ####################################################################################################
    //! Virtual destructor for safe inheritance
    // ####################################################################################################
    virtual ~ArUcoBlob() { }

    // ####################################################################################################
    //! Detect blobs in parallel threads
    // ####################################################################################################
    void detectBlobs(jevois::RawImage * outimg = nullptr)
    {
      itsContours.clear();
      
      // In a bunch of threads, detect blobs and get the contours:
      for (auto & b : itsBlobs)
	itsBlobFuts.push_back
	  (std::async(std::launch::async,
		      [this](std::shared_ptr<BlobDetector> b)
		      {
			auto c = b->detect(itsImgHsv);
			std::lock_guard<std::mutex> _(itsBlobMtx);
			itsContours[b->instanceName()] = std::move(c);
		      }, b));
    }

    // ####################################################################################################
    //! Gather our blob threads and send/draw the results
    // ####################################################################################################
    void sendBlobs(unsigned int w, unsigned int h, jevois::RawImage * outimg = nullptr)
    {
      for (auto & f : itsBlobFuts)
	try { f.get(); } catch (...) { LERROR("Ooops, some blob detector threw -- IGNORED"); }
      itsBlobFuts.clear();
      
      // Send a serial message for each detected blob:
      for (auto const & cc : itsContours)
	for (auto const & c : cc.second)
	  sendSerialContour2D(w, h, c, cc.first);
    }

    // ####################################################################################################
    //! Detect ArUcos
    // ####################################################################################################
    void detectArUco(cv::Mat cvimg, std::vector<int> & ids, std::vector<std::vector<cv::Point2f>> & corners,
		     std::vector<cv::Vec3d> & rvecs, std::vector<cv::Vec3d> & tvecs,
		     unsigned int h, jevois::RawImage * outimg = nullptr)
    {
      itsArUco->detectMarkers(cvimg, ids, corners);

      if (itsArUco->dopose::get() && ids.empty() == false)
        itsArUco->estimatePoseSingleMarkers(corners, rvecs, tvecs);

      // Show all the results:
      if (outimg) itsArUco->drawDetections(*outimg, 3, h+2, ids, corners, rvecs, tvecs);
    }

    // ####################################################################################################
    //! Processing function, no USB video output
    // ####################################################################################################
    virtual void process(jevois::InputFrame && inframe) override
    {
      // Wait for next available camera image. Any resolution and format ok:
      jevois::RawImage inimg = inframe.get(); unsigned int const w = inimg.width, h = inimg.height;

      // Convert input image to BGR24, then to HSV:
      cv::Mat imgbgr = jevois::rawimage::convertToCvBGR(inimg);
      cv::cvtColor(imgbgr, itsImgHsv, cv::COLOR_BGR2HSV);

      // Detect blobs in parallel threads:
      detectBlobs();
			    
      // In our thread, detect ArUcos; first convert to gray:
      cv::Mat cvimg = jevois::rawimage::convertToCvGray(inimg);

      // Let camera know we are done processing the input image:
      inframe.done();

      // Detect ArUcos:
      std::vector<int> ids; std::vector<std::vector<cv::Point2f> > corners; std::vector<cv::Vec3d> rvecs, tvecs;
      detectArUco(cvimg, ids, corners, rvecs, tvecs, h);

      // Send ArUco serial output:
      itsArUco->sendSerial(this, ids, corners, w, h, rvecs, tvecs);

      // Done with ArUco, gather the blobs and send the serial messages:
      sendBlobs(w, h);
    }
    
    // ####################################################################################################
    //! Processing function, with USB video output
    // ####################################################################################################
    virtual void process(jevois::InputFrame && inframe, jevois::OutputFrame && outframe) override
    {
      static jevois::Timer timer("processing");

      // Wait for next available camera image. Any resolution ok, but require YUYV since we assume it for drawings:
      jevois::RawImage inimg = inframe.get(); unsigned int const w = inimg.width, h = inimg.height;
      inimg.require("input", w, h, V4L2_PIX_FMT_YUYV);

      timer.start();

      // While we process it, start a thread to wait for output frame and paste the input image into it:
      jevois::RawImage outimg; // main thread should not use outimg until paste thread is complete
      auto paste_fut = std::async(std::launch::async, [&]() {
          outimg = outframe.get();
          outimg.require("output", w, h + 26, inimg.fmt);
          jevois::rawimage::paste(inimg, outimg, 0, 0);
          jevois::rawimage::writeText(outimg, "JeVois ArUco + Color Blobs", 3, 3, jevois::yuyv::White);
          jevois::rawimage::drawFilledRect(outimg, 0, h, w, outimg.height-h, 0x8000);
        });

      // Convert input image to BGR24, then to HSV:
      cv::Mat imgbgr = jevois::rawimage::convertToCvBGR(inimg);
      cv::cvtColor(imgbgr, itsImgHsv, cv::COLOR_BGR2HSV);

      // Detect blobs in parallel threads:
      detectBlobs(&outimg);
			    
      // In our thread, detect ArUcos; first convert to gray:
      cv::Mat cvimg = jevois::rawimage::convertToCvGray(inimg);

      // Let camera know we are done processing the input image:
      inframe.done();

      // Wait for paste to finish up:
      paste_fut.get();

      // Detect ArUcos:
      std::vector<int> ids; std::vector<std::vector<cv::Point2f> > corners; std::vector<cv::Vec3d> rvecs, tvecs;
      detectArUco(cvimg, ids, corners, rvecs, tvecs, h, &outimg);

      // Send ArUco serial output:
      itsArUco->sendSerial(this, ids, corners, w, h, rvecs, tvecs);

      // Done with ArUco, gather the blobs and send the serial messages:
      sendBlobs(w, h);

      // Draw all detected contours in a thread:
      std::future<void> draw_fut = std::async(std::launch::async, [&]() {
	  // We reinterpret the top portion of our YUYV output image as an opencv 8UC2 image:
	  cv::Mat outuc2 = jevois::rawimage::cvImage(outimg); // pixel data shared
	  for (auto const & cc : itsContours)
	  {
	    int color = (cc.first.back() - '0') * 123;
	    cv::drawContours(outuc2, cc.second, -1, color, 2, 8);
	    for (auto const & cont : cc.second)
	    {
	      cv::Moments moment = cv::moments(cont);
	      double const area = moment.m00;
	      int const x = int(moment.m10 / area + 0.4999);
	      int const y = int(moment.m01 / area + 0.4999);
	      jevois::rawimage::drawCircle(outimg, x, y, 20, 1, color);
	    }
	  }
	});

      // Show number of detected objects:
      std::string str = "Detected ";
      for (auto const & cc : itsContours) str += std::to_string(cc.second.size()) + ' ';
      jevois::rawimage::writeText(outimg, str + "blobs.", 3, h + 14, jevois::yuyv::White);
      
      // Show processing fps:
      std::string const & fpscpu = timer.stop();
      jevois::rawimage::writeText(outimg, fpscpu, 3, h - 13, jevois::yuyv::White);

      // Wait until all contours are drawn, if they had been requested:
      draw_fut.get();
      
      // Send the output image with our processing results to the host over USB:
      outframe.send();
    }

    // ####################################################################################################
  protected:
    std::shared_ptr<ArUco> itsArUco;
    std::vector<std::shared_ptr<BlobDetector> > itsBlobs;
    cv::Mat itsImgHsv;
    std::map<std::string, std::vector<std::vector<cv::Point>>> itsContours;
    std::vector<std::future<void>> itsBlobFuts;
    std::mutex itsBlobMtx;
};

// Allow the module to be loaded as a shared object (.so) file:
JEVOIS_REGISTER_MODULE(ArUcoBlob);
