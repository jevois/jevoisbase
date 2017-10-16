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
#include <jevois/Types/BoundedBuffer.H>

#include <linux/videodev2.h>
#include <jevoisbase/Components/ObjectMatcher/ObjectMatcher.H>
#include <jevoisbase/Components/Saliency/Saliency.H>
#include <opencv2/opencv.hpp>

// icon by Freepik in people at flaticon

// Module parameters:
#define PATHPREFIX "/jevois/data/saliencysurf/"
static jevois::ParameterCategory const ParamCateg("Salient Regions Options");

//! Parameter \relates SaliencySURF
JEVOIS_DECLARE_PARAMETER(inhsigma, float, "Sigma (pixels) used for inhibition of return", 32.0F, ParamCateg);

//! Parameter \relates SaliencySURF
JEVOIS_DECLARE_PARAMETER(regions, size_t, "Number of salient regions", 2, ParamCateg);

//! Parameter \relates SaliencySURF
JEVOIS_DECLARE_PARAMETER(rsiz, size_t, "Width and height (pixels) of salient regions", 64, ParamCateg);

//! Parameter \relates SaliencySURF
JEVOIS_DECLARE_PARAMETER(save, bool, "Save regions when true, useful to create a training set. They will be saved to "
                         PATHPREFIX, false, ParamCateg);

//! Simple salient region detection and identification using keypoint matching
/*! This module finds objects by matching keypoint descriptors between a current set of salient regions and a set of
    training images.

    Here we use SURF keypoints and descriptors as provided by OpenCV. The algorithm is quite slow and consists of 3
    phases:
    - detect keypoint locations,
    - compute keypoint descriptors,
    - and match descriptors from current image to training image descriptors.

    Here, we alternate between computing keypoints and descriptors on one frame (or more, depending on how slow that
    gets), and doing the matching on the next frame. This module also provides an example of letting some computation
    happen even after we exit the \c process() function. Here, we keep detecting keypoints and computing descriptors
    even outside \c process().

    Also see the \jvmod{ObjectDetect} module for a related algorithm (without attention).

    Training
    --------

    Simply add images of the objects you want to detect into <b>JEVOIS:/modules/JeVois/SaliencySURF/images/</b> on your
    JeVois microSD card.

    Those will be processed when the module starts.

    The names of recognized objects returned by this module are simply the file names of the pictures you have added in
    that directory. No additional training procedure is needed.

    Beware that the more images you add, the slower the algorithm will run, and the higher your chances of confusions
    among several of your objects.

    This module provides parameters that allow you to determine how strict a match should be. With stricter matching,
    you may sometimes miss an object (i.e., it was there, but was not detected by the algorithm). With looser matching,
    you may get more false alarms (i.e., there was something else in the camera's view, but it was matched as one of
    your objects). If you are experiencing difficulties getting any matches, try to loosen the settings, for example:

    \verbatim
    setpar goodpts 5 ... 100
    setpar distthresh 0.5
    \endverbatim

    @author Laurent Itti

    @displayname Saliency SURF
    @videomapping YUYV 320 288 30.0 YUYV 320 240 30.0 JeVois SaliencySURF
    @email itti\@usc.edu
    @address University of Southern California, HNB-07A, 3641 Watt Way, Los Angeles, CA 90089-2520, USA
    @copyright Copyright (C) 2016 by Laurent Itti, iLab and the University of Southern California
    @mainurl http://jevois.org
    @supporturl http://jevois.org/doc
    @otherurl http://iLab.usc.edu
    @license GPL v3
    @distribution Unrestricted
    @restrictions None
    \ingroup modules */
class SaliencySURF : public jevois::Module, public jevois::Parameter<inhsigma, regions, rsiz, save>
{
  public:
    // ####################################################################################################
    //! Constructor
    // ####################################################################################################
    SaliencySURF(std::string const & instance) : jevois::Module(instance), itsBuf(1000)
    {
      itsSaliency = addSubComponent<Saliency>("saliency");
      itsMatcher = addSubComponent<ObjectMatcher>("surf");
    }

    // ####################################################################################################
    //! Virtual destructor for safe inheritance
    // ####################################################################################################
    virtual ~SaliencySURF()
    { }

    // ####################################################################################################
    //! Get started
    // ####################################################################################################
    void postInit() override
    {
      // Get our run() thread going, it is in charge of compresing and saving frames:
      itsRunFut = std::async(std::launch::async, &SaliencySURF::run, this);

      LINFO("Using " << itsMatcher->numtrain() << " Training Images.");
    }

    // ####################################################################################################
    //! Get stopped
    // ####################################################################################################
    void postUninit() override
    {
      // Push an empty frame into our buffer to signal the end of video to our thread:
      itsBuf.push(cv::Mat());

      // Wait for the thread to complete:
      LINFO("Waiting for writer thread to complete, " << itsBuf.filled_size() << " frames to go...");
      try { itsRunFut.get(); } catch (...) { jevois::warnAndIgnoreException(); }
      LINFO("Writer thread completed. Syncing disk...");
      if (std::system("/bin/sync")) LERROR("Error syncing disk -- IGNORED");
    }

    // ####################################################################################################
    //! Processing function
    // ####################################################################################################
    virtual void process(jevois::InputFrame && inframe, jevois::OutputFrame && outframe) override
    {
      static jevois::Timer timer("processing", 30, LOG_DEBUG);

      // Wait for next available camera image. Any resolution ok, but require YUYV since we assume it for drawings:
      jevois::RawImage inimg = inframe.get(); unsigned int const w = inimg.width, h = inimg.height;
      inimg.require("input", w, h, V4L2_PIX_FMT_YUYV);

      timer.start();

      // While we process it, start a thread to wait for output frame and paste the input image into it:
      jevois::RawImage outimg; // main thread should not use outimg until paste thread is complete
      cv::Mat grayimg;

      auto paste_fut = std::async(std::launch::async, [&]() {
          // Convert input image to greyscale:
          grayimg = jevois::rawimage::convertToCvGray(inimg);

          // Wait output frame and paste input into it:
          outimg = outframe.get();
          outimg.require("output", w, h + 4*12, inimg.fmt);
          jevois::rawimage::paste(inimg, outimg, 0, 0);
          jevois::rawimage::writeText(outimg, "JeVois Saliency + SURF Demo", 3, 3, jevois::yuyv::White);
          jevois::rawimage::drawFilledRect(outimg, 0, h, w, outimg.height-h, 0x8000);
        });

      // Compute the saliency map, no gist:
      itsSaliency->process(inimg, false);

      // Get some info from the saliency computation:
      int const smlev = itsSaliency->smscale::get();
      int const smfac = (1 << smlev);
      int const rwh = rsiz::get();

      // We need the grayscale and output images to proceed:
      paste_fut.get();
      
      // Process each region:
      int k = 0;
      for (size_t i = 0; i < regions::get(); ++i)
      {
        // Find most salient point:
        int mx, my; intg32 msal; itsSaliency->getSaliencyMax(mx, my, msal);
      
        // Compute attended ROI (note: coords must be even to avoid flipping U/V when we later paste):
        unsigned int const dmx = (mx << smlev) + (smfac >> 2);
        unsigned int const dmy = (my << smlev) + (smfac >> 2);
        int rx = (std::min(int(w) - rwh/2, std::max(rwh/2, int(dmx + 1 + (smfac >> 2))))) & (~1);
        int ry = (std::min(int(h) - rwh/2, std::max(rwh/2, int(dmy + 1 + (smfac >> 2))))) & (~1);
        unsigned short col = jevois::yuyv::White;

        // Grab the ROI:
        cv::Mat roi = grayimg(cv::Rect(rx - rwh/2, ry - rwh/2, rwh, rwh));

        // Save it if desired:
        if (save::get()) itsBuf.push(roi);

        // Process it through our matcher:
        size_t trainidx;
        double dist = itsMatcher->process(roi, trainidx);
        if (dist < 100.0)
        {
          jevois::rawimage::writeText(outimg, std::string("Detected: ") + itsMatcher->traindata(trainidx).name +
                                      " avg distance " + std::to_string(dist), 3, h + k*12 + 2, jevois::yuyv::White);
          col = jevois::yuyv::LightGreen;
          ++k;
        }

        // Draw the ROI:
        jevois::rawimage::drawRect(outimg, rx - rwh/2, ry - rwh/2, rwh, rwh, 1, col);

        // Inhibit this salient location so we move to the next one:
        itsSaliency->inhibitionOfReturn(mx, my, inhsigma::get() / smfac);
      }

      // Show processing fps:
      std::string const & fpscpu = timer.stop();
      jevois::rawimage::writeText(outimg, fpscpu, 3, h - 13, jevois::yuyv::White);
      
      // Send the output image with our processing results to the host over USB:
      outframe.send();
    }

  private:
    // ####################################################################################################
    void run() // Runs in a thread to save regions as images, for training
    {
      size_t frame = 0; char tmp[2048];

      // Create directory just in case it does not exist:
      std::string const cmd = "/bin/mkdir -p " PATHPREFIX;
      if (std::system(cmd.c_str())) LERROR("Error running [" << cmd << "] -- IGNORED");

      while (true)
      {
        // Get next frame from the buffer:
        cv::Mat im = itsBuf.pop();
        
        // An empty image will be pushed when we are ready to unload the module:
        if (im.empty()) break;
        
        // Write the frame:
        std::snprintf(tmp, 2047, "%s/frame%06zu.png", PATHPREFIX, frame);
        cv::imwrite(tmp, im);
        
        // Report what is going on once in a while:
        if ((++frame % 100) == 0) LINFO("Saved " << frame << " salient regions.");
      }
    }

    std::shared_ptr<ObjectMatcher> itsMatcher;
    std::shared_ptr<Saliency> itsSaliency;
    std::future<void> itsRunFut;
    jevois::BoundedBuffer<cv::Mat, jevois::BlockingBehavior::Block, jevois::BlockingBehavior::Block> itsBuf;
};

// Allow the module to be loaded as a shared object (.so) file:
JEVOIS_REGISTER_MODULE(SaliencySURF);
