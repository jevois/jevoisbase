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
#include <jevois/Debug/Timer.H>
#include <jevois/Image/RawImageOps.H>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <jevoisbase/Components/ObjectDetection/Darknet.H>

// icon from https://pjreddie.com/darknet/

//! Identify objects using Darknet deep neural network
/*! Darknet is a popular neural network framework. This component identifies the object in box in the center of the
    camera field of view. It returns the top scoring candidates.

    See https://pjreddie.com/darknet

    This module runs a Darknet network and shows the top-scoring results. The network is currently a bit slow, hence it
    is only run once in a while. Point your camera towards some interesting object, make the object fit in the picture
    shown at right (which will be fed to the neural network), keep it stable, and wait for Darknet to tell you what it
    found.

    Note that by default this module runs the Imagenet1k tiny Darknet (it can also run the slightly slower but a bit
    more accurate Darknet Reference network; see parameters). There are 1000 different kinds of objects (object classes)
    that this network can recognize (too long to list here).

    Sometimes it will make mistakes! The performance of darknet-tiny is about 58.7% correct (mean average precision) on
    the test set, and Draknet Reference is about 61.1% correct on the test set.

    \youtube{d5CfljT5kec}

    Serial messages
    ---------------

    - On every frame where detection results were obtained, this module sends a message
      \verbatim
      DKF framenum
      \endverbatim
      where \a framenum is the frame number (starts at 0).
    - In addition, when detections are found which are avove threshold, up to \p top messages will be sent, for those
      category candidates that have scored above \a thresh:
      \verbatim
      DKR category score
      \endverbatim
      where \a category is the category name (from \p namefile) and \a score is the confidence scove from 0.0 to 100.0

    @author Laurent Itti

    @displayname Darknet Single
    @videomapping NONE 0 0 0.0 YUYV 320 240 2.1 JeVois DarknetSingle
    @videomapping YUYV 544 240 15.0 YUYV 320 240 15.0 JeVois DarknetSingle
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
class DarknetSingle : public jevois::Module
{
  public: 
    // ####################################################################################################
    //! Constructor
    // ####################################################################################################
    DarknetSingle(std::string const & instance) : jevois::Module(instance), itsFrame(0)
    {
      itsDarknet = addSubComponent<Darknet>("darknet");
    }

    // ####################################################################################################
    //! Virtual destructor for safe inheritance
    // ####################################################################################################
    virtual ~DarknetSingle()
    { }

    // ####################################################################################################
    //! Un-initialization
    // ####################################################################################################
    virtual void postUninit() override
    {
      try { itsPredictFut.get(); } catch (...) { }
    }
    
    // ####################################################################################################
    //! Send serial messages
    // ####################################################################################################
    void sendAllSerial()
    {
      sendSerial("DKF " + std::to_string(itsFrame));
      for (auto const & r : itsResults) sendSerial("DKR " + r.second + ' ' + jevois::sformat("%.1f", r.first));
    }
    
    // ####################################################################################################
    //! Processing function, no video output
    // ####################################################################################################
    virtual void process(jevois::InputFrame && inframe) override
    {
      // Wait for next available camera image:
      jevois::RawImage const inimg = inframe.get();
      unsigned int const w = inimg.width, h = inimg.height;

      int netw, neth, netc; bool ready = true;
      try { itsDarknet->getInDims(netw, neth, netc); } catch (std::logic_error const & e) { ready = false; }

      if (ready)
      {
        // Take a central crop of the input:
        int const offx = (w - netw) / 2;
        int const offy = (h - neth) / 2;

        cv::Mat cvimg = jevois::rawimage::cvImage(inimg);
        cv::Mat crop = cvimg(cv::Rect(offx, offy, netw, neth));
        
        // Convert crop to RGB for predictions:
        cv::cvtColor(crop, itsCvImg, CV_YUV2RGB_YUYV);
        
        // Let camera know we are done processing the input image:
        inframe.done();

        // Launch the predictions (do not catch exceptions, we already tested for network ready in this block):
        float const ptime = itsDarknet->predict(itsCvImg, itsResults);
        LINFO("Predicted in " << ptime << "ms");

        // Send serial results and switch to next frame:
        sendAllSerial();
        ++itsFrame;
      }
      else inframe.done();
    }

    // ####################################################################################################
    //! Processing function with video output to USB
    // ####################################################################################################
    virtual void process(jevois::InputFrame && inframe, jevois::OutputFrame && outframe) override
    {
      static jevois::Timer timer("processing", 50, LOG_DEBUG);

      // Fetch network input dims (will not be available until network loaded and ready):
      int netw = 0, neth = 0, netc = 0;
      try { itsDarknet->getInDims(netw, neth, netc); }
      catch (std::logic_error const & e) { itsRawPrevOutputCv.release(); }

      // Wait for next available camera image:
      jevois::RawImage const inimg = inframe.get();

      timer.start();
      
      // We only handle one specific pixel format, and any image size in this module:
      unsigned int const w = inimg.width, h = inimg.height;
      inimg.require("input", w, h, V4L2_PIX_FMT_YUYV);

      // While we process it, start a thread to wait for out frame and paste the input into it:
      jevois::RawImage outimg;
      auto paste_fut = std::async(std::launch::async, [&]() {
          outimg = outframe.get();
          if (netw) outimg.require("output", w + netw, h, inimg.fmt);

          // Paste the current input image:
          jevois::rawimage::paste(inimg, outimg, 0, 0);
          jevois::rawimage::writeText(outimg, "JeVois Darknet Single - input", 3, 3, jevois::yuyv::White);

          // Paste the latest prediction results, if any, otherwise a wait message:
          cv::Mat outimgcv = jevois::rawimage::cvImage(outimg);
          if (itsRawPrevOutputCv.empty() == false)
            itsRawPrevOutputCv.copyTo(outimgcv(cv::Rect(w, 0, netw, h)));
          else
          {
            jevois::rawimage::drawFilledRect(outimg, w, 0, outimg.width - w, h, jevois::yuyv::Black);
            jevois::rawimage::writeText(outimg, "Loading network -", w + 3, 3, jevois::yuyv::White);
            jevois::rawimage::writeText(outimg, "please wait...", w + 3, 15, jevois::yuyv::White);
          }
        });

      // Decide on what to do based on itsPredictFut: if it is valid, we are still predicting, so check whether we are
      // done and if so draw the results. Otherwise, start predicting using the current input frame:
      if (itsPredictFut.valid())
      {
        // Are we finished predicting?
        if (itsPredictFut.wait_for(std::chrono::milliseconds(5)) == std::future_status::ready)
        {
          // Do a get() on our future to free up the async thread and get any exception it might have thrown. In
          // particular, it will throw a logic_error if we are still loading the network:
          bool success = true; float ptime = 0.0F;
          try { ptime = itsPredictFut.get(); } catch (std::logic_error const & e) { success = false; }

          // Wait for paste to finish up and let camera know we are done processing the input image:
          paste_fut.get(); inframe.done();

          if (success)
          {
            cv::Mat outimgcv = jevois::rawimage::cvImage(outimg);

            // Update our output image: First paste the image we have been making predictions on:
            if (itsRawPrevOutputCv.empty()) itsRawPrevOutputCv = cv::Mat(h, netw, CV_8UC2);
            itsRawInputCv.copyTo(outimgcv(cv::Rect(w, 0, netw, neth)));
            jevois::rawimage::drawFilledRect(outimg, w, neth, netw, h - neth, jevois::yuyv::Black);

            // Then draw the detections: either below the detection crop if there is room, or on top of it if not enough
            // room below:
            int y = neth + 13;
            if (neth + itsResults.size() * 12 > h - 10) y = 3;
            for (auto const & p : itsResults)
            {
              jevois::rawimage::writeText(outimg, jevois::sformat("%s: %.2F", p.second.c_str(), p.first),
                                          w + 3, y, jevois::yuyv::White);
              y += 12;
            }

            // Send serial results:
            sendAllSerial();

            // Draw some text messages:
            jevois::rawimage::writeText(outimg, "Predict time: " + std::to_string(int(ptime)) + "ms",
                                        w + 3, h - 13, jevois::yuyv::White);

            // Finally make a copy of these new results so we can display them again while we wait for the next round:
            outimgcv(cv::Rect(w, 0, netw, h)).copyTo(itsRawPrevOutputCv);

            // Switch to next frame:
            ++itsFrame;
          }
        }
        else
        {
          // Future is not ready, do nothing except drawings on this frame (done in paste_fut thread) and we will try
          // again on the next one...
          paste_fut.get(); inframe.done();
        }
      }
      else if (netw) // We are not predicting but network is ready: start new predictions
      {
        // Take a central crop of the input:
        int const offx = (w - netw) / 2;
        int const offy = (h - neth) / 2;
        cv::Mat cvimg = jevois::rawimage::cvImage(inimg);
        cv::Mat crop = cvimg(cv::Rect(offx, offy, netw, neth));
        
        // Convert crop to RGB for predictions:
        cv::cvtColor(crop, itsCvImg, CV_YUV2RGB_YUYV);
        
        // Also make a raw YUYV copy of the crop for later displays:
        crop.copyTo(itsRawInputCv);

        // Wait for paste to finish up:
        paste_fut.get();

        // Let camera know we are done processing the input image:
        inframe.done();

        // Launch the predictions:
        itsPredictFut = std::async(std::launch::async, [&]() { return itsDarknet->predict(itsCvImg, itsResults); });
      }
      else // We are not predicting and network is not ready - do nothing except drawings of paste_fut:
      {
        paste_fut.get(); inframe.done();
      }

      // Show processing fps:
      std::string const & fpscpu = timer.stop();
      jevois::rawimage::writeText(outimg, fpscpu, 3, h - 13, jevois::yuyv::White);
      
      // Send the output image with our processing results to the host over USB:
      outframe.send();
    }

    // ####################################################################################################
  protected:
    std::shared_ptr<Darknet> itsDarknet;
    std::vector<Darknet::predresult> itsResults;
    std::future<float> itsPredictFut;
    cv::Mat itsRawInputCv;
    cv::Mat itsCvImg;
    cv::Mat itsRawPrevOutputCv;
    unsigned long itsFrame;
 };

// Allow the module to be loaded as a shared object (.so) file:
JEVOIS_REGISTER_MODULE(DarknetSingle);
