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
#include <jevoisbase/Components/ObjectDetection/Yolo.H>

// icon from https://pjreddie.com/darknet/yolo/

//! Detect multiple objects in scenes using the Darknet YOLO deep neural network
/*! Darknet is a popular neural network framework, and YOLO is a very interesting network that detects all objects in a
    scene in one pass. This component detects all instances of any of the objects it knows about (determined by the
    network structure, labels, dataset used for training, and weights obtained) in the image that is given to is.

    See https://pjreddie.com/darknet/yolo/

    This module runs a YOLO network and shows all detections obtained. The YOLO network is currently quite slow, hence
    it is only run once in a while. Point your camera towards some interesting scene, keep it stable, and wait for YOLO
    to tell you what it found.

    Note that by default this module runs the Pascal-VOC version of tiny-YOLO, with these object categories:

    - aeroplane
    - bicycle
    - bird
    - boat
    - bottle
    - bus
    - car
    - cat
    - chair
    - cow
    - diningtable
    - dog
    - horse
    - motorbike
    - person
    - pottedplant
    - sheep
    - sofa
    - train
    - tvmonitor

    Sometimes it will make mistakes! The performance of tiny-yolo-voc is about 57.1% correct (mean average precision) on
    the test set.

    \youtube{d5CfljT5kec}

    Serial messages
    ---------------

    - On every frame where detection results were obtained, this module sends a message
      \verbatim
      DKY framenum
      \endverbatim
      where \a framenum is the frame number (starts at 0).
    - In addition, when detections are found which are avove threhsold, one message will be sent for each detected
      object (i.e., for each box that gets drawn when USB outputs are used), using a standardized 2D message:
      + Serial message type: \b 2D
      + `id`: the category name of the recognized object
      + `x`, `y`, or vertices: standardized 2D coordinates of object center or corners
      + `w`, `h`: standardized object size
      + `extra`: recognition score (in percent confidence)

    @author Laurent Itti

    @displayname Darknet YOLO
    @videomapping NONE 0 0 0.0 YUYV 640 480 0.4 JeVois DarknetYOLO
    @videomapping YUYV 1280 480 15.0 YUYV 640 480 15.0 JeVois DarknetYOLO
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
class DarknetYOLO : public jevois::StdModule
{
  public: 
    // ####################################################################################################
    //! Constructor
    // ####################################################################################################
    DarknetYOLO(std::string const & instance) : jevois::StdModule(instance), itsFrame(0)
    {
      itsYolo = addSubComponent<Yolo>("yolo");
    }

    // ####################################################################################################
    //! Virtual destructor for safe inheritance
    // ####################################################################################################
    virtual ~DarknetYOLO()
    { }

    // ####################################################################################################
    //! Un-initialization
    // ####################################################################################################
    virtual void postUninit() override
    {
      try { itsPredictFut.get(); } catch (...) { }
    }
    
    // ####################################################################################################
    //! Processing function, no video output
    // ####################################################################################################
    virtual void process(jevois::InputFrame && inframe) override
    {
      int ready = true; float ptime = 0.0F;

      // Wait for next available camera image:
      jevois::RawImage const inimg = inframe.get();
      unsigned int const w = inimg.width, h = inimg.height;

      // Convert input image to RGB for predictions:
      itsCvImg = jevois::rawimage::convertToCvRGB(inimg);
      
      // Launch the predictions, will throw logic_error if we are still loading the network:
      try { ptime =  itsYolo->predict(itsCvImg); } catch (std::logic_error const & e) { ready = false; }

      // Let camera know we are done processing the input image:
      inframe.done();

      if (ready)
      {
        LINFO("Predicted in " << ptime << "ms");

        // Compute the boxes:
        itsYolo->computeBoxes(w, h);

        // Send serial results and switch to next frame:
        itsYolo->sendSerial(this, w, h, itsFrame);
        ++itsFrame;
      }
    }
    
    // ####################################################################################################
    //! Processing function with video output to USB
    // ####################################################################################################
    virtual void process(jevois::InputFrame && inframe, jevois::OutputFrame && outframe) override
    {
      static jevois::Timer timer("processing", 50, LOG_DEBUG);

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
          outimg.require("output", w * 2, h, inimg.fmt);

          // Paste the current input image:
          jevois::rawimage::paste(inimg, outimg, 0, 0);
          jevois::rawimage::writeText(outimg, "JeVois Darknet YOLO - input", 3, 3, jevois::yuyv::White);

          // Paste the latest prediction results, if any, otherwise a wait message:
          cv::Mat outimgcv = jevois::rawimage::cvImage(outimg);
          if (itsRawPrevOutputCv.empty() == false)
            itsRawPrevOutputCv.copyTo(outimgcv(cv::Rect(w, 0, w, h)));
          else
          {
            jevois::rawimage::drawFilledRect(outimg, w, 0, w, h, jevois::yuyv::Black);
            jevois::rawimage::writeText(outimg, "JeVois Darknet YOLO - loading network - please wait...",
                                        w + 3, 3, jevois::yuyv::White);
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

          // Wait for paste to finish up:
          paste_fut.get();

          // Let camera know we are done processing the input image:
          inframe.done();

          if (success)
          {
            cv::Mat outimgcv = jevois::rawimage::cvImage(outimg);

            // Update our output image: First paste the image we have been making predictions on:
            if (itsRawPrevOutputCv.empty()) itsRawPrevOutputCv = cv::Mat(h, w, CV_8UC2);
            itsRawInputCv.copyTo(outimgcv(cv::Rect(w, 0, w, h)));

            // Then draw the detections:
            itsYolo->drawDetections(outimg, w, h, w, 0);

            // Send serial messages:
            itsYolo->sendSerial(this, w, h, itsFrame);
            
            // Draw some text messages:
            jevois::rawimage::writeText(outimg, "JeVois Darknet YOLO - predictions", w + 3, 3, jevois::yuyv::White);
            jevois::rawimage::writeText(outimg, "YOLO predict time: " + std::to_string(int(ptime)) + "ms",
                                        w + 3, h - 13, jevois::yuyv::White);

            // Finally make a copy of these new results so we can display them again while we wait for the next round:
            outimgcv(cv::Rect(w, 0, w, h)).copyTo(itsRawPrevOutputCv);

            // Switch to next frame:
            ++itsFrame;
          }
        }
        else
        {
          // Future is not ready, do nothing except drawings on this frame (done in paste_fut thread) and we will try
          // again on the next one...
          paste_fut.get();
          inframe.done();
        }
      }
      else
      {
        // Convert input image to RGB for predictions:
        itsCvImg = jevois::rawimage::convertToCvRGB(inimg);
      
        // Also make a raw YUYV copy of the input image for later displays:
        cv::Mat inimgcv = jevois::rawimage::cvImage(inimg);
        inimgcv.copyTo(itsRawInputCv);

        // Wait for paste to finish up:
        paste_fut.get();

        // Let camera know we are done processing the input image:
        inframe.done();

        // Launch the predictions:
        itsPredictFut = std::async(std::launch::async, [&](int ww, int hh)
                                   {
                                     float pt = itsYolo->predict(itsCvImg);
                                     itsYolo->computeBoxes(ww, hh);
                                     return pt;
                                   }, w, h);
      }

      // Show processing fps:
      std::string const & fpscpu = timer.stop();
      jevois::rawimage::writeText(outimg, fpscpu, 3, h - 13, jevois::yuyv::White);
      
      // Send the output image with our processing results to the host over USB:
      outframe.send();
    }

    // ####################################################################################################
  protected:
    std::shared_ptr<Yolo> itsYolo;
    std::future<float> itsPredictFut;
    cv::Mat itsRawInputCv;
    cv::Mat itsCvImg;
    cv::Mat itsRawPrevOutputCv;
    unsigned long itsFrame;
};

// Allow the module to be loaded as a shared object (.so) file:
JEVOIS_REGISTER_MODULE(DarknetYOLO);
