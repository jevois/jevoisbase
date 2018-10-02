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
#include <jevoisbase/Components/Saliency/Saliency.H>

// icon from https://pjreddie.com/darknet/

static jevois::ParameterCategory const ParamCateg("Darknet Saliency Options");

//! Parameter \relates DarknetSaliency
JEVOIS_DECLARE_PARAMETER(foa, cv::Size, "Width and height (in pixels) of the focus of attention. "
                         "This is the size of the image crop that is taken around the most salient "
                         "location in each frame. The foa size must fit within the camera input frame size.",
                         cv::Size(128, 128), ParamCateg);

//! Parameter \relates DarknetSaliency
JEVOIS_DECLARE_PARAMETER(netin, cv::Size, "Width and height (in pixels) of the neural network input "
                         "layer. This is the size to which the image crop taken around the most salient "
                         "location in each frame will be rescaled before feeding to the neural network.",
                         cv::Size(128, 128), ParamCateg);


//! Detect salient objects and identify them using Darknet deep neural network
/*! Darknet is a popular neural network framework. This module first finds the most conspicuous (salient) object in the
    scene, then identifies it using a deep neural network. It returns the top scoring candidates.

    See http://ilab.usc.edu/bu/ for more information about saliency detection, and https://pjreddie.com/darknet for more
    information about the Darknet deep neural network framework.

    This module runs a Darknet network on an image window around the most salient point and shows the top-scoring
    results. The network is currently a bit slow, hence it is only run once in a while. Point your camera towards some
    interesting object, and wait for Darknet to tell you what it found.  The framerate figures shown at the bottom left
    of the display reflect the speed at which each new video frame from the camera is processed, but in this module this
    just amounts to computing the saliency map from the camera input, converting the input image to RGB, cropping it
    around the most salient location, sending it to the neural network for processing in a separate thread, and creating
    the demo display. Actual network inference speed (time taken to compute the predictions on one image crop) is shown
    at the bottom right. See below for how to trade-off speed and accuracy.

    Note that by default this module runs the Imagenet1k tiny Darknet (it can also run the slightly slower but a bit
    more accurate Darknet Reference network; see parameters). There are 1000 different kinds of objects (object classes)
    that this network can recognize (too long to list here).

    Sometimes it will make mistakes! The performance of darknet-tiny is about 58.7% correct (mean average precision) on
    the test set, and Darknet Reference is about 61.1% correct on the test set. This is when running these networks at
    224x224 network input resolution (see parameter \p netin below).

    \youtube{77VRwFtIe8I}

    Neural network size and speed
    -----------------------------

    When using networks that are fully convolutional (as is the case for the default networks provided with this
    module), one can resize the network to any desired input size.  The network size direcly affects both speed and
    accuracy. Larger networks run slower but are more accurate.

    This module provides two parameters that allow you to adjust this tradeoff:
    - \p foa determines the size of a region of interest that is cropped around the most salient location
    - \p netin determines the size to which that region of interest is rescaled and fed to the neural network

    For example:

    - with netin = (224 224), this module runs at about 450ms/prediction.
    - with netin = (128 128), this module runs at about 180ms/prediction.

    Finally note that, when using video mappings with USB output, irrespective of \p foa and \p netin, the crop around
    the most salient image region (with size given by \p foa) will always also be rescaled so that, when placed to the
    right of the input image, it fills the desired USB output dims. For example, if camera mode is 320x240 and USB
    output size is 544x240, then the attended and recognized object will be rescaled to 224x224 (since 224 = 544-320)
    for display purposes only. This is so that one does not need to change USB video resolution while playing with
    different values of \p foa and \p netin live.

    Serial messages
    ---------------

    On every frame where detection results were obtained that are above \p thresh, this module sends a standardized 2D
    message as specified in \ref UserSerialStyle:
      + Serial message type: \b 2D
      + `id`: top-scoring category name of the recognized object, followed by ':' and the confidence score in percent
      + `x`, `y`, or vertices: standardized 2D coordinates of object center or corners
      + `w`, `h`: standardized object size
      + `extra`: any number of additional category:score pairs which had an above-threshold score, in order of
         decreasing score
      where \a category is the category name (from \p namefile) and \a score is the confidence score from 0.0 to 100.0

    See \ref UserSerialStyle for more on standardized serial messages, and \ref coordhelpers for more info on
    standardized coordinates.


    @author Laurent Itti

    @displayname Darknet Saliency
    @videomapping NONE 0 0 0.0 YUYV 320 240 5.0 JeVois DarknetSaliency
    @videomapping YUYV 460 240 15.0 YUYV 320 240 15.0 JeVois DarknetSaliency # not for mac (width not multiple of 16)
    @videomapping YUYV 560 240 15.0 YUYV 320 240 15.0 JeVois DarknetSaliency
    @videomapping YUYV 880 480 15.0 YUYV 640 480 15.0 JeVois DarknetSaliency # set foa param to 256 256
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
class DarknetSaliency : public jevois::StdModule,
                        public jevois::Parameter<foa, netin>
{
  public: 
    // ####################################################################################################
    //! Constructor
    // ####################################################################################################
    DarknetSaliency(std::string const & instance) : jevois::StdModule(instance)
    {
      itsSaliency = addSubComponent<Saliency>("saliency");
      itsDarknet = addSubComponent<Darknet>("darknet");
    }

    // ####################################################################################################
    //! Virtual destructor for safe inheritance
    // ####################################################################################################
    virtual ~DarknetSaliency()
    { }

    // ####################################################################################################
    //! Un-initialization
    // ####################################################################################################
    virtual void postUninit() override
    {
      try { itsPredictFut.get(); } catch (...) { }
    }
    
    // ####################################################################################################
    //! Helper function: compute saliency ROI in a thread, return top-left corner and size
    // ####################################################################################################
    virtual void getSalROI(jevois::RawImage const & inimg, int & rx, int & ry, int & rw, int & rh)
    {
      int const w = inimg.width, h = inimg.height;

      // Check whether the input image size is small, in which case we will scale the maps up one notch:
      if (w < 170) { itsSaliency->centermin::set(1); itsSaliency->smscale::set(3); }
      else { itsSaliency->centermin::set(2); itsSaliency->smscale::set(4); }

      // Find the most salient location, no gist for now:
      itsSaliency->process(inimg, false);

      // Get some info from the saliency computation:
      int const smlev = itsSaliency->smscale::get();
      int const smfac = (1 << smlev);

      // Find most salient point:
      int mx, my; intg32 msal; itsSaliency->getSaliencyMax(mx, my, msal);
  
      // Compute attended ROI (note: coords must be even to avoid flipping U/V when we later paste):
      cv::Size roisiz = foa::get(); rw = roisiz.width; rh = roisiz.height;
      rw = std::min(rw, w); rh = std::min(rh, h); rw &= ~1; rh &= ~1;
      unsigned int const dmx = (mx << smlev) + (smfac >> 2);
      unsigned int const dmy = (my << smlev) + (smfac >> 2);
      rx = int(dmx + 1 + smfac / 4) - rw / 2;
      ry = int(dmy + 1 + smfac / 4) - rh / 2;
      rx = std::max(0, std::min(rx, w - rw));
      ry = std::max(0, std::min(ry, h - rh));
      rx &= ~1; ry &= ~1;
    }
    
    // ####################################################################################################
    //! Processing function, no video output
    // ####################################################################################################
    virtual void process(jevois::InputFrame && inframe) override
    {
      // Wait for next available camera image:
      jevois::RawImage const inimg = inframe.get();
      unsigned int const w = inimg.width, h = inimg.height;

      // Find the most salient location, no gist for now:
      int rx, ry, rw, rh;
      getSalROI(inimg, rx, ry, rw, rh);

      // Extract a raw YUYV ROI around attended point:
      cv::Mat rawimgcv = jevois::rawimage::cvImage(inimg);
      cv::Mat rawroi = rawimgcv(cv::Rect(rx, ry, rw, rh));

      // Convert the ROI to RGB:
      cv::Mat rgbroi;
      cv::cvtColor(rawroi, rgbroi, cv::COLOR_YUV2RGB_YUYV);

      // Let camera know we are done processing the input image:
      inframe.done();

      // Launch the predictions, will throw if network is not ready:
      itsResults.clear();
      try
      {
	int netinw, netinh, netinc; itsDarknet->getInDims(netinw, netinh, netinc);

	// Scale the ROI if needed:
	cv::Mat scaledroi = jevois::rescaleCv(rgbroi, cv::Size(netinw, netinh));

	// Predict:
	float const ptime = itsDarknet->predict(scaledroi, itsResults);
	LINFO("Predicted in " << ptime << "ms");

	// Send serial results and switch to next frame:
	sendSerialObjDetImg2D(w, h, rx + rw/2, ry + rh/2, rw, rh, itsResults);
      }
      catch (std::logic_error const & e) { } // network still loading
    }

    // ####################################################################################################
    //! Processing function with video output to USB
    // ####################################################################################################
    virtual void process(jevois::InputFrame && inframe, jevois::OutputFrame && outframe) override
    {
      static jevois::Timer timer("processing", 30, LOG_DEBUG);

      // Wait for next available camera image:
      jevois::RawImage const inimg = inframe.get();

      timer.start();

      // We only handle one specific pixel format, but any image size in this module:
      unsigned int const w = inimg.width, h = inimg.height;
      inimg.require("input", w, h, V4L2_PIX_FMT_YUYV);
      
      // Launch the saliency computation in a thread:
      int rx, ry, rw, rh;
      auto sal_fut = std::async(std::launch::async, [&](){ this->getSalROI(inimg, rx, ry, rw, rh); });

      // While we process it, start a thread to wait for out frame and paste the input into it:
      jevois::RawImage outimg;
      auto paste_fut = std::async(std::launch::async, [&]() {
          outimg = outframe.get();
          outimg.require("output", outimg.width, outimg.height, V4L2_PIX_FMT_YUYV);
          
          // Paste the current input image:
          jevois::rawimage::paste(inimg, outimg, 0, 0);
          jevois::rawimage::writeText(outimg, "JeVois Darknet Saliency", 3, 3, jevois::yuyv::White);
          
          // Paste the latest prediction results, if any, otherwise a wait message:
          cv::Mat outimgcv = jevois::rawimage::cvImage(outimg);
          if (itsRawPrevOutputCv.empty() == false)
            itsRawPrevOutputCv.copyTo(outimgcv(cv::Rect(w, 0, itsRawPrevOutputCv.cols, itsRawPrevOutputCv.rows)));
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
            int const dispw = itsRawInputCv.cols, disph = itsRawInputCv.rows;
            cv::Mat outimgcv = jevois::rawimage::cvImage(outimg);
            
            // Update our output image: First paste the image we have been making predictions on:
            itsRawInputCv.copyTo(outimgcv(cv::Rect(w, 0, dispw, disph)));
            jevois::rawimage::drawFilledRect(outimg, w, disph, dispw, h - disph, jevois::yuyv::Black);

            // Then draw the detections: either below the detection crop if there is room, or on top of it if not enough
            // room below:
	    int y = disph + 3; if (y + itsDarknet->top::get() * 12 > h - 21) y = 3;

            for (auto const & p : itsResults)
            {
              jevois::rawimage::writeText(outimg, jevois::sformat("%s: %.2F", p.category.c_str(), p.score),
                                          w + 3, y, jevois::yuyv::White);
              y += 12;
            }

            // Send serial results:
            sal_fut.get();
	    sendSerialObjDetImg2D(w, h, rx + rw/2, ry + rh/2, rw, rh, itsResults);

            // Draw some text messages:
            jevois::rawimage::writeText(outimg, "Predict time: " + std::to_string(int(ptime)) + "ms",
                                        w + 3, h - 11, jevois::yuyv::White);

            // Finally make a copy of these new results so we can display them again while we wait for the next round:
	    itsRawPrevOutputCv = cv::Mat(h, dispw, CV_8UC2);
	    outimgcv(cv::Rect(w, 0, dispw, h)).copyTo(itsRawPrevOutputCv);
          }
        }
        else
        {
          // Future is not ready, do nothing except drawings on this frame (done in paste_fut thread) and we will try
          // again on the next one...
          paste_fut.get(); sal_fut.get(); inframe.done();
        }
      }
      else // We are not predicting: start new predictions
      {
        // Wait for paste to finish up. Also wait for saliency to finish up so that rx, ry, rw, rh are available:
        paste_fut.get(); sal_fut.get();
        
        // Extract a raw YUYV ROI around attended point:
        cv::Mat rawimgcv = jevois::rawimage::cvImage(inimg);
        cv::Mat rawroi = rawimgcv(cv::Rect(rx, ry, rw, rh));

        // Convert the ROI to RGB:
        cv::Mat rgbroi;
        cv::cvtColor(rawroi, rgbroi, cv::COLOR_YUV2RGB_YUYV);

        // Let camera know we are done processing the input image:
        inframe.done();

        // Scale the ROI if needed to the desired network input dims:
	itsCvImg = jevois::rescaleCv(rgbroi, netin::get());

        // Also scale the ROI to the desired output size, i.e., USB width - camera width:
        float fac = float(outimg.width - w) / float(rgbroi.cols);
        cv::Size displaysize(outimg.width - w, int(rgbroi.rows * fac + 0.4999F));
        cv::Mat displayroi = jevois::rescaleCv(rgbroi, displaysize);

        // Convert back the display ROI to YUYV and store for later display, while we are still computing the network
        // predictions on that ROI:
        jevois::rawimage::convertCvRGBtoCvYUYV(displayroi, itsRawInputCv);

        // Launch the predictions; will throw if network is not ready:
	try
	{
	  int netinw, netinh, netinc; itsDarknet->getInDims(netinw, netinh, netinc); // will throw if not ready
	  itsPredictFut = std::async(std::launch::async, [&]() { return itsDarknet->predict(itsCvImg, itsResults); });
	}
	catch (std::logic_error const & e) { itsRawPrevOutputCv.release(); } // network is not ready yet
      }

      // Show processing fps:
      std::string const & fpscpu = timer.stop();
      jevois::rawimage::writeText(outimg, fpscpu, 3, h - 13, jevois::yuyv::White);

      // Show attended location:
      jevois::rawimage::drawFilledRect(outimg, rx + rw/2 - 4, ry + rh/2 - 4, 8, 8, jevois::yuyv::LightPink);
      jevois::rawimage::drawRect(outimg, rx, ry, rw, rh, 2, jevois::yuyv::LightPink);
      
      // Send the output image with our processing results to the host over USB:
      outframe.send();
    }

    // ####################################################################################################
  protected:
    std::shared_ptr<Saliency> itsSaliency;
    std::shared_ptr<Darknet> itsDarknet;
    std::vector<jevois::ObjReco> itsResults;
    std::future<float> itsPredictFut;
    cv::Mat itsRawInputCv;
    cv::Mat itsCvImg;
    cv::Mat itsRawPrevOutputCv;
 };

// Allow the module to be loaded as a shared object (.so) file:
JEVOIS_REGISTER_MODULE(DarknetSaliency);
