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
#include <jevoisbase/Components/ObjectDetection/TensorFlow.H>
#include <jevoisbase/Components/Saliency/Saliency.H>

// icon from tensorflow youtube

static jevois::ParameterCategory const ParamCateg("TensorFlow Saliency Options");

//! Parameter \relates TensorFlowSaliency
JEVOIS_DECLARE_PARAMETER(foa, cv::Size, "Width and height (in pixels) of the focus of attention. "
                         "This is the size of the image crop that is taken around the most salient "
                         "location in each frame. The foa size must fit within the camera input frame size. To avoid "
			 "rescaling, it is best to use here the size that the deep network expects as input.",
                         cv::Size(128, 128), ParamCateg);

//! Detect salient objects and identify them using TensorFlow deep neural network
/*! TensorFlow is a popular neural network framework. This module first finds the most conspicuous (salient) object in
    the scene, then identifies it using a deep neural network. It returns the top scoring candidates.

    See http://ilab.usc.edu/bu/ for more information about saliency detection, and https://www.tensorflow.org for more
    information about the TensorFlow deep neural network framework.

    \youtube{TRk8rCuUVEE}

    This module runs a TensorFlow network on an image window around the most salient point and shows the top-scoring
    results. We alternate, on every other frame, between updating the salient window crop location, and predicting what
    is in it. Actual network inference speed (time taken to compute the predictions on one image crop) is shown at the
    bottom right. See below for how to trade-off speed and accuracy.

    Note that by default this module runs fast variant of MobileNets trained on the ImageNet dataset. There are 1000
    different kinds of objects (object classes) that this network can recognize (too long to list here). It is possible
    to use bigger and more complex networks, but it will likely slow down the framerate.

    For more information about MobileNets, see
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet_v1.md
    
    For more information about the ImageNet dataset used for training, see
    http://www.image-net.org/challenges/LSVRC/2012/
    
    Sometimes this module will make mistakes! The performance of mobilenets is about 40% to 70% correct (mean average
    precision) on the test set, depending on network size (bigger networks are more accurate but slower).

    Neural network size and speed
    -----------------------------

    This module provides a parameter, \p foa, which determines the size of a region of interest that is cropped around
    the most salient location. This region will then be rescaled, if needed, to the neural network's expected input
    size. To avoid wasting time rescaling, it is hence best to select an \p foa size that is equal to the network's
    input size.

    The network actual input size varies depending on which network is used; for example, mobilenet_v1_0.25_128_quant
    expects 128x128 input images, while mobilenet_v1_1.0_224 expects 224x224. We automatically rescale the cropped
    window to the network's desired input size. Note that there is a cost to rescaling, so, for best performance, you
    should match \p foa size to the network input size.

    For example:

    - mobilenet_v1_0.25_128_quant (network size 128x128), runs at about 12ms/prediction (83.3 frames/s).
    - mobilenet_v1_0.5_128_quant (network size 128x128), runs at about 26ms/prediction (38.5 frames/s).
    - mobilenet_v1_0.25_224_quant (network size 224x224), runs at about 35ms/prediction (28.5 frames/s).
    - mobilenet_v1_1.0_224_quant (network size 224x224), runs at about 185ms/prediction (5.4 frames/s).


    When using video mappings with USB output, irrespective of \p foa, the crop around the most salient image region
    (with size given by \p foa) will always also be rescaled so that, when placed to the right of the input image, it
    fills the desired USB output dims. For example, if camera mode is 320x240 and USB output size is 544x240, then the
    attended and recognized object will be rescaled to 224x224 (since 224 = 544-320) for display purposes only. This is
    so that one does not need to change USB video resolution while playing with different values of \p foa live.

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

    Using your own network
    ----------------------

    For a step-by-step tutorial, see [Training custom TensorFlow networks for
    JeVois](http://jevois.org/tutorials/UserTensorFlowTraining.html).

    This module supports RGB or grayscale inputs, byte or float32. You should create and train your network using fast
    GPUs, and then follow the instruction here to convert your trained network to TFLite format:

    https://www.tensorflow.org/mobile/tflite/

    Then you just need to create a directory under <b>JEVOIS:/share/tensorflow/</B> with the name of your network, and,
    in there, two files, \b labels.txt with the category labels, and \b model.tflite with your model converted to
    TensorFlow Lite (flatbuffer format). Finally, edit <B>JEVOIS:/modules/JeVois/TensorFlowEasy/params.cfg</B> to
    select your new network when the module is launched.

    @author Laurent Itti

    @displayname TensorFlow Saliency
    @videomapping NONE 0 0 0.0 YUYV 320 240 15.0 JeVois TensorFlowSaliency
    @videomapping YUYV 448 240 30.0 YUYV 320 240 30.0 JeVois TensorFlowSaliency # recommended network size 128x128
    @videomapping YUYV 512 240 30.0 YUYV 320 240 30.0 JeVois TensorFlowSaliency # recommended network size 192x192
    @videomapping YUYV 544 240 30.0 YUYV 320 240 30.0 JeVois TensorFlowSaliency # recommended network size 224x224
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
class TensorFlowSaliency : public jevois::StdModule,
			   public jevois::Parameter<foa>
{
  public: 
    // ####################################################################################################
    //! Constructor
    // ####################################################################################################
    TensorFlowSaliency(std::string const & instance) : jevois::StdModule(instance), itsRx(0), itsRy(0),
	itsRw(0), itsRh(0)
    {
      itsSaliency = addSubComponent<Saliency>("saliency");
      itsTensorFlow = addSubComponent<TensorFlow>("tensorflow");
    }

    // ####################################################################################################
    //! Virtual destructor for safe inheritance
    // ####################################################################################################
    virtual ~TensorFlowSaliency()
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
    virtual void getSalROI(jevois::RawImage const & inimg)
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
      cv::Size roisiz = foa::get(); itsRw = roisiz.width; itsRh = roisiz.height;
      itsRw = std::min(itsRw, w); itsRh = std::min(itsRh, h); itsRw &= ~1; itsRh &= ~1;
      unsigned int const dmx = (mx << smlev) + (smfac >> 2);
      unsigned int const dmy = (my << smlev) + (smfac >> 2);
      itsRx = int(dmx + 1 + smfac / 4) - itsRw / 2;
      itsRy = int(dmy + 1 + smfac / 4) - itsRh / 2;
      itsRx = std::max(0, std::min(itsRx, w - itsRw));
      itsRy = std::max(0, std::min(itsRy, h - itsRh));
      itsRx &= ~1; itsRy &= ~1;
      if (itsRw <= 0 || itsRh <= 0) LFATAL("Ooops, foa size cannot be zero or negative");
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
      getSalROI(inimg);

      // Extract a raw YUYV ROI around attended point:
      cv::Mat rawimgcv = jevois::rawimage::cvImage(inimg);
      cv::Mat rawroi = rawimgcv(cv::Rect(itsRx, itsRy, itsRw, itsRh));

      // Convert the ROI to RGB:
      cv::Mat rgbroi;
      cv::cvtColor(rawroi, rgbroi, cv::COLOR_YUV2RGB_YUYV);

      // Let camera know we are done processing the input image:
      inframe.done();

      // Launch the predictions, will throw if network is not ready:
      itsResults.clear();
      try
      {
	int netinw, netinh, netinc; itsTensorFlow->getInDims(netinw, netinh, netinc);

	// Scale the ROI if needed:
	cv::Mat scaledroi = jevois::rescaleCv(rgbroi, cv::Size(netinw, netinh));

	// Predict:
	float const ptime = itsTensorFlow->predict(scaledroi, itsResults);
	LINFO("Predicted in " << ptime << "ms");

	// Send serial results and switch to next frame:
	sendSerialObjDetImg2D(w, h, itsRx + itsRw/2, itsRy + itsRh/2, itsRw, itsRh, itsResults);
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
      
      // While we process it, start a thread to wait for out frame and paste the input into it:
      jevois::RawImage outimg;
      auto paste_fut = std::async(std::launch::async, [&]() {
          outimg = outframe.get();
          outimg.require("output", outimg.width, outimg.height, V4L2_PIX_FMT_YUYV);
          
          // Paste the current input image:
          jevois::rawimage::paste(inimg, outimg, 0, 0);
          jevois::rawimage::writeText(outimg, "JeVois TensorFlow Saliency", 3, 3, jevois::yuyv::White);
          
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

      // On even frames, update the salient ROI, on odd frames, run the deep network on the latest ROI:
      if ((frameNum() & 1) == 0 || itsRw == 0)
      {
	// Run the saliency model, will update itsRx, itsRy, itsRw, and itsRh:
	getSalROI(inimg);
      }
      else
      {
        // Extract a raw YUYV ROI around attended point:
        cv::Mat rawimgcv = jevois::rawimage::cvImage(inimg);
        cv::Mat rawroi = rawimgcv(cv::Rect(itsRx, itsRy, itsRw, itsRh));

        // Convert the ROI to RGB:
        cv::Mat rgbroi;
        cv::cvtColor(rawroi, rgbroi, cv::COLOR_YUV2RGB_YUYV);

        // Let camera know we are done processing the input image:
        inframe.done();

	// Launch the predictions, will throw if network is not ready:
	itsResults.clear();
	try
	{
	  // Get the network input dims:
	  int netinw, netinh, netinc; itsTensorFlow->getInDims(netinw, netinh, netinc);

	  // Scale the ROI if needed:
	  cv::Mat scaledroi = jevois::rescaleCv(rgbroi, cv::Size(netinw, netinh));

	  // In a thread, also scale the ROI to the desired output size, i.e., USB width - camera width:
	  auto scale_fut = std::async(std::launch::async, [&]() {
	      float fac = float(outimg.width - w) / float(rgbroi.cols);
	      cv::Size displaysize(outimg.width - w, int(rgbroi.rows * fac + 0.4999F));
	      cv::Mat displayroi = jevois::rescaleCv(rgbroi, displaysize);
	      
	      // Convert back the display ROI to YUYV:
	      jevois::rawimage::convertCvRGBtoCvYUYV(displayroi, itsRawInputCv);
	    });
	  
	  // Predict:
	  float const ptime = itsTensorFlow->predict(scaledroi, itsResults);
	  
	  // Wait for paste and scale to finish up:
	  paste_fut.get(); scale_fut.get();

	  int const dispw = itsRawInputCv.cols, disph = itsRawInputCv.rows;
	  cv::Mat outimgcv = jevois::rawimage::cvImage(outimg);
            
	  // Update our output image: First paste the image we have been making predictions on:
	  itsRawInputCv.copyTo(outimgcv(cv::Rect(w, 0, dispw, disph)));
	  jevois::rawimage::drawFilledRect(outimg, w, disph, dispw, h - disph, jevois::yuyv::Black);

	  // Then draw the detections: either below the detection crop if there is room, or on top of it if not enough
	  // room below:
	  int y = disph + 3; if (y + itsTensorFlow->top::get() * 12 > h - 21) y = 3;

	  for (auto const & p : itsResults)
	  {
	    jevois::rawimage::writeText(outimg, jevois::sformat("%s: %.2F", p.category.c_str(), p.score),
					w + 3, y, jevois::yuyv::White);
	    y += 12;
	  }

	  // Send serial results:
	  sendSerialObjDetImg2D(w, h, itsRx + itsRw/2, itsRy + itsRh/2, itsRw, itsRh, itsResults);

	  // Draw some text messages:
	  jevois::rawimage::writeText(outimg, "Predict time: " + std::to_string(int(ptime)) + "ms",
				      w + 3, h - 11, jevois::yuyv::White);

	  // Finally make a copy of these new results so we can display them again on the next frame while we compute
	  // saliency:
	  itsRawPrevOutputCv = cv::Mat(h, dispw, CV_8UC2);
	  outimgcv(cv::Rect(w, 0, dispw, h)).copyTo(itsRawPrevOutputCv);

	}
	catch (std::logic_error const & e) { itsRawPrevOutputCv.release(); } // network still loading
      }

      // Show processing fps:
      std::string const & fpscpu = timer.stop();
      jevois::rawimage::writeText(outimg, fpscpu, 3, h - 13, jevois::yuyv::White);

      // Show attended location:
      jevois::rawimage::drawFilledRect(outimg, itsRx + itsRw/2 - 4, itsRy + itsRh/2 - 4, 8, 8, jevois::yuyv::LightPink);
      jevois::rawimage::drawRect(outimg, itsRx, itsRy, itsRw, itsRh, 2, jevois::yuyv::LightPink);
      
      // Send the output image with our processing results to the host over USB:
      outframe.send();
    }

    // ####################################################################################################
  protected:
    std::shared_ptr<Saliency> itsSaliency;
    std::shared_ptr<TensorFlow> itsTensorFlow;
    std::vector<jevois::ObjReco> itsResults;
    std::future<float> itsPredictFut;
    cv::Mat itsRawInputCv;
    cv::Mat itsCvImg;
    cv::Mat itsRawPrevOutputCv;
    int itsRx, itsRy, itsRw, itsRh; // last computed saliency ROI
 };

// Allow the module to be loaded as a shared object (.so) file:
JEVOIS_REGISTER_MODULE(TensorFlowSaliency);
