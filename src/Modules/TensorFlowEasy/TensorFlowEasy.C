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

// icon from tensorflow youtube

static jevois::ParameterCategory const ParamCateg("TensorFlow Easy Options");

//! Parameter \relates TensorFlowEasy
JEVOIS_DECLARE_PARAMETER(foa, cv::Size, "Width and height (in pixels) of the fixed, central focus of attention. "
                         "This is the size of the central image crop that is taken in each frame and fed to the "
			 "deep neural network. If the foa size does not fit within the camera input frame size, "
			 "it will be shrunk to fit. To avoid spending CPU resources on rescaling the selected "
			 "image region, it is best to use here the size that the deep network expects as input.",
                         cv::Size(128, 128), ParamCateg);

//! Identify objects using TensorFlow deep neural network
/*! TensorFlow is a popular neural network framework. This module identifies the object in a square region in the center
    of the camera field of view using a deep convolutional neural network.

    The deep network analyzes the image by filtering it using many different filter kernels, and several stacked passes
    (network layers). This essentially amounts to detecting the presence of both simple and complex parts of known
    objects in the image (e.g., from detecting edges in lower layers of the network to detecting car wheels or even
    whole cars in higher layers). The last layer of the network is reduced to a vector with one entry per known kind of
    object (object class). This module returns the class names of the top scoring candidates in the output vector, if
    any have scored above a minimum confidence threshold. When nothing is recognized with sufficiently high confidence,
    there is no output.

    \youtube{TRk8rCuUVEE}

    This module runs a TensorFlow network and shows the top-scoring results. In this module, we run the deep network on
    every video frame, so framerate will vary depending on network complexity (see below). Point your camera towards
    some interesting object, make the object fit within the grey box shown in the video (which will be fed to the neural
    network), keep it stable, and TensorFlow will tell you what it thinks this object is.

    Note that by default this module runs different flavors of MobileNets trained on the ImageNet dataset.  There are
    1000 different kinds of objects (object classes) that these networks can recognize (too long to list here). The
    input layer of these networks is 299x299, 224x224, 192x192, 160x160, or 128x128 pixels by default, depending on the
    network used. The networks provided on the JeVois microSD image have been trained on large clusters of GPUs, using
    1.2 million training images from the ImageNet dataset.

    For more information about MobileNets, see
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet_v1.md

    For more information about the ImageNet dataset used for training, see
    http://www.image-net.org/challenges/LSVRC/2012/

    Sometimes this module will make mistakes! The performance of mobilenets is about 40% to 70% correct (mean average
    precision) on the test set, depending on network size (bigger networks are more accurate but slower).

    Neural network size and speed
    -----------------------------

    This module takes a central image region of size given by the \p foa parameter. If necessary, this image region is
    then rescaled to match the deep network's expected input size.  The network input size varies depending on which
    network is used; for example, mobilenet_v1_0.25_128_quant expects 128x128 input images, while mobilenet_v1_1.0_224
    expects 224x224. Note that there is a CPU cost to rescaling, so, for best performance, you should match the \p foa
    size to the network's input size.

    For example:

    - mobilenet_v1_0.25_128_quant (network size 128x128), runs at about 12ms/prediction (83.3 frames/s).
    - mobilenet_v1_0.5_128_quant (network size 128x128), runs at about 26ms/prediction (38.5 frames/s).
    - mobilenet_v1_0.25_224_quant (network size 224x224), runs at about 35ms/prediction (28.5 frames/s).
    - mobilenet_v1_1.0_224_quant (network size 224x224), runs at about 185ms/prediction (5.4 frames/s).

    To easily select one of the available networks, see <B>JEVOIS:/modules/JeVois/TensorFlowEasy/params.cfg</B> on the
    microSD card of your JeVois camera.

    Serial messages
    ---------------

    When detections are found with confidence scores above \p thresh, a message containing up to \p top category:score
    pairs will be sent per video frame. Exact message format depends on the current \p serstyle setting and is described
    in \ref UserSerialStyle. For example, when \p serstyle is \b Detail, this module sends:

    \verbatim
    DO category:score category:score ... category:score
    \endverbatim

    where \a category is a category name (from \p namefile) and \a score is the confidence score from 0.0 to 100.0 that
    this category was recognized. The pairs are in order of decreasing score.

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

    @displayname TensorFlow Easy
    @videomapping NONE 0 0 0.0 YUYV 320 240 60.0 JeVois TensorFlowEasy
    @videomapping YUYV 320 308 30.0 YUYV 320 240 30.0 JeVois TensorFlowEasy
    @videomapping YUYV 640 548 30.0 YUYV 640 480 30.0 JeVois TensorFlowEasy
    @videomapping YUYV 1280 1092 7.0 YUYV 1280 1024 7.0 JeVois TensorFlowEasy
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
class TensorFlowEasy : public jevois::StdModule,
		       public jevois::Parameter<foa>
{
  public: 
    // ####################################################################################################
    //! Constructor
    // ####################################################################################################
    TensorFlowEasy(std::string const & instance) : jevois::StdModule(instance)
    {
      itsTensorFlow = addSubComponent<TensorFlow>("tf");
    }

    // ####################################################################################################
    //! Virtual destructor for safe inheritance
    // ####################################################################################################
    virtual ~TensorFlowEasy()
    { }

    // ####################################################################################################
    //! Processing function, no video output
    // ####################################################################################################
    virtual void process(jevois::InputFrame && inframe) override
    {
      // Wait for next available camera image:
      jevois::RawImage const inimg = inframe.get();
      unsigned int const w = inimg.width, h = inimg.height;

      // Adjust foa size if needed so it fits within the input frame:
      cv::Size foasiz = foa::get(); int foaw = foasiz.width, foah = foasiz.height;
      if (foaw > w) { foaw = w; foah = std::min(foah, foaw); }
      if (foah > h) { foah = h; foaw = std::min(foaw, foah); }
        
      // Take a central crop of the input, with size given by foa parameter:
      int const offx = ((w - foaw) / 2) & (~1);
      int const offy = ((h - foah) / 2) & (~1);
      
      cv::Mat cvimg = jevois::rawimage::cvImage(inimg);
      cv::Mat crop = cvimg(cv::Rect(offx, offy, foaw, foah));
        
      // Convert crop to RGB for predictions:
      cv::Mat rgbroi; cv::cvtColor(crop, rgbroi, cv::COLOR_YUV2RGB_YUYV);
      
      // Let camera know we are done processing the input image:
      inframe.done();

      // Launch the predictions, will throw if network is not ready (still loading):
      itsResults.clear();
      try
      {
	int netinw, netinh, netinc; itsTensorFlow->getInDims(netinw, netinh, netinc);

	// Scale the ROI if needed:
	cv::Mat scaledroi = jevois::rescaleCv(rgbroi, cv::Size(netinw, netinh));

	// Predict:
	float const ptime = itsTensorFlow->predict(scaledroi, itsResults);
	LINFO("Predicted in " << ptime << "ms");

	// Send serial results:
	sendSerialObjReco(itsResults);
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

      // Compute central crop window size from foa parameter:
      cv::Size foasiz = foa::get(); int foaw = foasiz.width, foah = foasiz.height;
      if (foaw > w) { foaw = w; foah = std::min(foah, foaw); }
      if (foah > h) { foah = h; foaw = std::min(foaw, foah); }
      int const offx = ((w - foaw) / 2) & (~1);
      int const offy = ((h - foah) / 2) & (~1);

      // While we process it, start a thread to wait for out frame and paste the input into it:
      jevois::RawImage outimg;
      auto paste_fut = std::async(std::launch::async, [&]() {
          outimg = outframe.get();
          outimg.require("output", outimg.width, h + 68, V4L2_PIX_FMT_YUYV);

          // Paste the current input image:
          jevois::rawimage::paste(inimg, outimg, 0, 0);
          jevois::rawimage::writeText(outimg, "JeVois TensorFlow Easy - input", 3, 3, jevois::yuyv::White);

	  // Draw a grey rectangle for the FOA:
	  jevois::rawimage::drawRect(outimg, offx, offy, foaw, foah, 2, jevois::yuyv::MedGrey);

	  // Blank out the bottom of the frame:
	  jevois::rawimage::drawFilledRect(outimg, 0, h, w, outimg.height - h, jevois::yuyv::Black);
        });

      // Take a central crop of the input, with size given by foa parameter:
      cv::Mat cvimg = jevois::rawimage::cvImage(inimg);
      cv::Mat crop = cvimg(cv::Rect(offx, offy, foaw, foah));
        
      // Convert crop to RGB for predictions:
      cv::Mat rgbroi; cv::cvtColor(crop, rgbroi, cv::COLOR_YUV2RGB_YUYV);
      
      // Let camera know we are done processing the input image:
      paste_fut.get(); inframe.done();

      // Launch the predictions, will throw if network is not ready:
      itsResults.clear();
      try
      {
	int netinw, netinh, netinc; itsTensorFlow->getInDims(netinw, netinh, netinc);

	// Scale the ROI if needed:
	cv::Mat scaledroi = jevois::rescaleCv(rgbroi, cv::Size(netinw, netinh));

	// Predict:
	float const ptime = itsTensorFlow->predict(scaledroi, itsResults);

	// Draw some text messages:
	jevois::rawimage::writeText(outimg, "Predict",
				    w - 7 * 6 - 2, h + 16, jevois::yuyv::White);
	jevois::rawimage::writeText(outimg, "time:",
				    w - 7 * 6 - 2, h + 28, jevois::yuyv::White);
	jevois::rawimage::writeText(outimg, std::to_string(int(ptime)) + "ms",
				    w - 7 * 6 - 2, h + 40, jevois::yuyv::White);
 
	// Send serial results:
	sendSerialObjReco(itsResults);
      }
      catch (std::logic_error const & e)
      {
	// network still loading:
	jevois::rawimage::writeText(outimg, "Loading network -", 3, h + 4, jevois::yuyv::White);
	jevois::rawimage::writeText(outimg, "please wait...", 3, h + 16, jevois::yuyv::White);
      }

      // Then write the names and scores of the detections:
      int y = h + 4; if (y + itsTensorFlow->top::get() * 12 > outimg.height - 2) y = 16;

      for (auto const & p : itsResults)
      {
	jevois::rawimage::writeText(outimg, jevois::sformat("%s: %.2F", p.category.c_str(), p.score),
				    3, y, jevois::yuyv::White);
	y += 12;
      }

      // Show processing fps:
      std::string const & fpscpu = timer.stop();
      jevois::rawimage::writeText(outimg, fpscpu, 3, h - 13, jevois::yuyv::White);
      
      // Send the output image with our processing results to the host over USB:
      outframe.send();
    }

    // ####################################################################################################
  protected:
    std::shared_ptr<TensorFlow> itsTensorFlow;
    std::vector<jevois::ObjReco> itsResults;
};

// Allow the module to be loaded as a shared object (.so) file:
JEVOIS_REGISTER_MODULE(TensorFlowEasy);
