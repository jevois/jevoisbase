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
/*! Darknet is a popular neural network framework. This module identifies the object in a square region in the center
    of the camera field of view using a deep convolutional neural network.

    The deep network analyzes the image by filtering it using many different filter kernels, and several stacked passes
    (network layers). This essentially amounts to detecting the presence of both simple and complex parts of known
    objects in the image (e.g., from detecting edges in lower layers of the network to detecting car wheels or even
    whole cars in higher layers). The last layer of the network is reduced to a vector with one entry per known kind of
    object (object class). This module returns the class names of the top scoring candidates in the output vector, if
    any have scored above a minimum confidence threshold. When nothing is recognized with sufficiently high confidence,
    there is no output.

    Darknet is a great alternative to popular neural network frameworks like Caffe, TensorFlow, MxNet, pyTorch, Theano,
    etc as it features: 1) small footprint which is great for small embedded systems; 2) hardware acceleration using ARM
    NEON instructions; 3) support for large GPUs when compiled on expensive servers, which is useful to train the
    neural networks on big servers, then copying the trained weights directly to JeVois for use with live video.

    See https://pjreddie.com/darknet for more details about darknet.

    \youtube{d5CfljT5kec}

    This module runs a Darknet network and shows the top-scoring results. The network is currently a bit slow, hence it
    is only run once in a while. Point your camera towards some interesting object, make the object fit in the picture
    shown at right (which will be fed to the neural network), keep it stable, and wait for Darknet to tell you what it
    found. The framerate figures shown at the bottom left of the display reflect the speed at which each new video frame
    from the camera is processed, but in this module this just amounts to converting the image to RGB, sending it to the
    neural network for processing in a separate thread, and creating the demo display. Actual network inference speed
    (time taken to compute the predictions on one image) is shown at the bottom right. See below for how to trade-off
    speed and accuracy.

    Note that by default this module runs the Imagenet1k tiny Darknet (it can also run the slightly slower but a bit
    more accurate Darknet Reference network; see parameters). There are 1000 different kinds of objects (object classes)
    that these networks can recognize (too long to list here). The input layer of these two networks is 224x224 pixels
    by default. This modules takes a crop at the center of the video image, with size determined by the network input
    size. With the default network parameters, this module hence requires at least 320x240 camera sensor resolution. The
    networks provided on the JeVois microSD image have been trained on large clusters of GPUs, typically using 1.2
    million training images from the ImageNet dataset.

    Sometimes this module will make mistakes! The performance of darknet-tiny is about 58.7% correct (mean average
    precision) on the test set, and Darknet Reference is about 61.1% correct on the test set, using the default 224x224
    network input layer size.

    Neural network size and speed
    -----------------------------

    When using a video mapping with USB output, the network is automatically resized to a square size that is the
    difference between the USB output video width and the camera sensor input width (e.g., when USB video mode is
    544x240 and camera sensor mode is 320x240, the network will be resized to 224x224 since 224=544-320).

    The network size direcly affects both speed and accuracy. Larger networks run slower but are more accurate.

    For example:

    - with USB output 544x240 (network size 224x224), this module runs at about 450ms/prediction.
    - with USB output 448x240 (network size 128x128), this module runs at about 180ms/prediction.

    When using a videomapping with no USB output, the network is not resized (since we would not know what to resize it
    to). You can still change its native size by changing the network's config file, for example, change the width and
    height fields in <b>JEVOIS:/share/darknet/single/cfg/tiny.cfg</b>.

    Note that network dims must always be such that they fit inside the camera input image.

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

    @author Laurent Itti

    @displayname Darknet Single
    @videomapping NONE 0 0 0.0 YUYV 320 240 2.1 JeVois DarknetSingle
    @videomapping YUYV 544 240 15.0 YUYV 320 240 15.0 JeVois DarknetSingle
    @videomapping YUYV 448 240 15.0 YUYV 320 240 15.0 JeVois DarknetSingle
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
class DarknetSingle : public jevois::StdModule
{
  public: 
    // ####################################################################################################
    //! Constructor
    // ####################################################################################################
    DarknetSingle(std::string const & instance) : jevois::StdModule(instance)
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
    //! Processing function, no video output
    // ####################################################################################################
    virtual void process(jevois::InputFrame && inframe) override
    {
      // Wait for next available camera image:
      jevois::RawImage const inimg = inframe.get();
      unsigned int const w = inimg.width, h = inimg.height;

      // Check input vs network dims, will throw if network not ready:
      int netw, neth, netc;
      try { itsDarknet->getInDims(netw, neth, netc); }
      catch (std::logic_error const & e) { inframe.done(); return; }

      if (netw > w) netw = w;
      if (neth > h) neth = h;
        
      // Take a central crop of the input:
      int const offx = ((w - netw) / 2) & (~1);
      int const offy = ((h - neth) / 2) & (~1);

      cv::Mat cvimg = jevois::rawimage::cvImage(inimg);
      cv::Mat crop = cvimg(cv::Rect(offx, offy, netw, neth));
        
      // Convert crop to RGB for predictions:
      cv::cvtColor(crop, itsCvImg, cv::COLOR_YUV2RGB_YUYV);
        
      // Let camera know we are done processing the input image:
      inframe.done();

      // Launch the predictions (do not catch exceptions, we already tested for network ready in this block):
      float const ptime = itsDarknet->predict(itsCvImg, itsResults);
      LINFO("Predicted in " << ptime << "ms");

      // Send serial results:
      sendSerialObjReco(itsResults);
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
          jevois::rawimage::writeText(outimg, "JeVois Darknet Single - input", 3, 3, jevois::yuyv::White);

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
            int const netw = itsRawInputCv.cols, neth = itsRawInputCv.rows;
            cv::Mat outimgcv = jevois::rawimage::cvImage(outimg);
            
            // Update our output image: First paste the image we have been making predictions on:
            itsRawInputCv.copyTo(outimgcv(cv::Rect(w, 0, netw, neth)));
            jevois::rawimage::drawFilledRect(outimg, w, neth, netw, h - neth, jevois::yuyv::Black);

            // Then draw the detections: either below the detection crop if there is room, or on top of it if not enough
            // room below:
	    int y = neth + 3; if (y + itsDarknet->top::get() * 12 > h - 21) y = 3;

            for (auto const & p : itsResults)
            {
              jevois::rawimage::writeText(outimg, jevois::sformat("%s: %.2F", p.category.c_str(), p.score),
                                          w + 3, y, jevois::yuyv::White);
              y += 12;
            }

            // Send serial results:
	    sendSerialObjReco(itsResults);

            // Draw some text messages:
            jevois::rawimage::writeText(outimg, "Predict time: " + std::to_string(int(ptime)) + "ms",
                                        w + 3, h - 11, jevois::yuyv::White);

            // Finally make a copy of these new results so we can display them again while we wait for the next round:
	    itsRawPrevOutputCv = cv::Mat(h, netw, CV_8UC2);
	    outimgcv(cv::Rect(w, 0, netw, h)).copyTo(itsRawPrevOutputCv);
	    
          } else { itsRawPrevOutputCv.release(); } // network is not ready yet
        }
        else
        {
          // Future is not ready, do nothing except drawings on this frame (done in paste_fut thread) and we will try
          // again on the next one...
          paste_fut.get(); inframe.done();
        }
      }
      else // We are not predicting: start new predictions
      {
        // Wait for paste to finish up:
        paste_fut.get();

        // In this module, we use square crops for the network, with size given by USB width - camera width:
        if (outimg.width < inimg.width) LFATAL("USB output image must be larger than camera input");
        int const netw = outimg.width - inimg.width;
        int const neth = netw; // square crop
        
        // Check input vs network dims:
        if (netw > w || neth > h) LFATAL("Network input window must fit within camera frame");

        // Take a central crop of the input:
        int const offx = ((w - netw) / 2) & (~1);
        int const offy = ((h - neth) / 2) & (~1);
        cv::Mat cvimg = jevois::rawimage::cvImage(inimg);
        cv::Mat crop = cvimg(cv::Rect(offx, offy, netw, neth));
        
        // Convert crop to RGB for predictions:
        cv::cvtColor(crop, itsCvImg, cv::COLOR_YUV2RGB_YUYV);
        
        // Also make a raw YUYV copy of the crop for later displays:
        crop.copyTo(itsRawInputCv);

        // Let camera know we are done processing the input image:
        inframe.done();

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
      
      // Send the output image with our processing results to the host over USB:
      outframe.send();
    }

    // ####################################################################################################
  protected:
    std::shared_ptr<Darknet> itsDarknet;
    std::vector<jevois::ObjReco> itsResults;
    std::future<float> itsPredictFut;
    cv::Mat itsRawInputCv;
    cv::Mat itsCvImg;
    cv::Mat itsRawPrevOutputCv;
 };

// Allow the module to be loaded as a shared object (.so) file:
JEVOIS_REGISTER_MODULE(DarknetSingle);
