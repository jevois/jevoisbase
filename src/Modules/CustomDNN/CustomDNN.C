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
#include <jevois/DNN/Pipeline.H>
#include <jevois/DNN/PostProcessorDetect.H>
#include <opencv2/imgproc/imgproc.hpp>

// icon from opencv

// ####################################################################################################
// A custom PostProcessorDetect that allows access to detected boxes
class MyPostProc : public jevois::dnn::PostProcessorDetect
{
  public:
    // Use inherited constructor
    using jevois::dnn::PostProcessorDetect::PostProcessorDetect;
    
    // Virtual destructor for safe inheritance
    virtual ~MyPostProc()
    { }
    
    // Get access to the detections. Beware that this is not thread safe.
    std::vector<jevois::ObjDetect> const & getDetections()
    { return itsDetections; }
};

// ####################################################################################################
// A modified Pipeline class that loads our custom MyPostProc for detection networks
class MyPipeline : public jevois::dnn::Pipeline
{
  public:
    // Use inherited constructor
    using jevois::dnn::Pipeline::Pipeline;
    
    // Virtual destructor
    virtual ~MyPipeline()
    { }
    
    // When postproc is Detect, instantiate MyPostProc
    /* This callback is invoked when "set(val)" is called on our postproc parameter, which happens when a model is
       selected in the JeVois-Pro GUI by setting the "pipe" parameter of Pipeline, and the definition of that model is
       parsed from YAML files and instantiated. */
    void onParamChange(jevois::dnn::pipeline::postproc const & param,
                       jevois::dnn::pipeline::PostProc const & val) override 
    {
      // Override default behavior when a Detect post-processor is desired:
      if (val == jevois::dnn::pipeline::PostProc::Detect)
      {
        // Model your code here to follow that in Pipeline.C:

        // If currently processing async net, wait until done:
        asyncNetWait();

        // Nuke any old post-processor:
        itsPostProcessor.reset(); removeSubComponent("postproc", false);

        // Instantiate our custom post-processor:
        itsPostProcessor = addSubComponent<MyPostProc>("postproc");
        LINFO("Instantiated post-processor of type " << itsPostProcessor->className());
      }
      else
      {
        // Default behavior from base class for other post-processors:
        jevois::dnn::Pipeline::onParamChange(param, val);
      }
    }
    
    // Allow external users to gain access to our post processor
    std::shared_ptr<jevois::dnn::PostProcessor> getPostProcessor()
    { return itsPostProcessor; }
    
    // Allow external users to gain access to our pre processor
    std::shared_ptr<jevois::dnn::PreProcessor> getPreProcessor()
    { return itsPreProcessor; }
};


//! Example of modified DNN module with custom post-processing
/*! This example shows you how to customize the JeVois DNN processing framework. Here, we create a custom post-processor
    that enhances the functionality of PostProcessorDetect by extracting regions of interest for each detected
    object. For this example, we simply compute an edge map for each detection box and display it as an overlay. Other
    processing is possible, for example, detected regions of interest could be sent to another neural network for
    additional processing.

    The main goal of this module is as a tutorial for how to implement custom operations within the JeVois DNN framework
    in C++. The main idea is to create derived classes over Pipeline and PostProcessorDetect.

    @author Laurent Itti

    @displayname Custom DNN
    @videomapping JVUI 0 0 30.0 CropScale=RGB24@1024x576:YUYV 1920 1080 30.0 JeVois CustomDNN
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
class CustomDNN : public jevois::StdModule
{
  public: 
    // ####################################################################################################
    //! Constructor
    // ####################################################################################################
    CustomDNN(std::string const & instance) : jevois::StdModule(instance)
    {
      // Modified here, we use our customized MyPipeline instead of jevois::dnn::Pipeline:
      itsPipeline = addSubComponent<MyPipeline>("pipeline");
    }

    // ####################################################################################################
    //! Virtual destructor for safe inheritance
    // ####################################################################################################
    virtual ~CustomDNN()
    { }

    // ####################################################################################################
    //! Processing function implementation
    // ####################################################################################################
    void doprocess(jevois::InputFrame const & inframe, jevois::RawImage * outimg,
                   jevois::OptGUIhelper * helper, bool idle)
    {
      // Here, we add code to extract regions of interest and draw edge maps:

      // If we have a second (scaled) image, assume this is the one we want to process:
      jevois::RawImage const inimg = inframe.getp();

      // Get a handle to the post-processor and check whether it is of derived type MyPostProc:
      std::shared_ptr<MyPostProc> mpp = std::dynamic_pointer_cast<MyPostProc>(itsPipeline->getPostProcessor());

      // If the dynamic cast succeeded, current pipeline is indeed using a post processor of type MyPostProc. Otherwise,
      // just run the standard processing as in the regular DNN module, and return:
      if ( ! mpp)
      {
        // Selected post-processor not of type MyPostProc, run regular processing:
        itsPipeline->process(inimg, this, outimg, helper, idle);

        return;
      }

      // If we make it here, the selected pipeline is using MyPostProc:

      // In a parallel thread, convert full HD input frame to RGB while the net runs:
      auto fut = jevois::async([&]() { return inframe.getCvRGB(); });  // could also use getCvBGR(), getCvGray(), etc
      
      // Meanwhile, run the network:
      itsPipeline->process(inimg, this, outimg, helper, idle);
      
      // Get the converted frame (may block until ready):
      cv::Mat inhd = fut.get();
      if (jevois::frameNum() % 30 == 0) LINFO("Input frame is " << inhd.cols << 'x' << inhd.rows);
      
      // Get the latest detections:
      std::vector<jevois::ObjDetect> const & detections = mpp->getDetections();
      if (jevois::frameNum() % 30 == 0) LINFO("Got " << detections.size() << " detections");
      
      // As noted above, beware of thread safety. If you need to keep the detections across multiple video frames,
      // make a deep copy of that "detections" vector. Here, it's just a zero-copy reference to the vector internally
      // stored in the post processor. That vector will be overwritten on the next video frame.
      
      // add your code here....
      // full HD input image is in cv::Mat "inhd"
      // object detection boxes are in "detections"
      
      // If you need access to the PreProcessor, which will allow you to do coordinate transforms from full HD image
      // to blob that was sent to the deep net, use itsPipeline->getPreProcessor()
      
      // Here for example, we compute edge maps within each detected object and display them as overlays:
      int i = 0;
      for (jevois::ObjDetect const & d : detections)
      {
        // Get a crop from the full HD image:
        cv::Rect r(d.tlx * inhd.cols / inimg.width,
                   d.tly * inhd.rows / inimg.height,
                   (d.brx - d.tlx) * inhd.cols / inimg.width,
                   (d.bry - d.tly) * inhd.rows / inimg.height);
        cv::Mat roi = inhd(r); // no need to clone() here, cvtColor() below can handle this zero-copy cropping;
        // but you may have to clone if you want to store the roi across multiple video frames.

        // Some ROIs may be empty and openCV does not like that, so skip those:
        if (roi.empty()) continue;
        
        // For demo, compute Canny edges and create an RGBA image that is all transparent except for the edges:
        cv::Mat gray_roi; cv::cvtColor(roi, gray_roi, cv::COLOR_BGR2GRAY);
        cv::Mat edges; cv::Canny(gray_roi, edges, 50, 100, 3);
        cv::Mat chans[4] { edges, edges, edges, edges };
        cv::Mat mask; cv::merge(chans, 4, mask);

        // Anything that is using the helper should be marked as for JEVOIS_PRO only:
#ifdef JEVOIS_PRO
        if (helper) // need this test to support headless mode (no display, no helper)
        {
          // Give a unique name to each edge ROI and display it. Note: if is_overlay is true in the call to drawImage()
          // below, then coordinates for subsequent drawings will remain relative to the origin of the full frame drawn
          // above using drawInputFrame(). If false, then coordinates will shift to be relative to the origin of the
          // last drawn image. Here, we want is_overlay=true because we will be drawing several ROIs (one per detection)
          // and we do not want the origin of the next ROI to be relative to the previous ROI; we want all ROIs to be
          // drawn at specific locations relative to the input frame:
          std::string roi_name = "roi" + std::to_string(i);
          unsigned short rw = mask.cols, rh = mask.rows;
          
          helper->drawImage(roi_name.c_str(), mask, true, r.x, r.y, rw, rh, false /*noalias*/, true /*is_overlay*/);
          
          // Also draw a circle over each ROI to test that drawing works. Because is_overlay was true in drawImage, we
          // need to shift the center of the circle by (r.x, r.y) which is the top-left corner of the last drawn ROI:
          helper->drawCircle(r.x + r.width/2, r.y + r.height/2, std::min(r.width, r.height)/2);
        }
#endif
        
        // Test serial: if you want to send modified messages, perhaps the easiest is to modify the contents of the
        // ObjDetect object (make a copy first as here d is just a const ref):
        sendSerialObjDetImg2D(inhd.cols, inhd.rows, d);
      }
    }
    
    // ####################################################################################################
    //! Processing function, no video output
    // ####################################################################################################
    virtual void process(jevois::InputFrame && inframe) override
    {
      // Same code here as in base DNN module:

      doprocess(inframe, nullptr, nullptr, false);
    }
    
    // ####################################################################################################
    //! Processing function with video output to USB on JeVois-A33
    // ####################################################################################################
    virtual void process(jevois::InputFrame && inframe, jevois::OutputFrame && outframe) override
    {
      // Same code here as in base DNN module:

      // Get the input frame:
      jevois::RawImage const & inimg = inframe.get();
      unsigned int const w = inimg.width, h = inimg.height;

      // Get the output image:
      jevois::RawImage outimg = outframe.get();

      // Input and output sizes and format must match:
      outimg.require("output", w, h, inimg.fmt);

      // Copy input to output:
      jevois::rawimage::paste(inimg, outimg, 0, 0);

      // Process and draw any results (e.g., detected boxes) into outimg:
      doprocess(inframe, &outimg, nullptr, false);

      // Send the output image with our processing results to the host over USB:
      outframe.send();
    }

#ifdef JEVOIS_PRO
    // ####################################################################################################
    //! Processing function with zero-copy and GUI on JeVois-Pro
    // ####################################################################################################
    virtual void process(jevois::InputFrame && inframe, jevois::GUIhelper & helper) override
    {
      // Same code here as in base DNN module:

      // Compute overall frame rate, CPU usage, etc:
      static jevois::Timer timer("main", 20, LOG_DEBUG);
      std::string const & fpscpu = timer.stop();
      timer.start();
      
      // Start the display frame: winw, winh will be set by startFrame() to the display size, e.g., 1920x1080
      unsigned short winw, winh;
      bool idle = helper.startFrame(winw, winh);

      // Display the camera input frame: if all zeros, x, y, w, h will be set by drawInputFrame() so as to show the
      // video frame as large as possible and centered within the display (of size winw,winh)
      int x = 0, y = 0; unsigned short w = 0, h = 0;
      helper.drawInputFrame("c", inframe, x, y, w, h, true);

      // Process and draw any results (e.g., detected boxes) as OpenGL overlay:
      doprocess(inframe, nullptr, &helper, idle);

      // Show overall frame rate, CPU, camera resolution, and display resolution, at bottom of screen:
      helper.iinfo(inframe, fpscpu, winw, winh);
      
      // Render the image and GUI:
      helper.endFrame();
    }
#endif
    
    // ####################################################################################################
  protected:
    // Modified here, we use MyPipeline instead of jevois::dnn::Pipeline:
    std::shared_ptr<MyPipeline> itsPipeline;
};

// Allow the module to be loaded as a shared object (.so) file:
JEVOIS_REGISTER_MODULE(CustomDNN);
