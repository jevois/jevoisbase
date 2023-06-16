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

// icon from opencv

//! Detect and recognize multiple objects in scenes using OpenCV, NPU, TPU, or VPU Deep Neural Nets
/*! This module runs a deep neural network using the OpenCV #DNN library. Classification networks try to identify the
    whole object or scene in the field of view, and return the top scoring object classes. Detection networks analyze a
    scene and produce a number of bounding boxes around detected objects, together with identity labels and confidence
    scores for each detected box. Semantic segmentation networks create a pixel-by-pixel mask which assigns a class
    label to every location in the camera view.

    To select a network, see parameter \p pipe of component Pipeline.

    The following keys are used in the JeVois-Pro GUI (\p pipe parameter of Pipeline component):

    - **OpenCV:** network loaded by OpenCV #DNN framework and running on CPU.
    - **ORT:**  network loaded by ONNX-Runtime framework and running on CPU.
    - **NPU:** network running native on the JeVois-Pro integrated 5-TOPS NPU (neural processing unit).
    - **SPU:** network running on the optional 26-TOPS Hailo8 SPU accelerator (stream processing unit).
    - **TPU:** network running on the optional 4-TOPS Google Coral TPU accelerator (tensor processing unit).
    - **VPU:** network running on the optional 1-TOPS MyriadX VPU accelerator (vector processing unit).
    - **NPUX:** network loaded by OpenCV and running on NPU via the TIM-VX OpenCV extension. To run efficiently, network
      should have been quantized to int8, otherwise some slow CPU-based emulation will occur.
    - **VPUX:** network optimized for VPU but running on CPU if VPU is not available. Note that VPUX entries are
      automatically created by scanning all VPU entries and changing their target from Myriad to CPU, if a VPU
      accelerator is not detected. If a VPU is detected, then VPU models are listed and VPUX ones are not.
      VPUX emulation runs on the JeVois-Pro CPU using the Arm Compute Library to provide efficient implementation
      of various network layers and operations.

    For expected network speed, see \subpage JeVoisProBenchmarks

    Serial messages
    ---------------

    For classification networks, when object classes are found with confidence scores above \p thresh, a message
    containing up to \p top category:score pairs will be sent per video frame. Exact message format depends on the
    current \p serstyle setting and is described in \ref UserSerialStyle. For example, when \p serstyle is \b Detail,
    this module sends:

    \verbatim
    DO category:score category:score ... category:score
    \endverbatim

    where \a category is a category name (from \p namefile) and \a score is the confidence score from 0.0 to 100.0 that
    this category was recognized. The pairs are in order of decreasing score.

    See \ref UserSerialStyle for more on standardized serial messages, and \ref coordhelpers for more info on
    standardized coordinates.

    For object detection networks, when detections are found which are above threshold, one message will be sent for
    each detected object (i.e., for each box that gets drawn when USB outputs are used), using a standardized 2D
    message:
    + Serial message type: \b 2D
    + `id`: the category of the recognized object, followed by ':' and the confidence score in percent
    + `x`, `y`, or vertices: standardized 2D coordinates of object center or corners
    + `w`, `h`: standardized object size
    + `extra`: any number of additional category:score pairs which had an above-threshold score for that box
    
    See \ref UserSerialStyle for more on standardized serial messages, and \ref coordhelpers for more info on
    standardized coordinates.

    @author Laurent Itti

    @displayname DNN
    @videomapping NONE 0 0 0.0 YUYV 640 480 15.0 JeVois DNN
    @videomapping YUYV 640 498 15.0 YUYV 640 480 15.0 JeVois DNN
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
class DNN : public jevois::StdModule
{
  public: 
    // ####################################################################################################
    //! Constructor
    // ####################################################################################################
    DNN(std::string const & instance) : jevois::StdModule(instance)
    {
      itsPipeline = addSubComponent<jevois::dnn::Pipeline>("pipeline");
    }

    // ####################################################################################################
    //! Virtual destructor for safe inheritance
    // ####################################################################################################
    virtual ~DNN()
    { }

    // ####################################################################################################
    //! Processing function implementation
    // ####################################################################################################
    void doprocess(jevois::InputFrame const & inframe, jevois::RawImage * outimg,
                   jevois::OptGUIhelper * helper, bool idle)
    {
      // If we have a second (scaled) image, assume this is the one we want to process:
      jevois::RawImage const inimg = inframe.getp();

      // Ok, process it:
      itsPipeline->process(inimg, this, outimg, helper, idle);
    }
    
    // ####################################################################################################
    //! Processing function, no video output
    // ####################################################################################################
    virtual void process(jevois::InputFrame && inframe) override
    {
      doprocess(inframe, nullptr, nullptr, false);
    }
    
    // ####################################################################################################
    //! Processing function with video output to USB on JeVois-A33
    // ####################################################################################################
    virtual void process(jevois::InputFrame && inframe, jevois::OutputFrame && outframe) override
    {
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
    std::shared_ptr<jevois::dnn::Pipeline> itsPipeline;
};

// Allow the module to be loaded as a shared object (.so) file:
JEVOIS_REGISTER_MODULE(DNN);
