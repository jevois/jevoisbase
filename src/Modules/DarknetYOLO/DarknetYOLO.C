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
#include <jevois/Debug/Profiler.H>
#include <jevois/Image/RawImageOps.H>
#include <opencv2/core/core.hpp>
#include <jevoisbase/Components/ObjectDetection/Yolo.H>

//! Detect multiple objects in scenes using the Darknet YOLO deep neural network
/*! 
    @author Laurent Itti

    @displayname Darknet YOLO
    @videomapping NONE 0 0 0 YUYV 320 240 30.0 JeVois DarknetYOLO
    @videomapping YUYV 320 260 30.0 YUYV 320 240 30.0 JeVois DarknetYOLO
    @videomapping YUYV 640 500 20.0 YUYV 640 480 20.0 JeVois DarknetYOLO
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
class DarknetYOLO : public jevois::Module
{
  public: 
    // ####################################################################################################
    //! Constructor
    // ####################################################################################################
    DarknetYOLO(std::string const & instance) : jevois::Module(instance)
    {
      itsYolo = addSubComponent<Yolo>("yolo");
    }

    // ####################################################################################################
    //! Virtual destructor for safe inheritance
    // ####################################################################################################
    virtual ~DarknetYOLO()
    { }

    // ####################################################################################################
    //! Processing function, no video output
    // ####################################################################################################
    virtual void process(jevois::InputFrame && inframe) override
    {
      // todo
    }

    // ####################################################################################################
    //! Processing function with video output to USB
    // ####################################################################################################
    virtual void process(jevois::InputFrame && inframe, jevois::OutputFrame && outframe) override
    {
      static jevois::Profiler prof("processing", 10, LOG_INFO);

      // Wait for next available camera image:
      jevois::RawImage const inimg = inframe.get();

      prof.start();
      
      // We only handle one specific pixel format, and any image size in this module:
      unsigned int const w = inimg.width, h = inimg.height;
      inimg.require("input", w, h, V4L2_PIX_FMT_YUYV);

      // While we process it, start a thread to wait for out frame and paste the input into it:
      jevois::RawImage outimg;
      auto paste_fut = std::async(std::launch::async, [&]() {
          outimg = outframe.get();
          outimg.require("output", w, h + 60, inimg.fmt);
          jevois::rawimage::paste(inimg, outimg, 0, 0);
          jevois::rawimage::writeText(outimg, "JeVois Darknet YOLO", 3, 3, jevois::yuyv::White);
          jevois::rawimage::drawFilledRect(outimg, 0, h, w, outimg.height-h, jevois::yuyv::Black);
        });

      prof.checkpoint("paste started");
      
      // Convert the image to RGB and process:
      cv::Mat cvimg = jevois::rawimage::convertToCvRGB(inimg);

      prof.checkpoint("converted to rgb");
      
      // Wait for paste to finish up:
      paste_fut.get();

      // Let camera know we are done processing the input image:
      inframe.done();


      prof.checkpoint("paste done");
      
      itsYolo->predict(cvimg);

      prof.checkpoint("predicted");

      itsYolo->drawDetections(outimg);

      // Show processing fps:
      //std::string const & fpscpu = timer.stop();
      //jevois::rawimage::writeText(outimg, fpscpu, 3, h - 13, jevois::yuyv::White);

      prof.checkpoint("draw done");
      
      // Send the output image with our processing results to the host over USB:
      outframe.send();

      prof.stop();
    }

    // ####################################################################################################
  protected:
    std::shared_ptr<Yolo> itsYolo;
 };

// Allow the module to be loaded as a shared object (.so) file:
JEVOIS_REGISTER_MODULE(DarknetYOLO);
