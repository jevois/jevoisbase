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
// Contact information: Shixian Wen - 3641 Watt Way, HNB-10A - Los Angeles, BA 90089-2520 - USA.
// Tel: +1 213 740 3527 - shixianw@usc.edu - http://iLab.usc.edu - http://jevois.org
// ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/*! \file */

#include <jevois/Core/Module.H>
#include <jevois/Debug/Timer.H>
#include <jevois/Image/RawImageOps.H>
#include <jevois/Util/Coordinates.H>
#include <jevoisbase/Components/ARtoolkit/ARtoolkit.H>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

//! Augmented reality markers using ARtoolkit
/*! Detect and decode patterns known as ARtoolkit markers, which are small 2D barcodes often used in augmented
    reality and robotics.

    @author Shixian Wen

    @displayname Demo ARtoolkit
    @videomapping NONE 0 0 0 YUYV 320 240 30.0 JeVois DemoARtoolkit
    @videomapping YUYV 320 262 30.0 YUYV 320 240 30.0 JeVois DemoARtoolkit
    @videomapping YUYV 640 502 20.0 YUYV 640 480 20.0 JeVois DemoARtoolkit
    @email shixianw\@usc.edu
    @address University of Southern California, HNB-10A, 3641 Watt Way, Los Angeles, CA 90089-2520, USA
    @copyright Copyright (C) 2017 by Shixian Wen, iLab and the University of Southern California
    @mainurl http://jevois.org
    @supporturl http://jevois.org/doc
    @otherurl http://iLab.usc.edu
    @license GPL v3
    @distribution Unrestricted
    @restrictions None
    \ingroup modules */
class DemoARtoolkit :public jevois::Module
{
  public:
    
    DemoARtoolkit(std::string const & instance) : jevois::Module(instance)
    {
      itsARtoolkit = addSubComponent<ARtoolkit>("artoolkit");
    }

    virtual ~DemoARtoolkit()
    { }

    virtual void process(jevois::InputFrame && inframe, jevois::OutputFrame && outframe) override
    {
      // Wait for next available camera image:
      jevois::RawImage const inimg = inframe.get();
      if (!enterflag)
      {
        //when first enter the process mannualy init the itsARtoolkit component
        enterflag = true;
        unsigned int const w = inimg.width, h = inimg.height;
        itsARtoolkit->xsize::set(w);
        itsARtoolkit->ysize::set(h);
        itsARtoolkit->manualinit();
      }

      // Convert the image to BGR and process:
      cv::Mat imgbgr = jevois::rawimage::convertToCvBGR(inimg);
      inframe.done();

      jevois::RawImage outimg = outframe.get();
      outimg.require("output", inimg.width, inimg.height, outimg.fmt);

      itsARtoolkit->detectMarkers(imgbgr);
      jevois::rawimage::convertCvBGRtoRawImage(imgbgr, outimg, 75 /* JPEG quality */);
      outframe.send();
    }

  protected:
    std::shared_ptr<ARtoolkit> itsARtoolkit;
    bool enterflag = false; //!< a flag to initialize itsARtoolkit
};

// Allow the module to be loaded as a shared object (.so) file:
JEVOIS_REGISTER_MODULE(DemoARtoolkit);
