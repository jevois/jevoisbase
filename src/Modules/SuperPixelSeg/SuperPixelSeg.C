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
#include <jevois/Debug/Log.H>
#include <jevois/Util/Utils.H>
#include <jevois/Image/RawImageOps.H>
#include <jevois/Debug/Timer.H>
#include <jevois/Types/Enum.H>

#include <jevoisbase/Components/ImageProc/SuperPixel.H>

#include <linux/videodev2.h>

// icon by Freepik in interface at flaticon

//! Segment an image using super-pixels
/*! Segment an image into regions with somewhat uniform appearance, using the SuperPixel algorithms of OpenCV.

    The result is an image with a reduced number of grey levels, where each grey level represents the label or ID of a
    given region (so, all pixels with a given grey value belong to a given region in the image).

    How to use this module
    ----------------------

    When you run this algorithm and obtain a low number of clusters, those may not be well visible to the human eye. For
    example, if you end up with 10 clusters, they will take grayscale values 0 to 9, which all look completely black
    on a standard monitor where 0 is black and 255 is white. To get started with this module, you may hence want to
    change the parameters a bit. For example:

    \verbatim
    setpar algo SEEDS
    setpar output Labels
    setpar numpix 255
    \endverbatim
    and you should now see something.

    To work with a smaller number of super-pixels, you would usually want to first create some software that runs on the
    host computer and which will grab the greyscale frames from JeVois, then will assign colors to the regions somehow,
    and finally will display the superpixels in color. For testing, you may want to just capture and save some frames
    from JeVois (which may look all black) and then use some paint program to change color 0, color 1, etc to more
    visible colors than very similar shades of black.

    @author Laurent Itti

    @videomapping GREY 320 240 30.0 YUYV 320 240 30.0 JeVois SuperPixelSeg
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
class SuperPixelSeg : public jevois::Module
{
  public:
    //! Constructor
    SuperPixelSeg(std::string const & instance) : jevois::Module(instance)
    { itsSuperPixel = addSubComponent<SuperPixel>("superpixel"); }
    
    //! Virtual destructor for safe inheritance
    virtual ~SuperPixelSeg() { }

    //! Processing function
    virtual void process(jevois::InputFrame && inframe, jevois::OutputFrame && outframe) override
    {
      static jevois::Timer timer("processing");

      // Wait for next available camera image:
      jevois::RawImage inimg = inframe.get(); unsigned int const w = inimg.width, h = inimg.height;
      inimg.require("input", w, h, V4L2_PIX_FMT_YUYV); // any image size but require YUYV pixels

      timer.start();

      // Convert to openCV RGB:
      cv::Mat cvimg = jevois::rawimage::convertToCvRGB(inimg);

      // Let camera know we are done processing the input image:
      inframe.done();

      // Get the output image and require size dims as input but grey pixels:
      jevois::RawImage outimg = outframe.get();
      outimg.require("output", w, h, V4L2_PIX_FMT_GREY);

      // Process ths input and get output:
      cv::Mat grayimg = jevois::rawimage::cvImage(outimg); // grayimg's pixels point to outimg's, no copy
      itsSuperPixel->process(cvimg, grayimg);

      // All done:
      timer.stop();
      
      // Send the output image with our processing results to the host over USB:
      outframe.send();
    }

  private:
    std::shared_ptr<SuperPixel> itsSuperPixel;
};

// Allow the module to be loaded as a shared object (.so) file:
JEVOIS_REGISTER_MODULE(SuperPixelSeg);
