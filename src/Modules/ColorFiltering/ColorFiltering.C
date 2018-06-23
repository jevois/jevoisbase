// ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//
// JeVois Smart Embedded Machine Vision Toolkit - Copyright (C) 2017 by Laurent Itti, the University of Southern
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
#include <jevois/Types/Enum.H>
#include <jevois/Debug/Timer.H>
#include <jevois/Image/RawImageOps.H>

#include <opencv2/imgproc/imgproc.hpp>

#include <jevoisbase/Components/Filters/Filter.H>
#include <jevoisbase/Components/Filters/BlurFilter.H>
#include <jevoisbase/Components/Filters/MedianFilter.H>
#include <jevoisbase/Components/Filters/MorphologyFilter.H>
#include <jevoisbase/Components/Filters/LaplacianFilter.H>
#include <jevoisbase/Components/Filters/BilateralFilter.H>

// icon by Freepik in other at flaticon

static jevois::ParameterCategory const ParamCateg("Color Filtering Parameters");

//! Enum \relates ColorFiltering
JEVOIS_DEFINE_ENUM_CLASS(Effect, (NoEffect) (Blur) (Median) (Morpho) (Laplacian) (Bilateral) );

//! Parameter \relates ColorFiltering
JEVOIS_DECLARE_PARAMETER_WITH_CALLBACK(effect, Effect, "Image processing effect to apply",
                                       Effect::Morpho, Effect_Values, ParamCateg);

//! Image filtering using OpenCV
/*! This module is to learn about basic image filtering. It was developed to allow students to instantly observe the
    effects of different filters and their parameters on live video. The module implements a variety of filters using
    OpenCV. Each filter exposes some paremeters (e.g., kernel size) that can be set interactively to understand their
    effects onto the filter behavior.

    Available filters:

    - Blur: Replace each pixel by the average of all pixels in a box around the pixel of interest. Leads to a
      blur effect on the image, more pronounced with bigger box sizes. See
      http://docs.opencv.org/3.2.0/d4/d86/group__imgproc__filter.html#ga8c45db9afe636703801b0b2e440fce37

    - Median: Replaces each pixel by the median value within a box around that pixel. Tends to blur and remove
      salt-and-peper noise from the
      image. http://docs.opencv.org/3.2.0/d4/d86/group__imgproc__filter.html#ga564869aa33e58769b4469101aac458f9

    - Morpho: mathematical morphology operations, see
      http://docs.opencv.org/3.2.0/d9/d61/tutorial_py_morphological_ops.html for an introduction,
      and http://docs.opencv.org/3.2.0/d4/d86/group__imgproc__filter.html

    - Laplacian: Computes second spatial derivative of the image. Tends to amplify edges and noise. See
      http://docs.opencv.org/3.2.0/d4/d86/group__imgproc__filter.html#gad78703e4c8fe703d479c1860d76429e6

    - Bilateral: bi-lateral filter, very slow, see
      http://homepages.inf.ed.ac.uk/rbf/CVonline/LOCAL_COPIES/MANDUCHI1/Bilateral_Filtering.html and
      http://docs.opencv.org/3.2.0/d4/d86/group__imgproc__filter.html#ga9d7064d478c95d60003cf839430737ed

    How to use this module
    ----------------------

    - Either try it with JeVois Inventor using <b>YUYV 640x240 @@ 30 fps</b>. Note that each time you select a new \p
      effect, this will affect the set of parameters that are available for that effect, but currently JeVois Inventor
      has no way of being notified of that change. So just click to another tab (e.g., the \b Info tab), and then back
      to the \b Parameters tab each time you change the effect. This will refresh the parameter list.

    - Or, open a video viewer on your host computer and select <b>YUYV 640x240 @@ 30 fps</b> (see \ref UserQuick)

    - Open a serial communication to JeVois (see \ref UserCli)

    - Type `help` and observe the available parameter called \p effect

    - Start by setting the \p effect parameter to a given effect type. For example:
      `setpar effect Median` or `setpar effect Morpho` (commands are case-sensitive).

    - Then type \c help to see what additional parameters are available for each effect. For example, for Median, you
      can adjust the kernel size (parameter \p ksize). For Morpho, you can select the type of morphological operation
      (parameter \p op), structuring element shape (parameter \p kshape) and size (parameter \p ksize), etc.

      Complete example:
      \code
      setpar effect Morpho
      help
      setpar op Open
      setpar ksize 7 7
      \endcode

      With \jvversion{1.5} and above, you may want to use `help2` instead of `help`, which is a shorter and more compact
      help message that shows parameters and commands of the running machine vision module only (and no general
      parameters related to the JeVois core).


    @author Laurent Itti

    @videomapping YUYV 640 240 30.0 YUYV 320 240 30.0 JeVois ColorFiltering
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
class ColorFiltering : public jevois::Module,
                       public jevois::Parameter<effect>
{
  public:
    //! Inherited constructor ok
    using jevois::Module::Module;

    //! Virtual destructor for safe inheritance
    virtual ~ColorFiltering()
    { }

    //! Processing function with video output to USB
    virtual void process(jevois::InputFrame && inframe, jevois::OutputFrame && outframe) override
    {
      static jevois::Timer timer("processing", 30, LOG_DEBUG);

      // Wait for next available camera image:
      jevois::RawImage inimg = inframe.get(); unsigned int const w = inimg.width, h = inimg.height;
      inimg.require("input", w, h, V4L2_PIX_FMT_YUYV); // accept any image size but require YUYV pixels

      timer.start();

      // While we process it, start a thread to wait for output frame and paste the input image into it:
      jevois::RawImage outimg; // main thread should not use outimg until paste thread is complete
      auto paste_fut = std::async(std::launch::async, [&]() {
          outimg = outframe.get();
          outimg.require("output", w * 2, h, inimg.fmt);
          jevois::rawimage::paste(inimg, outimg, 0, 0);
          jevois::rawimage::writeText(outimg, "JeVois Input Image", 3, 3, jevois::yuyv::White);
        });

      // Convert source image to BGR:
      cv::Mat src = jevois::rawimage::convertToCvBGR(inimg);

      // Apply the filter (if any):
      cv::Mat dst; std::string settings;
      if (itsFilter) settings = itsFilter->process(src, dst);
      else dst = cv::Mat(h, w, CV_8UC3, cv::Scalar(0,0,0));

      // Wait for paste to finish up:
      paste_fut.get();

      // Paste the results into our output:
      jevois::rawimage::pasteBGRtoYUYV(dst, outimg, w, 0);
                                       
      std::string const & fpscpu = timer.stop();

      // Write a few things:
      std::ostringstream oss; oss << "JeVois image filter: " << effect::get();
      jevois::rawimage::writeText(outimg, oss.str(), w + 3, 3, jevois::yuyv::White);

      std::vector<std::string> svec = jevois::split(settings, "\n");
      for (size_t i = 0; i < svec.size(); ++i)
        jevois::rawimage::writeText(outimg, svec[i], w + 3, 15 + 12 * i, jevois::yuyv::White);

      jevois::rawimage::writeText(outimg, fpscpu, 3, h - 13, jevois::yuyv::White);

      // Send the output image with our processing results to the host over USB:
      outframe.send();
    }

  protected:
    //! Parameter callback: set the selected filter algo
    void onParamChange(effect const & param, Effect const & val)
    {
      if (itsFilter) { removeSubComponent(itsFilter); itsFilter.reset(); }
      
      switch (val)
      {
      case Effect::NoEffect: break;
      case Effect::Blur: itsFilter = addSubComponent<BlurFilter>("filter"); break;
      case Effect::Median: itsFilter = addSubComponent<MedianFilter>("filter"); break;
      case Effect::Morpho: itsFilter = addSubComponent<MorphologyFilter>("filter"); break;
      case Effect::Laplacian: itsFilter = addSubComponent<LaplacianFilter>("filter"); break;
      case Effect::Bilateral: itsFilter = addSubComponent<BilateralFilter>("filter"); break;
      }
    }
    
  private:
    std::shared_ptr<Filter> itsFilter;
};

// Allow the module to be loaded as a shared object (.so) file:
JEVOIS_REGISTER_MODULE(ColorFiltering);
