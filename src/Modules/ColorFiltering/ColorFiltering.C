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

    - On JeVois-Pro, select the effect and set its parameters in the Parameters tab of the GUI. You can also drag the
      cyan handles left and right on screen to change which portion of the image is shown as original vs processed.

    - On JeVois-A33, either try it with JeVois Inventor using <b>YUYV 640x240 @@ 30 fps</b>. Note that each time you
      select a new \p effect, this will affect the set of parameters that are available for that effect, but currently
      JeVois Inventor has no way of being notified of that change. So just click to another tab (e.g., the \b Info tab),
      and then back to the \b Parameters tab each time you change the effect. This will refresh the parameter list.

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
    // ####################################################################################################
    //! Inherited constructor ok
    // ####################################################################################################
    using jevois::Module::Module;

    // ####################################################################################################
    //! Virtual destructor for safe inheritance
    // ####################################################################################################
    virtual ~ColorFiltering()
    { }

    // ####################################################################################################
    //! Processing function with video output to USB
    // ####################################################################################################
    virtual void process(jevois::InputFrame && inframe, jevois::OutputFrame && outframe) override
    {
      static jevois::Timer timer("processing", 30, LOG_DEBUG);

      // Wait for next available camera image:
      jevois::RawImage inimg = inframe.get(); unsigned int const w = inimg.width, h = inimg.height;
      inimg.require("input", w, h, V4L2_PIX_FMT_YUYV); // accept any image size but require YUYV pixels

      timer.start();

      // While we process it, start a thread to wait for output frame and paste the input image into it:
      jevois::RawImage outimg; // main thread should not use outimg until paste thread is complete
      auto paste_fut = jevois::async([&]() {
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
      jevois::rawimage::writeText(outimg, "JeVois image filter: " + effect::strget(), w + 3, 3, jevois::yuyv::White);

      std::vector<std::string> svec = jevois::split(settings, "\n");
      for (size_t i = 0; i < svec.size(); ++i)
        jevois::rawimage::writeText(outimg, svec[i], w + 3, 15 + 12 * i, jevois::yuyv::White);

      jevois::rawimage::writeText(outimg, fpscpu, 3, h - 13, jevois::yuyv::White);

      // Send the output image with our processing results to the host over USB:
      outframe.send();
    }

#ifdef JEVOIS_PRO
    // ####################################################################################################
    //! Processing function with zero-copy and GUI on JeVois-Pro
    // ####################################################################################################
    virtual void process(jevois::InputFrame && inframe, jevois::GUIhelper & helper) override
    {
      static jevois::Timer timer("processing", 100, LOG_DEBUG);
      static int left = -1, right = -1;
      
      // Start the GUI frame:
      unsigned short winw, winh;
      helper.startFrame(winw, winh);

      // Draw the camera frame:
      int x = 0, y = 0; unsigned short iw = 0, ih = 0;
      helper.drawInputFrame("camera", inframe, x, y, iw, ih);

      helper.itext("JeVois-Pro Color Filtering");
      
      timer.start();

      // Wait for next available camera image:
      cv::Mat src = inframe.getCvRGBp(); // should be BGR but RGB is faster and should not affect processing...
      int const sw = src.cols; int const sh = src.rows;
      inframe.done();

      // Apply the filter (if any):
      cv::Mat dst; std::string settings;
      if (itsFilter) settings = itsFilter->process(src, dst);
      else dst = cv::Mat(sh, sw, CV_8UC3, cv::Scalar(0,0,0));

      // We will display the processed image as a partial overlay between two vertical lines defined by 'left' and
      // 'right', where each also has a mouse drag handle (small square).
      
      ImU32 const col = IM_COL32(128,255,255,255); // color of the vertical lines and square handles
      int const siz = 20; // size of the square handles, in image pixels
      static bool dragleft = false, dragright = false;

      // Initialize the handles at 1/4 and 3/4 of image width on first video frame after module is loaded:
      if (jevois::frameNum() == 0) { left = sw / 4; right = 3 * sw / 4; }

      // Make sure the handles do not overlap and/or get out of the image bounds:
      if (left > right - siz) { if (dragright) left = right - siz; else right = left + siz; }
      left = std::max(siz, std::min(sw - siz * 2, left));
      right = std::max(left + siz, std::min(sw - siz, right));

      // Mask and draw the overlay.  To achieve this, we convert the whole result image to RGBA and then assign a zero
      // alpha channel to all pixels to the left of the 'left' bound and to the right of the 'right' bound:
      cv::Mat ovl; cv::cvtColor(dst, ovl, cv::COLOR_RGB2RGBA);
      ovl(cv::Rect(0, 0, left, sh)) = 0; // make left side transparent
      ovl(cv::Rect(right, 0, sw-right, sh)) = 0; // make right side transparent
      int ox = 0, oy = 0; unsigned short ow = 0, oh = 0;
      helper.drawImage("filtered", ovl, true, ox, oy, ow, oh, false, true /* overlay */);

      helper.drawLine(left, 0, left, sh, col);
      helper.drawRect(left-siz, (sh-siz)/2, left, (sh+siz)/2, col);
      helper.drawLine(right, 0, right, sh, col);
      helper.drawRect(right, (sh-siz)/2, right+siz, (sh+siz)/2, col);

      // Adjust the left and right handles if they get clicked and dragged:
      ImVec2 const ip = helper.d2i(ImGui::GetMousePos());

      if (ImGui::IsMouseClicked(0))
      {
        // Are we clicking on the left or right handle?
        if (ip.x > left-siz && ip.x < left && ip.y > (sh-siz)/2 && ip.y < (sh+siz)/2) dragleft = true;
        if (ip.x > right && ip.x < right+siz && ip.y > (sh-siz)/2 && ip.y < (sh+siz)/2) dragright = true;
      }

      if (ImGui::IsMouseDragging(0))
      {
        if (dragleft) left = ip.x + 0.5F * siz;
        if (dragright) right = ip.x - 0.5F * siz;
        // We will enforce validity of left and right on next frame.
      }

      if (ImGui::IsMouseReleased(0))
      {
        dragleft = false;
        dragright = false;
      }

      // Write a few things:
      std::string const & fpscpu = timer.stop();
      helper.itext("JeVois image filter: " + effect::strget());
      std::vector<std::string> svec = jevois::split(settings, "\n");
      for (std::string const & s : svec) helper.itext(s);
      helper.iinfo(inframe, fpscpu, winw, winh);

      // Render the image and GUI:
      helper.endFrame();
     }
#endif

    // ####################################################################################################
  protected:
    //! Parameter callback: set the selected filter algo
    void onParamChange(effect const & /*param*/, Effect const & val) override
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
