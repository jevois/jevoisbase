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

#include <linux/videodev2.h>
#include <jevoisbase/Components/ObjectMatcher/ObjectMatcher.H>
#include <opencv2/imgcodecs.hpp>

#include <cstdio> // for std::remove

// icon by Vectors Market in arrows at flaticon

static jevois::ParameterCategory const ParamCateg("Object Detection Options");

//! Define a pair of floats, to avoid macro parsing problems when used as a parameter:
typedef std::pair<float, float> floatpair;

//! Parameter \relates ObjectDetect
JEVOIS_DECLARE_PARAMETER_WITH_CALLBACK(win, floatpair,
                                       "Width and height (in percent of image size, with valid percentages between "
                                       "10.0 and 100.0) of the window used to interactively save objects",
                                       floatpair(50.0F, 50.0F), ParamCateg);

//! Parameter \relates ObjectDetect
JEVOIS_DECLARE_PARAMETER(showwin, bool, "Show the interactive image capture window when true", false, ParamCateg);

//! Simple object detection using keypoint matching
/*! This module finds objects by matching keypoint descriptors between the current image and a set of training
    images. Here we use SURF keypoints and descriptors as provided by OpenCV.

    The algorithm consists of 4 phases:
    - detect keypoint locations, typically corners or other distinctive texture elements or markings;
    - compute keypoint descriptors, which are summary representations of the image neighborhood around each keypoint;
    - match descriptors from current image to descriptors previously extracted from training images;
    - if enough matches are found between the current image and a given training image, and they are of good enough
      quality, compute the homography (geometric transformation) between keypoint locations in that training image and
      locations of the matching keypoints in the current image. If it is well conditioned (i.e., a 3D viewpoint change
      could well explain how the keypoints moved between the training and current images), declare that a match was
      found, and draw a green rectangle around the detected object.

    The algorithm comes by default with one training image, for the Priority Mail logo of the U.S. Postal
    Service. Search for "USPS priority mail" on the web and point JeVois to a picture of the logo on your screen to
    recognize it. See the screenshots of this module for examples of how that logo looks.

    Offline training
    ----------------

    Simply add images of the objects you want to detect in <b>JEVOIS:/modules/JeVois/ObjectDetect/images/</b> on your
    JeVois microSD card. Those will be processed when the module starts. The names of recognized objects returned by
    this module are simply the file names of the pictures you have added in that directory. No additional training
    procedure is needed. Beware that the more images you add, the slower the algorithm will run, and the higher your
    chances of confusions among several of your objects.

    With \jvversion{1.1} or later, you do not need to eject the microSD from JeVois, and you can instead add images live
    by exporting the microSD inside JeVois using the \c usbsd command. See \ref MicroSD (last section) for details. When
    you are done adding new images or deleting unwanted ones, properly eject the virtual USB flash drive, and JeVois
    will restart and load the new training data.

    Live training
    -------------

    With \jvversion{1.2} and later you can train this algorithm live by telling JeVois to capture and save an image of
    an object, which can be used later to identify this object again.

    First, enable display of a training window using:
    \verbatim
    setpar showwin true
    \endverbatim

    You should now see a gray rectangle. You can adjust the window size and aspect ratio using the \p win parameter. By
    default, the algorithm will train new objects that occupy half the width and height of the camera image.
    
    Point your JeVois camera to a clean view of an object you want to learn (if possible, with a blank, featureless
    background, as this algorithm does not attempt to segment objects and would otherwise also learn features of the
    background as part of the object). Make sure the objects fits inside the gray rectangle and fills as much of it as
    possible. You should adjust the distance between the object and the camera, and the grey rectangle, to roughly match
    the distance at which you want to detect that object in the future. Then issue the command:

    \verbatim
    save somename
    \endverbatim

    over a serial connection to JeVois, where \a somename is the name you want to give to this object. This will grab
    the current camera image, crop it using the gray rectangle, and save the crop as a new training image
    <b>somename.png</b> for immediate use. The algorithm will immediately re-train on all objects, including the new
    one. You should see the object being detected shortly after you send your save command. Note that we save the image
    as grayscale since this algorithm does not use color anyway.

    You can see the list of current images by using command:

    \verbatim
    list
    \endverbatim

    Finally, you can delete an image using command:

    \verbatim
    del somename
    \endverbatim

    where \a somename is the object name without extension, and a .png extension will be added. The image will
    immediately be deleted and that object will not be recognized anymore.

    For more information, see JeVois tutorial [Live training of the Object Detection
    module](http://jevois.org/tutorials/UserObjectDetect.html) and the associated video:

    \youtube{qwJOcsbkZLE}

    Serial Messages
    ---------------

    This module can send standardized serial messages as described in \ref UserSerialStyle. One message is issued on
    every video frame for the best detected object (highest score).

    - Serial message type: \b 2D
    - `id`: filename of the recognized object
    - `x`, `y`: standardized 2D coordinates of the object center
    - `w`, `h`, or vertices: Standardized bounding box around the object
    - `extra`: none (empty string)

    See \ref UserSerialStyle for more on standardized serial messages, and \ref coordhelpers for more info on
    standardized coordinates.

    Programmer notes
    ----------------

    This algorithm is quite slow. So, here, we alternate between computing keypoints and descriptors on one frame (or
    more, depending on how slow that gets), and doing the matching on the next frame. This module also provides an
    example of letting some computation happen even after we exit the `process()` function. Here, we keep detecting
    keypoints and computing descriptors even outside `process()`. The itsKPfut future is our handle to that thread, and
    we also use it to alternate between detection and matching on alternating frames.


    @author Laurent Itti

    @videomapping YUYV 320 252 30.0 YUYV 320 240 30.0 JeVois ObjectDetect
    @modulecommand list - show current list of training images
    @modulecommand save somename - grab current frame and save as new training image somename.png
    @modulecommand del somename - delete training image somename.png
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
class ObjectDetect : public jevois::StdModule,
                     public jevois::Parameter<win, showwin>
{
  public:
    // ####################################################################################################
    //! Constructor
    // ####################################################################################################
    ObjectDetect(std::string const & instance) : jevois::StdModule(instance), itsDist(1.0e30)
    { itsMatcher = addSubComponent<ObjectMatcher>("surf"); }

    // ####################################################################################################
    //! Virtual destructor for safe inheritance
    // ####################################################################################################
    virtual ~ObjectDetect() { }

    // ####################################################################################################
    //! Parameter callback
    // ####################################################################################################
    void onParamChange(win const & JEVOIS_UNUSED_PARAM(param), floatpair const & newval)
    {
      // Just check that the values are valid here. They will get stored in our param and used later:
      if (newval.first < 10.0F || newval.first > 100.0F || newval.second < 10.0F || newval.second > 100.0F)
        throw std::range_error("Invalid window percentage values, must be between 10.0 and 100.0");
    }

    // ####################################################################################################
    //! Processing function with no USB output
    // ####################################################################################################
    virtual void process(jevois::InputFrame && inframe) override
    {
      static jevois::Timer timer("processing", 100, LOG_DEBUG);

      // Wait for next available camera image. Any resolution and format ok, we just convert to grayscale:
      itsGrayImg = inframe.getCvGRAY();

      timer.start();

      // Compute keypoints and descriptors, then match descriptors to our training images:
      itsDist = itsMatcher->process(itsGrayImg, itsTrainIdx, itsCorners);

      // Send message about object if a good one was found:
      if (itsDist < 100.0 && itsCorners.size() == 4)
	sendSerialContour2D(itsGrayImg.cols, itsGrayImg.rows, itsCorners, itsMatcher->traindata(itsTrainIdx).name);

      // Show processing fps to log:
      timer.stop();
    }

    // ####################################################################################################
    //! Processing function with USB output
    // ####################################################################################################
    virtual void process(jevois::InputFrame && inframe, jevois::OutputFrame && outframe) override
    {
      static jevois::Timer timer("processing", 100, LOG_DEBUG);

      // Wait for next available camera image. Any resolution ok, but require YUYV since we assume it for drawings:
      jevois::RawImage inimg = inframe.get(); unsigned int const w = inimg.width, h = inimg.height;
      inimg.require("input", w, h, V4L2_PIX_FMT_YUYV);

      timer.start();

      // While we process it, start a thread to wait for output frame and paste the input image into it:
      jevois::RawImage outimg; // main thread should not use outimg until paste thread is complete
      auto paste_fut = std::async(std::launch::async, [&]() {
          outimg = outframe.get();
          outimg.require("output", w, h + 12, inimg.fmt);
          jevois::rawimage::paste(inimg, outimg, 0, 0);
          jevois::rawimage::writeText(outimg, "JeVois SURF Object Detection Demo", 3, 3, jevois::yuyv::White);
          jevois::rawimage::drawFilledRect(outimg, 0, h, w, outimg.height-h, 0x8000);
        });

      // Decide what to do on this frame depending on itsKPfut: if it is valid, we have been computing some new
      // keypoints and descriptors and we should match them now if that computation is finished. If it is not finished,
      // we will just skip this frame and only do some drawings on it while we wait some more. If we have not been
      // computing keypoints and descriptors, that means we did some matching on the last frame, so start computing a
      // new set of keypoints and descriptors now:
      if (itsKPfut.valid())
      {
        // Are we finished yet with computing the keypoints and descriptors?
        if (itsKPfut.wait_for(std::chrono::milliseconds(2)) == std::future_status::ready)
        {
          // Do a get() on our future to free up the async thread and get any exception it might have thrown:
          itsKPfut.get();

          // Match descriptors to our training images:
          itsDist = itsMatcher->match(itsKeypoints, itsDescriptors, itsTrainIdx, itsCorners);
        }

        // Future is not ready, do nothing except drawings on this frame and we will try again on the next one...
      }
      else
      {
        // Convert input image to greyscale:
        itsGrayImg = jevois::rawimage::convertToCvGray(inimg);

        // Start a thread that will compute keypoints and descriptors:
        itsKPfut = std::async(std::launch::async, [&]() {
            itsMatcher->detect(itsGrayImg, itsKeypoints);
            itsMatcher->compute(itsGrayImg, itsKeypoints, itsDescriptors);
          });
      }
      
      // Wait for paste to finish up:
      paste_fut.get();

      // Let camera know we are done processing the input image:
      inframe.done();

      // Draw object if one was found (note: given the flip-flop above, drawing locations only get updated at half the
      // frame rate, i.e., we draw the same thing on two successive video frames):
      if (itsDist < 100.0 && itsCorners.size() == 4)
      {
        jevois::rawimage::drawLine(outimg, int(itsCorners[0].x + 0.499F), int(itsCorners[0].y + 0.499F),
                                   int(itsCorners[1].x + 0.499F), int(itsCorners[1].y + 0.499F),
                                   2, jevois::yuyv::LightGreen);
        jevois::rawimage::drawLine(outimg, int(itsCorners[1].x + 0.499F), int(itsCorners[1].y + 0.499F),
                                   int(itsCorners[2].x + 0.499F), int(itsCorners[2].y + 0.499F), 2,
                                   jevois::yuyv::LightGreen);
        jevois::rawimage::drawLine(outimg, int(itsCorners[2].x + 0.499F), int(itsCorners[2].y + 0.499F),
                                   int(itsCorners[3].x + 0.499F), int(itsCorners[3].y + 0.499F), 2,
                                   jevois::yuyv::LightGreen);
        jevois::rawimage::drawLine(outimg, int(itsCorners[3].x + 0.499F), int(itsCorners[3].y + 0.499F),
                                   int(itsCorners[0].x + 0.499F), int(itsCorners[0].y + 0.499F), 2,
                                   jevois::yuyv::LightGreen);
        jevois::rawimage::writeText(outimg, std::string("Detected: ") + itsMatcher->traindata(itsTrainIdx).name +
                                    " avg distance " + std::to_string(itsDist), 3, h + 1, jevois::yuyv::White);

        sendSerialContour2D(w, h, itsCorners, itsMatcher->traindata(itsTrainIdx).name);
      }

      // Show capture window if desired:
      if (showwin::get())
      {
        floatpair const wi = win::get();
        int const ww = (wi.first * 0.01F) * w, wh = (wi.second * 0.01F) * h;
        jevois::rawimage::drawRect(outimg, (w - ww) / 2, (h - wh) / 2, ww, wh, 1, jevois::yuyv::MedGrey);
      }

      // Show processing fps:
      std::string const & fpscpu = timer.stop();
      jevois::rawimage::writeText(outimg, fpscpu, 3, h - 13, jevois::yuyv::White);
      
      // Send the output image with our processing results to the host over USB:
      outframe.send();
    }

    // ####################################################################################################
    //! Receive a string from a serial port which contains a user command
    // ####################################################################################################
    void parseSerial(std::string const & str, std::shared_ptr<jevois::UserInterface> s) override
    {
      std::vector<std::string> tok = jevois::split(str);
      if (tok.empty()) throw std::runtime_error("Unsupported empty module command");
      std::string const dirname = absolutePath(itsMatcher->traindir::get());

      if (tok[0] == "save")
      {
        if (tok.size() == 1) throw std::runtime_error("save command requires one <name> argument");
        
        // Crop itsGrayImg using the desired window:
        floatpair const wi = win::get();
        int ww = (wi.first * 0.01F) * itsGrayImg.cols;
        int wh = (wi.second * 0.01F) * itsGrayImg.rows;
        cv::Rect cr( (itsGrayImg.cols - ww) / 2, (itsGrayImg.rows - wh) / 2, ww, wh);
      
        // Save it:     
        cv::imwrite(dirname + '/' + tok[1] + ".png", itsGrayImg(cr));
        s->writeString(tok[1] + ".png saved and trained.");
      }
      else if (tok[0] == "del")
      {
        if (tok.size() == 1) throw std::runtime_error("del command requires one <name> argument");
        if (std::remove((dirname + '/' + tok[1] + ".png").c_str()))
          throw std::runtime_error("Failed to delete " + tok[1] + ".png");
        s->writeString(tok[1] + ".png deleted and forgotten.");
      }
      else if (tok[0] == "list")
      {
        std::string lst = jevois::system("/bin/ls \"" + dirname + '\"');
        std::vector<std::string> files = jevois::split(lst, "\\n");
        for (std::string const & f : files) s->writeString(f);
        return;
      }
      else throw std::runtime_error("Unsupported module command [" + str + ']');

      // If we get here, we had a successful save or del. We need to nuke our matcher and re-load it to retrain:
      // First, wait until our component is not computing anymore:
      try { if (itsKPfut.valid()) itsKPfut.get(); } catch (...) { }

      // Detach the sub:
      removeSubComponent(itsMatcher);

      // Nuke it:
      itsMatcher.reset();

      // Nuke any other old data:
      itsKeypoints.clear(); itsDescriptors = cv::Mat(); itsDist = 1.0e30; itsCorners.clear();
      
      // Instantiate a new one, it will load the training data:
      itsMatcher = addSubComponent<ObjectMatcher>("surf");
    }

    // ####################################################################################################
    //! Human-readable description of this Module's supported custom commands
    // ####################################################################################################
    void supportedCommands(std::ostream & os) override
    {
      os << "list - show current list of training images" << std::endl;
      os << "save <somename> - grab current frame and save as new training image <somename>.png" << std::endl;
      os << "del <somename> - delete training image <somename>.png" << std::endl;
    }
    
  private:
    std::shared_ptr<ObjectMatcher> itsMatcher;
    std::future<void> itsKPfut;
    cv::Mat itsGrayImg;
    std::vector<cv::KeyPoint> itsKeypoints;
    cv::Mat itsDescriptors;
    size_t itsTrainIdx;
    double itsDist;
    std::vector<cv::Point2f> itsCorners;
};

// Allow the module to be loaded as a shared object (.so) file:
JEVOIS_REGISTER_MODULE(ObjectDetect);
