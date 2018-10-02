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
#include <jevois/Debug/Timer.H>
#include <jevois/Image/RawImageOps.H>
#include <jevoisbase/Components/Saliency/Saliency.H>
#include <jevoisbase/Components/FaceDetection/FaceDetector.H>
#include <jevoisbase/Components/ObjectRecognition/ObjectRecognitionMNIST.H>
#include <jevoisbase/Components/Tracking/Kalman2D.H>
#include <jevoisbase/Components/Utilities/BufferedVideoReader.H>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <linux/videodev2.h> // for v4l2 pixel types
//#include <opencv2/highgui/highgui.hpp> // used for debugging only, see imshow below

// icon by Freepik in interface at flaticon

struct ScriptItem { char const * msg; int blinkx, blinky; };
static ScriptItem const TheScript[] = {
  { "Hello! Welcome to this simple demonstration of JeVois", 0, 0 },
  { "JeVois = camera sensor + quad-core processor + USB video output", 0, 0 },
  { "This demo is running on the small processor inside JeVois", 0, 0 },
  { "Neat, isn't it?", 0, 0 },
  { "", 0, 0 },
  { "We will help you discover what you see on this screen", 0, 0 },
  { "We will use this blinking marker to point at things:", 600, 335 },
  { "", 0, 0 },
  { "Now a brief tutorial...", 0, 0 },
  { "", 0, 0 },
  { "This demo: Attention + Gist + Faces + Objects", 0, 0 },
  { "Attention: detect things that catch the human eye", 0, 0 },
  { "Pink square in video above: most interesting (salient) location", 0, 0 },
  { "Green circle in video above: smoothed attention trajectory", 0, 0 },
  { "Try it: wave at JeVois, show it some objects, move it around", 0, 0 },
  { "", 0, 0 },
  { "Did you catch the attention of JeVois?", 0, 0 },
  { "", 0, 0 },
  { "Attention is guided by color contrast, ...", 40, 270 },
  { "by luminance (intensity) contrast, ...", 120, 270 },
  { "by oriented edges, ...", 200, 270 },
  { "by flickering or blinking lights, ...", 280, 270 },
  { "and by moving objects.", 360, 270 },
  { "All these visual cues combine into a measure of saliency", 480, 120 },
  { "or visual interest for every location in view.", 480, 120 },
  { "", 0, 0 },
  { "", 0, 0 },
  { "Gist: statistical summary of a scene, also based on ...", 440, 270 },
  { "color, intensity, orientation, flicker and motion features.", 0, 0 },
  { "Gist can be used to recognize places, such as a kitchen or ...", 0, 0 },
  { "a bathroom, or a road turning left versus turning right.", 0, 0 },
  { "Try it: point JeVois to different things and see gist change", 440, 270 },
  { "", 0, 0 },
  { "", 0, 0 },
  { "Face detection finds human faces in the camera's view", 612, 316 },
  { "Try it: point JeVois towards a face. Adjust distance until ...", 612, 316 },
  { "the face fits inside the attention pink square. When a face ...", 0, 0 },
  { "is detected, it will appear in the bottom-right corner.", 612, 316 },
  { "You may have to move a bit farther than arm's length for ...", 0, 0 },
  { "your face to fit inside the attention pink square.", 0, 0 },
  { "", 0, 0 },
  { "", 0, 0 },
  { "Objects: Here we recognize handwritten digits using ...", 525, 316 },
  { "deep neural networks. Try it! Draw a number on paper ...", 0, 0 },
  { "and point JeVois towards it. Adjust distance until the", 0, 0 },
  { "number fits in the attention pink square.", 0, 0 },
  { "", 0, 0 },
  { "Recognized digits are shown near the detected faces.", 525, 316 },
  { "", 0, 0 },
  { "If your number is too small, too big, or not upright ...", 0, 0 },
  { "keep adjusting the distance and angle of the camera.", 0, 0 },
  { "", 0, 0 },
  { "Recognition scores for digits 0 to 9 are shown above.", 464, 310 },
  { "Sometimes the neural network makes mistakes and thinks it ...", 0, 0 },
  { "found a digit when actually it is looking at something else.", 0, 0 },
  { "This is still a research issue", 0, 0 },
  { "but machine vision is improving fast, so stay tuned!", 0, 0 },
  { "", 0, 0 },
  { "With JeVois the future of machine vision is in your hands.", 0, 0 },
  { "", 0, 0 },
  { "", 0, 0 },
  { "", 0, 0 },
  { "This tutorial is now complete. It will restart.", 0, 0 },
  { "", 0, 0 },
  { nullptr, 0, 0 }
};

//! Simple introduction to JeVois and demo that combines saliency, gist, face detection, and object recognition
/*! This module plays an introduction movie, and then launches the equivalent of the \jvmod{DemoSalGistFaceObj} module,
    but with some added text messages that explain what is going on, on the screen.

    Try it and follow the instructions on screen!

    @author Laurent Itti

    @displayname JeVois Intro
    @videomapping YUYV 640 360 50.0 YUYV 320 240 50.0 JeVois JeVoisIntro
    @videomapping YUYV 640 480 50.0 YUYV 320 240 50.0 JeVois JeVoisIntro
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
class JeVoisIntro : public jevois::StdModule
{
  public:
    //! Constructor
    JeVoisIntro(std::string const & instance) : jevois::StdModule(instance), itsScoresStr(" ")
    {
      itsSaliency = addSubComponent<Saliency>("saliency");
      itsFaceDetector = addSubComponent<FaceDetector>("facedetect");
      itsObjectRecognition = addSubComponent<ObjectRecognitionMNIST>("MNIST");
      itsKF = addSubComponent<Kalman2D>("kalman");
      itsVideo = addSubComponent<BufferedVideoReader>("intromovie");
    }

    //! Virtual destructor for safe inheritance
    virtual ~JeVoisIntro() { }

    //! Initialization once parameters are set:
    virtual void postInit() override
    {
      // Read the banner image and convert to YUYV RawImage:
      cv::Mat banner_bgr = cv::imread(absolutePath("jevois-banner-notext.png"));
      itsBanner.width = banner_bgr.cols;
      itsBanner.height = banner_bgr.rows;
      itsBanner.fmt = V4L2_PIX_FMT_YUYV;
      itsBanner.bufindex = 0;
      itsBanner.buf.reset(new jevois::VideoBuf(-1, itsBanner.bytesize(), 0));
      jevois::rawimage::convertCvBGRtoRawImage(banner_bgr, itsBanner, 100);

      // Allow our movie to load a bit:
      std::this_thread::sleep_for(std::chrono::milliseconds(750));
    }
    
    //! Processing function
    virtual void process(jevois::InputFrame && inframe, jevois::OutputFrame && outframe) override
    {
      static jevois::Timer itsProcessingTimer("Processing");
      static cv::Mat itsLastFace(60, 60, CV_8UC2, 0x80aa) ; // Note that this one will contain raw YUV pixels
      static cv::Mat itsLastObject(60, 60, CV_8UC2, 0x80aa) ; // Note that this one will contain raw YUV pixels
      static std::string itsLastObjectCateg;
      static bool doobject = false; // alternate between object and face recognition
      static bool intromode = false; // intro mode plays a video at the beginning, then shows some info messages
      static bool intromoviedone = false; // turns true when intro movie complete
      static ScriptItem const * scriptitem = &TheScript[0];
      static int scriptframe = 0;
      
      // Wait for next available camera image:
      jevois::RawImage inimg = inframe.get();

      // We only handle one specific input format in this demo:
      inimg.require("input", 320, 240, V4L2_PIX_FMT_YUYV);
      
      itsProcessingTimer.start();
      int const roihw = 32; // face & object roi half width and height
      
      // Compute saliency, in a thread:
      auto sal_fut = std::async(std::launch::async, [&](){ itsSaliency->process(inimg, true); });
      
      // While computing, wait for an image from our gadget driver into which we will put our results:
      jevois::RawImage outimg = outframe.get();
      outimg.require("output", 640, outimg.height, V4L2_PIX_FMT_YUYV);
      switch (outimg.height)
      {
      case 312: break; // normal mode
      case 360:
      case 480: intromode = true; break; // intro mode
      default: LFATAL("Incorrect output height: should be 312, 360 or 480");
      }

      // Play the intro movie first if requested:
      if (intromode && intromoviedone == false)
      {
        cv::Mat m = itsVideo->get();

        if (m.empty()) intromoviedone = true;
        else
        {
          jevois::rawimage::convertCvBGRtoRawImage(m, outimg, 75);

          // Handle bottom of th eframe: blank or banner
          if (outimg.height == 480)
            jevois::rawimage::paste(itsBanner, outimg, 0, 360);
          else if (outimg.height > 360)
            jevois::rawimage::drawFilledRect(outimg, 0, 360, outimg.width, outimg.height - 360, 0x8000);

          // If on a mac with height = 480, need to flip horizontally for photobooth to work (it will flip again):
          if (outimg.height == 480) jevois::rawimage::hFlipYUYV(outimg);

          sal_fut.get(); // yes, we are wasting CPU here, just to keep code more readable with the intro stuff added
          inframe.done();
          outframe.send();
          return;
        }
      }
      
      // Paste the original image to the top-left corner of the display:
      unsigned short const txtcol = jevois::yuyv::White;
      jevois::rawimage::paste(inimg, outimg, 0, 0);
      jevois::rawimage::writeText(outimg, "JeVois Saliency + Gist + Faces + Objects", 3, 3, txtcol);
      
      // Wait until saliency computation is complete:
      sal_fut.get();
      
      // find most salient point:
      int mx, my; intg32 msal;
      itsSaliency->getSaliencyMax(mx, my, msal);

      // Scale back to original image coordinates:
      int const smlev = itsSaliency->smscale::get();
      int const smadj = smlev > 0 ? (1 << (smlev-1)) : 0; // half a saliency map pixel adjustment
      int const dmx = (mx << smlev) + smadj;
      int const dmy = (my << smlev) + smadj;

      // Compute instantaneous attended ROI (note: coords must be even to avoid flipping U/V when we later paste):
      int const rx = std::min(int(inimg.width) - roihw, std::max(roihw, dmx));
      int const ry = std::min(int(inimg.height) - roihw, std::max(roihw, dmy));
      
      // Asynchronously launch a bunch of saliency drawings and filter the attended locations
      auto draw_fut =
        std::async(std::launch::async, [&]() {
            // Paste the various saliency results:
            drawMap(outimg, &itsSaliency->salmap, 320, 0, 16, 20);
            jevois::rawimage::writeText(outimg, "Saliency Map", 640 - 12*6-4, 3, txtcol);
            
            drawMap(outimg, &itsSaliency->color, 0, 240, 4, 18);
            jevois::rawimage::writeText(outimg, "Color", 3, 243, txtcol);
            
            drawMap(outimg, &itsSaliency->intens, 80, 240, 4, 18);
            jevois::rawimage::writeText(outimg, "Intensity", 83, 243, txtcol);
            
            drawMap(outimg, &itsSaliency->ori, 160, 240, 4, 18);
            jevois::rawimage::writeText(outimg, "Orientation", 163, 243, txtcol);
            
            drawMap(outimg, &itsSaliency->flicker, 240, 240, 4, 18);
            jevois::rawimage::writeText(outimg, "Flicker", 243, 243, txtcol);
            
            drawMap(outimg, &itsSaliency->motion, 320, 240, 4, 18);
            jevois::rawimage::writeText(outimg, "Motion", 323, 243, txtcol);
            
            // Draw the gist vector:
            drawGist(outimg, itsSaliency->gist, itsSaliency->gist_size, 400, 242, 40, 2);
            
            // Draw a small square at most salient location in image and in saliency map:
            jevois::rawimage::drawFilledRect(outimg, mx * 16 + 5, my * 16 + 5, 8, 8, 0xffff);
            jevois::rawimage::drawFilledRect(outimg, 320 + mx * 16 + 5, my * 16 + 5, 8, 8, 0xffff);
            jevois::rawimage::drawRect(outimg, rx - roihw, ry - roihw, roihw*2, roihw*2, 0xf0f0);
            jevois::rawimage::drawRect(outimg, rx - roihw+1, ry - roihw+1, roihw*2-2, roihw*2-2, 0xf0f0);

            // Blank out free space from 480 to 519 at the bottom, and small space above and below gist vector:
            jevois::rawimage::drawFilledRect(outimg, 480, 240, 40, 60, 0x8000);
            jevois::rawimage::drawRect(outimg, 400, 240, 80, 2, 0x80a0);
            jevois::rawimage::drawRect(outimg, 400, 298, 80, 2, 0x80a0);
            jevois::rawimage::drawFilledRect(outimg, 0, 300, 640, 12, jevois::yuyv::Black);

            // If intro mode, blank out rows 312 to bottom:
            if (outimg.height == 480)
            {
              jevois::rawimage::drawFilledRect(outimg, 0, 312, outimg.width, 48, 0x8000);
              jevois::rawimage::paste(itsBanner, outimg, 0, 360);
            }
            else if (outimg.height > 312)
              jevois::rawimage::drawFilledRect(outimg, 0, 312, outimg.width, outimg.height - 312, 0x8000);
            
            // Filter the attended locations:
            itsKF->set(dmx, dmy, inimg.width, inimg.height);
            float kfxraw, kfyraw, kfximg, kfyimg;
            itsKF->get(kfxraw, kfyraw, kfximg, kfyimg, inimg.width, inimg.height, 1.0F, 1.0F);
      
            // Draw a circle around the kalman-filtered attended location:
            jevois::rawimage::drawCircle(outimg, int(kfximg), int(kfyimg), 20, 1, jevois::yuyv::LightGreen);
            
            // Send saliency info to serial port (for arduino, etc):
            sendSerialImg2D(inimg.width, inimg.height, kfximg, kfyimg, roihw * 2, roihw * 2, "salient");

            // If intro mode, draw some text messages according to our script:
            if (intromode && intromoviedone)
            {
              // Compute fade: we do 1s fade in, 2s full luminance, 1s fade out:
              int lum = 255;
              if (scriptframe < 32) lum = scriptframe * 8;
              else if (scriptframe > 4*30 - 32) lum = std::max(0, (4*30 - scriptframe) * 8);

              // Display the text with the proper fade:
              int x = (640 - 10 * strlen(scriptitem->msg)) / 2;
              jevois::rawimage::writeText(outimg, scriptitem->msg, x, 325, 0x7700 | lum, jevois::rawimage::Font10x20);

              // Add a blinking marker if specified in the script:
              if (scriptitem->blinkx)
              {
                int phase = scriptframe / 10;
                if ((phase % 2) == 0) jevois::rawimage::drawDisk(outimg, scriptitem->blinkx, scriptitem->blinky,
                                                                 10, jevois::yuyv::LightTeal);
              }

              // Move to next video frame and possibly next script item or loop the script:
              if (++scriptframe >= 140)
              {
                scriptframe = 0; ++scriptitem;
                if (scriptitem->msg == nullptr) scriptitem = &TheScript[0];
              }
            }
          });

      // Extract a raw YUYV ROI around attended point:
      cv::Mat rawimgcv = jevois::rawimage::cvImage(inimg);
      cv::Mat rawroi = rawimgcv(cv::Rect(rx - roihw, ry - roihw, roihw * 2, roihw * 2));

      if (doobject)
      {
        // #################### Object recognition:
        
        // Prepare a color or grayscale ROI for the object recognition module:
        auto objsz = itsObjectRecognition->insize();
        cv::Mat objroi;
        switch (objsz.depth_)
        {
        case 1: // grayscale input
        {
          // mnist is white letters on black background, so invert the image before we send it for recognition, as we
          // assume here black letters on white backgrounds. We also need to provide a clean crop around the digit for
          // the deep network to work well:
          cv::cvtColor(rawroi, objroi, cv::COLOR_YUV2GRAY_YUYV);

          // Find the 10th percentile gray value:
          size_t const elem = (objroi.cols * objroi.rows * 10) / 100;
          std::vector<unsigned char> v; v.assign(objroi.datastart, objroi.dataend);
          std::nth_element(v.begin(), v.begin() + elem, v.end());
          unsigned char const thresh = std::min((unsigned char)(100), std::max((unsigned char)(30), v[elem]));

          // Threshold and invert the image:
          cv::threshold(objroi, objroi, thresh, 255, cv::THRESH_BINARY_INV);

          // Find the digit and center and crop it:
          cv::Mat pts; cv::findNonZero(objroi, pts);
          cv::Rect r = cv::boundingRect(pts);
          int const cx = r.x + r.width / 2;
          int const cy = r.y + r.height / 2;
          int const siz = std::min(roihw * 2, std::max(16, 8 + std::max(r.width, r.height))); // margin of 4 pix
          int const tlx = std::max(0, std::min(roihw*2 - siz, cx - siz/2));
          int const tly = std::max(0, std::min(roihw*2 - siz, cy - siz/2));
          cv::Rect ar(tlx, tly, siz, siz);
          cv::resize(objroi(ar), objroi, cv::Size(objsz.width_, objsz.height_), 0, 0, cv::INTER_AREA);
          //cv::imshow("cropped roi", objroi);cv::waitKey(1);
        }
        break;
          
        case 3: // color input
          cv::cvtColor(rawroi, objroi, cv::COLOR_YUV2RGB_YUYV);
          cv::resize(objroi, objroi, cv::Size(objsz.width_, objsz.height_), 0, 0, cv::INTER_AREA);
          break;
          
        default:
          LFATAL("Unsupported object detection input depth " << objsz.depth_);
        }
        
        // Launch object recognition on the ROI and get the recognition scores:
        auto scores = itsObjectRecognition->process(objroi);

        // Create a string to show all scores:
        std::ostringstream oss;
        for (size_t i = 0; i < scores.size(); ++i)
          oss << itsObjectRecognition->category(i) << ':' << std::fixed << std::setprecision(2) << scores[i] << ' ';
        itsScoresStr = oss.str();
        
        // Check whether the highest score is very high and significantly higher than the second best:
        float best1 = scores[0], best2 = scores[0]; size_t idx1 = 0, idx2 = 0;
        for (size_t i = 1; i < scores.size(); ++i)
        {
          if (scores[i] > best1) { best2 = best1; idx2 = idx1; best1 = scores[i]; idx1 = i; }
          else if (scores[i] > best2) { best2 = scores[i]; idx2 = i; }
        }
        
        // Update our display upon each "clean" recognition:
        if (best1 > 90.0F && best2 < 20.0F)
        {
          // Remember this recognized object for future displays:
          itsLastObjectCateg = itsObjectRecognition->category(idx1);
          itsLastObject = rawimgcv(cv::Rect(rx - 30, ry - 30, 60, 60)).clone(); // make a deep copy
          
          LINFO("Object recognition: best: " << itsLastObjectCateg <<" (" << best1 <<
                "), second best: " << itsObjectRecognition->category(idx2) << " (" << best2 << ')');
        }
      }
      else
      {
        // #################### Face detection:
        
        // Prepare a grey ROI from our raw YUYV roi:
        cv::Mat grayroi; cv::cvtColor(rawroi, grayroi, cv::COLOR_YUV2GRAY_YUYV);
        cv::equalizeHist(grayroi, grayroi);
        
        // Launch the face detector:
        std::vector<cv::Rect> faces; std::vector<std::vector<cv::Rect> > eyes;
        itsFaceDetector->process(grayroi, faces, eyes, false);
        
        // Draw the faces and eyes, if any:
        if (faces.size())
        {
          LINFO("detected " << faces.size() << " faces");
          // Store the attended ROI into our last ROI, fixed size 60x60 for our display:
          itsLastFace = rawimgcv(cv::Rect(rx - 30, ry - 30, 60, 60)).clone(); // make a deep copy
        }
        
        for (size_t i = 0; i < faces.size(); ++i)
        {
          // Draw one face:
          cv::Rect const & f = faces[i];
          jevois::rawimage::drawRect(outimg, f.x + rx - roihw, f.y + ry - roihw, f.width, f.height, 0xc0ff);
          
          // Draw the corresponding eyes:
          for (auto const & e : eyes[i])
            jevois::rawimage::drawRect(outimg, e.x + rx - roihw, e.y + ry - roihw, e.width, e.height, 0x40ff);
        }
      }
      
      // Let camera know we are done processing the raw YUV input image. NOTE: rawroi is now invalid:
      inframe.done();
      
      // Paste our last attended and recognized face and object (or empty pics):
      cv::Mat outimgcv(outimg.height, outimg.width, CV_8UC2, outimg.buf->data());
      itsLastObject.copyTo(outimgcv(cv::Rect(520, 240, 60, 60)));
      itsLastFace.copyTo(outimgcv(cv::Rect(580, 240, 60, 60)));
      
      // Wait until all saliency drawings are complete (since they blank out our object label area):
      draw_fut.get();
  
      // Print all object scores:
      jevois::rawimage::writeText(outimg, itsScoresStr, 2, 301, txtcol);

      // Write any positively recognized object category:
      jevois::rawimage::writeText(outimg, itsLastObjectCateg.c_str(), 517-6*itsLastObjectCateg.length(), 263, txtcol);
      
      // FIXME do svm on gist and write resuts here
      
      // Show processing fps:
      std::string const & fpscpu = itsProcessingTimer.stop() + ", v" JEVOIS_VERSION_STRING;
      jevois::rawimage::writeText(outimg, fpscpu, 3, 240 - 13, jevois::yuyv::White);

      // If on a mac with height = 480, need to flip horizontally for photobooth to work (it will flip again):
      if (outimg.height == 480) jevois::rawimage::hFlipYUYV(outimg);
      
      // Send the output image with our processing results to the host over USB:
      outframe.send();
      
      // Alternate between face and object recognition:
      doobject = ! doobject;
    }

  protected:
    std::shared_ptr<Saliency> itsSaliency;
    std::shared_ptr<FaceDetector> itsFaceDetector;
    std::shared_ptr<ObjectRecognitionBase> itsObjectRecognition;
    std::shared_ptr<Kalman2D> itsKF;
    std::shared_ptr<BufferedVideoReader> itsVideo;
    jevois::RawImage itsBanner;
    std::string itsScoresStr;
};

// Allow the module to be loaded as a shared object (.so) file:
JEVOIS_REGISTER_MODULE(JeVoisIntro);
