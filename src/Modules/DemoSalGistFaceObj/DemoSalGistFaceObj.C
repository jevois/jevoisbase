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

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <linux/videodev2.h> // for v4l2 pixel types
//#include <opencv2/highgui/highgui.hpp> // used for debugging only, see imshow below
// icon by Freepik in interface at flaticon

//! Simple demo that combines saliency, gist, face detection, and object recognition
/*! Run the visual saliency algorithm to find the most interesting location in the field of view. Then extract a square
    image region around that point. On alternating frames, either

    - attempt to detect a face in the attended region, and, if positively detected, show the face in the bottom-right
      corner of the display. The last detected face will remain shown in the bottom-right corner of the display until a
      new face is detected.

    - or attempt to recognize an object in the attended region, using a deep neural network. The default network is a
      handwritten digot recognition network that replicated the original LeNet by Yann LeCun and is one of the very
      first convolutional neural networks. The network has been trained on the standard MNIST database of handwritten
      digits, and achives over 99% correct recognition on the MNIST test dataset. When a digit is positively identified,
      a picture of it appears near the last detected face towards the bottom-right corner of the display, and a text
      string with the digit that has been identified appears to the left of the picture of the digit.

   Serial Messages
    ---------------

    This module can send standardized serial messages as described in \ref UserSerialStyle, where all coordinates and
    sizes are standardized using \ref coordhelpers. One message is issued on every video frame at the temporally
    filtered attended (most salient) location (green circle in the video display):

    - Serial message type: \b 2D
    - `id`: always \b sm (shorthand for saliency map)
    - `x`, `y`: standardized 2D coordinates of temporally-filtered most salient point
    - `w`, `h`: standardized size of the pink square box around each attended point
    - `extra`: none (empty string)

    See \ref UserSerialStyle for more on standardized serial messages, and \ref coordhelpers for more info on
    standardized coordinates.


    @author Laurent Itti

    @displayname Demo Saliency + Gist + Face Detection + Object Recognition
    @videomapping YUYV 640 312 50.0 YUYV 320 240 50.0 JeVois DemoSalGistFaceObj
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
class DemoSalGistFaceObj : public jevois::StdModule
{
  public:
    //! Constructor
    DemoSalGistFaceObj(std::string const & instance) : jevois::StdModule(instance), itsScoresStr(" ")
    {
      itsSaliency = addSubComponent<Saliency>("saliency");
      itsFaceDetector = addSubComponent<FaceDetector>("facedetect");
      itsObjectRecognition = addSubComponent<ObjectRecognitionMNIST>("MNIST");
      itsKF = addSubComponent<Kalman2D>("kalman");
    }

    //! Virtual destructor for safe inheritance
    virtual ~DemoSalGistFaceObj() { }

    //! Processing function
    virtual void process(jevois::InputFrame && inframe, jevois::OutputFrame && outframe) override
    {
      static jevois::Timer itsProcessingTimer("Processing");
      static cv::Mat itsLastFace(60, 60, CV_8UC2, 0x80aa) ; // Note that this one will contain raw YUV pixels
      static cv::Mat itsLastObject(60, 60, CV_8UC2, 0x80aa) ; // Note that this one will contain raw YUV pixels
      static std::string itsLastObjectCateg;
      static bool doobject = false; // alternate between object and face recognition

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
      outimg.require("output", 640, 312, V4L2_PIX_FMT_YUYV);
      
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

            // Filter the attended locations:
            itsKF->set(dmx, dmy, inimg.width, inimg.height);
            float kfxraw, kfyraw, kfximg, kfyimg;
            itsKF->get(kfxraw, kfyraw, kfximg, kfyimg, inimg.width, inimg.height, 1.0F, 1.0F);
      
            // Draw a circle around the kalman-filtered attended location:
            jevois::rawimage::drawCircle(outimg, int(kfximg), int(kfyimg), 20, 1, jevois::yuyv::LightGreen);
            
            // Send saliency info to serial port (for arduino, etc):
            sendSerialImg2D(inimg.width, inimg.height, kfximg, kfyimg, roihw * 2, roihw * 2, "sm");
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
      std::string const & fpscpu = itsProcessingTimer.stop();
      jevois::rawimage::writeText(outimg, fpscpu, 3, 240 - 13, jevois::yuyv::White);
 
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
    std::string itsScoresStr;
};

// Allow the module to be loaded as a shared object (.so) file:
JEVOIS_REGISTER_MODULE(DemoSalGistFaceObj);
