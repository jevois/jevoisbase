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
#include <jevoisbase/Components/Saliency/Surprise.H>
#include <jevois/Types/BoundedBuffer.H>
#include <jevois/Image/RawImageOps.H>
#include <opencv2/videoio.hpp> // for cv::VideoCapture
#include <opencv2/imgproc.hpp> // for cv::rectangle()
#include <linux/videodev2.h> // for v4l2 pixel types
#include <fstream>
#include <jevois/Debug/Profiler.H>

static jevois::ParameterCategory const ParamCateg("Surprise Recording Options");

#define PATHPREFIX "/jevois/data/surpriserecorder/"

//! Parameter \relates SurpriseRecorder
JEVOIS_DECLARE_PARAMETER(filename, std::string, "Name of the video file to write. If path is not absolute, "
                         PATHPREFIX " will be prepended to it. Name should contain a printf-like directive for "
                         "one int argument, which will start at 0 and be incremented on each streamoff command.",
                         "video%06d.avi", ParamCateg);

//! Parameter \relates SurpriseRecorder
JEVOIS_DECLARE_PARAMETER(fourcc, std::string, "FourCC of the codec to use. The OpenCV VideoWriter doc is unclear "
                         "as to which codecs are supported. Presumably, the ffmpeg library is used inside OpenCV. "
                         "Hence any video encoder supported by ffmpeg should work. Tested codecs include: MJPG, "
                         "MP4V, AVC1. Make sure you also pick the right filename extension (e.g., .avi for MJPG, "
                         ".mp4 for MP4V, etc)",
                         "MJPG", boost::regex("^\\w{4}$"), ParamCateg);

//! Parameter \relates SurpriseRecorder
JEVOIS_DECLARE_PARAMETER(fps, double, "Video frames/sec as stored in the file and to be used both for recording and "
                         "playback. Beware that the video writer will drop frames if you are capturing faster than "
                         "the frame rate specified here. For example, if capturing at 120fps, be sure to set this "
                         "parameter to 120, otherwise by default the saved video will be at 30fps even though capture "
                         "was running at 120fps.",
                         15.0, ParamCateg);

//! Parameter \relates SurpriseRecorder
JEVOIS_DECLARE_PARAMETER(thresh, double, "Surprise threshold. Lower values will record more events.",
                         1.0e7, ParamCateg);

//! Parameter \relates SurpriseRecorder
JEVOIS_DECLARE_PARAMETER(ctxframes, unsigned int, "Number of context video frames recorded before and after "
                         "each surprising event.",
                         150, ParamCateg);


//! Surprise-based recording of events
/*! This module detects surprising events in the live video feed from the camera, and records short video clips of each
    detected event.

    Surprising is here defined according to Itti and Baldi's mathematical theory of surprise (see, e.g.,
    http://ilab.usc.edu/surprise/) which is applied to monitoring live video streams. When a surprising event is
    detected, a short video clip of that event is saved to the microSD card inside JeVois, for later review.

    It was created in this JeVois tutorial: http://jevois.org/tutorials/ProgrammerSurprise.html

    Using this module
    -----------------

    This module does not send any video output to USB. Rather, it just saves surprising events to microSD for later
    review. Hence, you may want to try the following:

    - mount the JeVois camera where you want it to detect surprising events

    - run it connected to a laptop computer, using any mode which does have some video output over USB (e.g., 640x500
      YUYV). Adjust the camera orientation to best fit your needs.

    - edit <b>JEVOIS:/config/initscript.cfg</b> to contain:
      \verbatim
      setmapping2 YUYV 640 480 15.0 JeVois SurpriseRecorder
      setpar thresh 1e7
      setpar channels S
      streamon
      \endverbatim
      and see the above tutorial for more details. Next time you power JeVois, it will immediately start detecting and
      recording surprising events in its view.

    Example
    -------

    Here is one hour of video surveillance footage. It is very boring overall. Except that a few brief surprising things
    occur (a few seconds each). Can you find them?

    \youtube{aSKncW7Jxrs}

    Here is what the SurpriseRecorder module found (4 true events plus 2 false alarms):

    \youtube{zIslIsHBfYw}

    With only 6 surprising events, and assuming +/- 10 seconds of context frames around each event, we have achieved a
    compression of the surveillance footage from 60 minutes to 2 minutes (a factor 30x).


    @author Laurent Itti

    @videomapping NONE 0 0 0 YUYV 640 480 15.0 JeVois SurpriseRecorder
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
class SurpriseRecorder : public jevois::Module,
                         public jevois::Parameter<filename, fourcc, fps, thresh, ctxframes>

{
  public:
    // ####################################################################################################
    //! Constructor
    // ####################################################################################################
    SurpriseRecorder(std::string const & instance) : jevois::Module(instance), itsBuf(1000), itsToSave(0),
                                                     itsFileNum(0), itsRunning(false)
    { itsSurprise = addSubComponent<Surprise>("surprise"); }

    // ####################################################################################################
    //! Virtual destructor for safe inheritance
    // ####################################################################################################
    virtual ~SurpriseRecorder()
    { }

    // ####################################################################################################
    //! Get started
    // ####################################################################################################
    void postInit() override
    {
      itsRunning.store(true);
      
      // Get our run() thread going, it is in charge of compressing and saving frames:
      itsRunFut = std::async(std::launch::async, &SurpriseRecorder::run, this);
    }

    // ####################################################################################################
    //! Get stopped
    // ####################################################################################################
    void postUninit() override
    {
      // Signal end of run:
      itsRunning.store(false);
      
      // Push an empty frame into our buffer to signal the end of video to our thread:
      itsBuf.push(cv::Mat());

      // Wait for the thread to complete:
      LINFO("Waiting for writer thread to complete, " << itsBuf.filled_size() << " frames to go...");
      try { itsRunFut.get(); } catch (...) { jevois::warnAndIgnoreException(); }
      LINFO("Writer thread completed. Syncing disk...");
      if (std::system("/bin/sync")) LERROR("Error syncing disk -- IGNORED");
      LINFO("Video " << itsFilename << " saved.");
    }

    // ####################################################################################################
    //! Processing function, version with no video output
    // ####################################################################################################
    void process(jevois::InputFrame && inframe) override
    {
      static jevois::Profiler prof("surpriserecorder");

       // Wait for next available camera image:
      jevois::RawImage inimg = inframe.get(); unsigned int const w = inimg.width, h = inimg.height;
      inimg.require("input", w, h, V4L2_PIX_FMT_YUYV); // accept any image size but require YUYV pixels

      prof.start();
      
      // Compute surprise in a thread:
      std::future<double> itsSurpFut =
        std::async(std::launch::async, [&]() { return itsSurprise->process(inimg); } );

      prof.checkpoint("surprise launched");
 
      // Convert the image to OpenCV BGR and push into our context buffer:
      cv::Mat cvimg = jevois::rawimage::convertToCvBGR(inimg);
      itsCtxBuf.push_back(cvimg);
      if (itsCtxBuf.size() > ctxframes::get()) itsCtxBuf.pop_front();
      
      prof.checkpoint("image pushed");

      // Wait until our surprise thread is done:
      double surprise = itsSurpFut.get(); // this could throw and that is ok
      //LINFO("surprise = " << surprise << " itsToSave = " << itsToSave);
      
      prof.checkpoint("surprise done");

      // Let camera know we are done processing the raw input image:
      inframe.done();

      // If the current frame is surprising, check whether we are already saving. If so, just push the current frame for
      // saving and reset itsToSave to full context length (after the event). Otherwise, keep saving until the context
      // after the event is exhausted:
      if (surprise >= thresh::get())
      {
        // Draw a rectangle on surprising frames. Note that we draw it in cvimg but, since the pixel memory is shared
        // with the copy of it we just pushed into itsCtxBuf, the rectangle will get drawn in there too:
        cv::rectangle(cvimg, cv::Point(3, 3), cv::Point(w-4, h-4), cv::Scalar(0,0,255), 7);
        
        if (itsToSave)
        {
          // we are still saving the context after the previous event, just add our new one:
          itsBuf.push(cvimg);

          // Reset the number of frames we will save after the end of the event:
          itsToSave = ctxframes::get();
        }
        else
        {
          // Start of a new event. Dump the whole itsCtxBuf to the writer:
          for (cv::Mat const & im : itsCtxBuf) itsBuf.push(im);

          // Initialize the number of frames we will save after the end of the event:
          itsToSave = ctxframes::get();
        }
      }
      else if (itsToSave)
      {
        // No more surprising event, but we are still saving the context after the last one:
        itsBuf.push(cvimg);

        // One more context frame after the last event was saved:
        --itsToSave;

        // Last context frame after the event was just pushed? If so, push an empty frame as well to close the current
        // video file. We will open a new file on the next surprising event:
        if (itsToSave == 0) itsBuf.push(cv::Mat());
      }

      prof.stop();
    }

    // ####################################################################################################
  protected:
    std::shared_ptr<Surprise> itsSurprise;
    
    // ####################################################################################################
    //! Video writer thread
    // ####################################################################################################
    void run() // Runs in a thread
    {
      while (itsRunning.load())
      {
        // Create a VideoWriter here, since it has no close() function, this will ensure it gets destroyed and closes
        // the movie once we stop the recording:
        cv::VideoWriter writer;
        int frame = 0;
      
        while (true)
        {
          // Get next frame from the buffer:
          cv::Mat im = itsBuf.pop();

          // An empty image will be pushed when we are ready to close the video file:
          if (im.empty()) break;
        
          // Start the encoder if it is not yet running:
          if (writer.isOpened() == false)
          {
            // Parse the fourcc, regex in our param definition enforces 4 alphanumeric chars:
            std::string const fcc = fourcc::get();
            int const cvfcc = cv::VideoWriter::fourcc(fcc[0], fcc[1], fcc[2], fcc[3]);
          
            // Add path prefix if given filename is relative:
            std::string fn = filename::get();
            if (fn.empty()) LFATAL("Cannot save to an empty filename");
            if (fn[0] != '/') fn = PATHPREFIX + fn;

            // Create directory just in case it does not exist:
            std::string const cmd = "/bin/mkdir -p " + fn.substr(0, fn.rfind('/'));
            if (std::system(cmd.c_str())) LERROR("Error running [" << cmd << "] -- IGNORED");

            // Fill in the file number; be nice and do not overwrite existing files:
            while (true)
            {
              char tmp[2048];
              std::snprintf(tmp, 2047, fn.c_str(), itsFileNum);
              std::ifstream ifs(tmp);
              if (ifs.is_open() == false) { itsFilename = tmp; break; }
              ++itsFileNum;
            }
            
            // Open the writer:
            if (writer.open(itsFilename, cvfcc, fps::get(), im.size(), true) == false)
              LFATAL("Failed to open video encoder for file [" << itsFilename << ']');

            sendSerial("SAVETO " + itsFilename);
          }

          // Write the frame:
          writer << im;

          // Report what is going on once in a while:
          if ((++frame % 100) == 0) sendSerial("SAVEDNUM " + std::to_string(frame));
        }

        sendSerial("SAVEDONE " + itsFilename);

        // Our writer runs out of scope and closes the file here.
        ++itsFileNum;
      }
    }
    
    std::future<void> itsRunFut; //!< Future for our run() thread
    std::deque<cv::Mat> itsCtxBuf; //!< Buffer for context frames before event start
    jevois::BoundedBuffer<cv::Mat, jevois::BlockingBehavior::Block,
                          jevois::BlockingBehavior::Block> itsBuf; //!< Buffer for frames to save
    int itsToSave; //!< Number of context frames after end of event that remain to be saved
    int itsFileNum; //!< Video file number
    std::atomic<bool> itsRunning; //!< Flag to let run thread when to quit
    std::string itsFilename; //!< Current video file name
};

// Allow the module to be loaded as a shared object (.so) file:
JEVOIS_REGISTER_MODULE(SurpriseRecorder);
