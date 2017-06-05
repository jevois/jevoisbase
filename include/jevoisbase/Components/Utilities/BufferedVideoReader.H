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

#pragma once

#include <jevois/Component/Component.H>
#include <jevois/Types/BoundedBuffer.H>

#include <opencv2/opencv.hpp>
#include <opencv2/videoio.hpp> // for cv::VideoCapture

#include <future>
#include <atomic>

namespace bufferedvideoreader
{
  static jevois::ParameterCategory const ParamCateg("Buffered Video Reader Options");

  //! Parameter \relates BufferedVideoReader
  JEVOIS_DECLARE_PARAMETER(filename, std::string, "Filename of video to read (if not absolute, will be assumed to be "
                           "relative to Component path)", "movie.mpg", ParamCateg);

}

//! Simple class to read video frames from a movie file, decode them, and buffer them for smooth playback
/*! Reading and decoding is done in a thread, and decoded images are pushed into a producer/consumer queue. When the
    queue is full, decoding pauses until some images are popped off the queue by some other thread, for example to
    display them or to send them over USB link. \ingroup components */
class BufferedVideoReader : public jevois::Component,
                            public jevois::Parameter<bufferedvideoreader::filename>
{
  public:
    //! Constructor
    BufferedVideoReader(std::string const & instance, size_t bufsize = 100);

    //! Virtual destructor for safe inheritance
    ~BufferedVideoReader();

    //! Get the next frame as a BGR cv::Mat, or an empty cv::Mat when the movie is finished
    cv::Mat get();
    
  protected:
    //! Start the thread that loads, decodes and pushes the frames into our buffer
    virtual void postInit() override;

    //! Uninit, wait on our run thread and swallow any exception
    virtual void postUninit() override;

    //! Reader thread
    void run();

  private:
    jevois::BoundedBuffer<cv::Mat, jevois::BlockingBehavior::Block, jevois::BlockingBehavior::Block> itsBuf;
    std::future<void> itsRunFut;
    std::atomic<bool> itsRunning;
};

