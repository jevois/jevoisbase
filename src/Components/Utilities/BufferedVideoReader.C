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


#include <jevoisbase/Components/Utilities/BufferedVideoReader.H>

// ####################################################################################################
BufferedVideoReader::BufferedVideoReader(std::string const & instance, size_t bufsize) :
    jevois::Component(instance), itsBuf(bufsize), itsRunning(true)
{ }

// ####################################################################################################
BufferedVideoReader::~BufferedVideoReader()
{ }

// ####################################################################################################
void BufferedVideoReader::postInit()
{
  // Start our reader thread:
  itsRunFut = std::async(std::launch::async, &BufferedVideoReader::run, this);
}

// ####################################################################################################
void BufferedVideoReader::postUninit()
{
  // Tell run() thread to finish up:
  itsRunning.store(false);
  if (itsBuf.filled_size()) itsBuf.pop(); // in case run() is blocked trying to push
  
  try { itsRunFut.get(); } catch (...) { jevois::warnAndIgnoreException(); }
}

// ####################################################################################################
cv::Mat BufferedVideoReader::get()
{ return itsBuf.pop(); }

// ####################################################################################################
void BufferedVideoReader::run()
{
  // Open the video file:
  std::string const path = absolutePath(filename::get());
  cv::VideoCapture vcap(path);
  if (vcap.isOpened() == false) { itsBuf.push(cv::Mat()); LERROR("Could not open video file " << path); }

  cv::Mat frame;
  while (itsRunning.load())
    if (vcap.read(frame)) itsBuf.push(frame); else { itsBuf.push(cv::Mat()); break; }
}

