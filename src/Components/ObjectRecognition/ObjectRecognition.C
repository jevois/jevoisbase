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

#include <jevoisbase/Components/ObjectRecognition/ObjectRecognition.H>
#include <jevois/Debug/Log.H>
#include <fstream>

#include "tiny-dnn/tiny_dnn/tiny_dnn.h"

// ####################################################################################################
ObjectRecognitionBase::ObjectRecognitionBase(std::string const & instance) :
    jevois::Component(instance)
{ }

// ####################################################################################################
ObjectRecognitionBase::~ObjectRecognitionBase()
{ }

// ####################################################################################################
template <typename NetType>
ObjectRecognition<NetType>::ObjectRecognition(std::string const & instance) :
    ObjectRecognitionBase(instance), net(new tiny_dnn::network<NetType>())
{ }

// ####################################################################################################
template <typename NetType>
void ObjectRecognition<NetType>::postInit()
{
  // Load from file, if available, otherwise trigger training:
  std::string const wpath = absolutePath("tiny-dnn/" + instanceName() + "/weights.tnn");

  try
  {
    net->load(wpath);
    LINFO("Loaded pre-trained weights from " << wpath);
  }
  catch (...)
  {
    LINFO("Could not load pre-trained weights from " << wpath);

    // First define the network (implemented in derived classes):
    this->define();

    // Then train it:
    this->train(absolutePath("tiny-dnn/" + instanceName()));

    // Finally save:
    LINFO("Saving trained weights to " << wpath);
    net->save(wpath);
    LINFO("Weights saved. Network ready to work.");
  }
}

// ####################################################################################################
template <typename NetType>
ObjectRecognition<NetType>::~ObjectRecognition()
{ delete net; }

// ####################################################################################################
template <typename NetType>
typename tiny_dnn::index3d<tiny_dnn::serial_size_t>
ObjectRecognition<NetType>::insize() const
{ return (*net)[0]->in_shape()[0]; }

// ####################################################################################################
template <typename NetType>
typename ObjectRecognition<NetType>::vec_t
ObjectRecognition<NetType>::process(cv::Mat const & img, bool normalize)
{
  auto inshape = (*net)[0]->in_shape()[0];

  if (img.cols != int(inshape.width_) ||
      img.rows != int(inshape.height_) ||
      img.channels() != int(inshape.depth_)) LFATAL("Incorrect input image size or format");
  
  // Convert input image to vec_t with values in [-1..1]:
  size_t const sz = inshape.size();
  tiny_dnn::vec_t data(sz);
  unsigned char const * in = img.data; tiny_dnn::float_t * out = &data[0];
  for (size_t i = 0; i < sz; ++i) *out++ = (*in++) * (2.0F / 255.0F) - 1.0F;

  // Recognize:
  if (normalize)
  {
    // Get the raw scores:
    auto scores = net->predict(data);

    // Normalize activation values between 0...100:
    tiny_dnn::layer * lastlayer = (*net)[net->depth() - 1];
    std::pair<tiny_dnn::float_t, tiny_dnn::float_t> outrange = lastlayer->out_value_range();
    tiny_dnn::float_t const mi = outrange.first;
    tiny_dnn::float_t const ma = outrange.second;

    for (tiny_dnn::float_t & s : scores) s = tiny_dnn::float_t(100) * (s - mi) / (ma - mi);

    return scores;
  }
  else
    return net->predict(data);
}

// ####################################################################################################
// Expplicit instantiations:
template class ObjectRecognition<tiny_dnn::sequential>;
