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


// Much of this code inspired by the label_image example for TensorFlow Lite:

/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the
License.  You may obtain a copy of the License at  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an
"AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  See the License for the
specific language governing permissions and limitations under the License. */

#include <jevoisbase/Components/ObjectDetection/TensorFlow.H>
#include <jevois/Core/Module.H>
#include <opencv2/imgproc/imgproc.hpp>
#include <queue>
#include <fstream>
#include <cstdio>

#include <tensorflow/lite/kernels/register.h>
#include <tensorflow/lite/optional_debug_tools.h>
#include <tensorflow/lite/string_util.h>

// ####################################################################################################
int TensorFlow::JeVoisReporter::Report(char const * format, va_list args)
{
  char buf[1024];
  int ret = vsnprintf(buf, 1024, format, args);
  LERROR(buf);
  return ret;
}

// ####################################################################################################
TensorFlow::TensorFlow(std::string const & instance) :
    jevois::Component(instance), itsReady(false), itsNeedReload(false)
{ }

// ####################################################################################################
TensorFlow::~TensorFlow()
{ }

// ####################################################################################################
void TensorFlow::onParamChange(tflow::netdir const & param, std::string const & newval)
{ itsNeedReload.store(true); }

// ####################################################################################################
void TensorFlow::onParamChange(tflow::dataroot const & param, std::string const & newval)
{ itsNeedReload.store(true); }

// ####################################################################################################
void TensorFlow::readLabelsFile(std::string const & fname)
{
  std::ifstream file(fname);
  if (!file) LFATAL("Could not open labels file " << fname);
  labels.clear();

  std::string line;
  while (std::getline(file, line)) labels.push_back(line);

  // Tensorflow requires the number of labels to be a multiple of 16:
  numlabels = labels.size();
  int const padding = 16;
  while (labels.size() % padding) labels.emplace_back();

  LINFO("Loaded " << numlabels << " category names from " << fname);
}

// ####################################################################################################
void TensorFlow::postInit()
{
  // Start a thread to load the desired network:
  loadNet();
}

// ####################################################################################################
template <class T>
void TensorFlow::get_top_n(T * prediction, int prediction_size, std::vector<jevois::ObjReco> & top_results,
			   bool input_floating)
{
  int const topn = top::get();
  float const th = thresh::get() * 0.01F;
  top_results.clear();

  // Will contain top N results in ascending order.
  std::priority_queue<std::pair<float, int>, std::vector<std::pair<float, int> >,
                      std::greater<std::pair<float, int> > > top_result_pq;

  float const scale = scorescale::get() * (input_floating ? 1.0 : 1.0F / 255.0F);
  
  for (int i = 0; i < prediction_size; ++i)
  {
    float value = prediction[i] * scale;

    // Only add it if it beats the threshold and has a chance at being in the top N.
    if (value < th) continue;

    top_result_pq.push(std::pair<float, int>(value, i));

    // If at capacity, kick the smallest value out.
    if (top_result_pq.size() > topn) top_result_pq.pop();
  }

  // Copy to output vector and reverse into descending order.
  while (!top_result_pq.empty())
  {
    auto const & tr = top_result_pq.top();
    top_results.push_back( { tr.first * 100, std::string(labels[tr.second]) } );
    top_result_pq.pop();
  }
  std::reverse(top_results.begin(), top_results.end());
}

// ####################################################################################################
void TensorFlow::loadNet()
{
  // Users will neet to wait until last load is done before they load again:
  if (itsReadyFut.valid())
  {
    if (itsReadyFut.wait_for(std::chrono::milliseconds(5)) == std::future_status::ready)
      try { itsReadyFut.get(); } catch (...) { }
    else
      throw std::logic_error("Loading already in progress. Attempt to load again rejected");  
  }
  
  // Flip itsNeedReload to false so we do not get called several times for the same need to reload:
  itsNeedReload.store(false);

  // We are not ready anymore:
  itsReady.store(false);

  // Since loading big networks can take a while, do it in a thread so we can keep streaming video in the
  // meantime. itsReady will flip to true when the load is complete.
  itsReadyFut = std::async(std::launch::async, [&]() {
      if (model)
      {
	LINFO("Closing model..");
	model.reset();
	interpreter.reset();
	labels.clear();
	numlabels = 0;
      }

      std::string root = dataroot::get(); if (root.empty() == false) root += '/';
      std::string const modelfile = absolutePath(root + netdir::get() + "/model.tflite");
      std::string const labelfile = absolutePath(root + netdir::get() + "/labels.txt");

      LINFO("Using model from " << modelfile);
      LINFO("Using labels from " << labelfile);

      // Load the labels:
      readLabelsFile(labelfile);

      // Create the model from flatbuffer file using mmap:
      model = tflite::FlatBufferModel::BuildFromFile(modelfile.c_str(), &itsErrorReporter);
      if (!model) LFATAL("Failed to mmap model " << modelfile);
      LINFO("Loaded model " << modelfile);
      
      tflite::ops::builtin::BuiltinOpResolver resolver;
      tflite::InterpreterBuilder(*model, resolver)(&interpreter);
      if (!interpreter) LFATAL("Failed to construct interpreter");
      //////interpreter->UseNNAPI(s->accel);

      LINFO("Tensors size: " << interpreter->tensors_size());
      LINFO("Nodes size: " << interpreter->nodes_size());
      LINFO("inputs: " << interpreter->inputs().size());
      LINFO("input(0) name: " << interpreter->GetInputName(0));

      int t_size = interpreter->tensors_size();
      for (int i = 0; i < t_size; ++i)
	if (interpreter->tensor(i)->name)
	  LINFO(i << ": " << interpreter->tensor(i)->name << ", "
		<< interpreter->tensor(i)->bytes << ", "
		<< interpreter->tensor(i)->type << ", "
		<< interpreter->tensor(i)->params.scale << ", "
		<< interpreter->tensor(i)->params.zero_point);

      if (threads::get()) interpreter->SetNumThreads(threads::get());

      LINFO("input: " << interpreter->inputs()[0]);
      LINFO("number of inputs: " << interpreter->inputs().size());
      LINFO("number of outputs: " << interpreter->outputs().size());

      if (interpreter->AllocateTensors() != kTfLiteOk) LFATAL("Failed to allocate tensors");

      LINFO("TensorFlow network ready");

      // We are ready to rock:
      itsReady.store(true);
    });
}

// ####################################################################################################
void TensorFlow::postUninit()
{
  try { itsReadyFut.get(); } catch (...) { }
}

// ####################################################################################################
float TensorFlow::predict(cv::Mat const & cvimg, std::vector<jevois::ObjReco> & results)
{
  if (itsNeedReload.load()) loadNet();
  if (itsReady.load() == false) throw std::logic_error("not ready yet...");
  
  int const image_width = cvimg.cols;
  int const image_height = cvimg.rows;
  int const image_type = cvimg.type();

  // get input dimension from the input tensor metadata assuming one input only
  int input = interpreter->inputs()[0];
  TfLiteIntArray * dims = interpreter->tensor(input)->dims;
  int const wanted_height = dims->data[1];
  int const wanted_width = dims->data[2];
  int const wanted_channels = dims->data[3];

  if (wanted_channels != 1 && wanted_channels != 3)
    LFATAL("Network wants " << wanted_channels << " input channels, but only 1 or 3 are supported");
  if (wanted_channels == 3 && image_type != CV_8UC3) LFATAL("Network wants RGB but input image is not CV_8UC3");
  if (wanted_channels == 1 && image_type != CV_8UC1) LFATAL("Network wants Gray but input image is not CV_8UC1");
  
  if (image_width != wanted_width || image_height != wanted_height)
    LFATAL("Wrong input size " << image_width << 'x' << image_height << " but network wants "
	   << wanted_width << 'x' << wanted_height);

  // Copy input image to input tensor, converting pixel type if needed:
  switch (interpreter->tensor(input)->type)
  {
  case kTfLiteUInt8:
  {
    memcpy(interpreter->typed_tensor<uint8_t>(input), cvimg.data, cvimg.total() * cvimg.elemSize());
    break;
  }
  case kTfLiteFloat32:
  {
    cv::Mat convimg;
    if (wanted_channels == 1) cvimg.convertTo(convimg, CV_32FC1, 1.0F / 127.5F, -1.0F);
    else cvimg.convertTo(convimg, CV_32FC3, 1.0F / 127.5F, -1.0F);
    memcpy(interpreter->typed_tensor<float>(input), convimg.data, convimg.total() * convimg.elemSize());
    break;
  }
  default:
    LFATAL("only uint8 or float32 network input pixel types are supported");
  }

  // Run the inference:
  struct timeval start, stop;
  gettimeofday(&start, 0);
  if (interpreter->Invoke() != kTfLiteOk) LFATAL("Failed to invoke tflite");
  gettimeofday(&stop, 0);
  float predtime = (stop.tv_sec * 1000 + stop.tv_usec / 1000) - (start.tv_sec * 1000 + start.tv_usec / 1000);
  
  const size_t num_results = 5;
  const float threshold = 0.001f;

  std::vector<std::pair<float, int>> top_results;

  int output = interpreter->outputs()[0];
  switch (interpreter->tensor(output)->type)
  {
  case kTfLiteFloat32:
    get_top_n<float>(interpreter->typed_output_tensor<float>(0), numlabels, results, true);
    break;
    
  case kTfLiteUInt8:
    get_top_n<uint8_t>(interpreter->typed_output_tensor<uint8_t>(0), numlabels, results, false);
    break;

  default:
    LFATAL("cannot handle output type " << interpreter->tensor(input)->type << " yet");
  }

  return predtime;
}

// ####################################################################################################
void TensorFlow::getInDims(int & w, int & h, int & c)
{
  if (itsNeedReload.load()) loadNet();
  if (itsReady.load() == false) throw std::logic_error("not ready yet...");

  // get input dimension from the input tensor metadata assuming one input only
  int input = interpreter->inputs()[0];
  TfLiteIntArray * dims = interpreter->tensor(input)->dims;
  h = dims->data[1]; w = dims->data[2]; c = dims->data[3];
}
