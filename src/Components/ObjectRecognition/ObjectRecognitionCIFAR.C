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

#include <jevoisbase/src/Components/ObjectRecognition/ObjectRecognitionCIFAR.H>
#include "tiny-dnn/tiny_dnn/tiny_dnn.h"
#include <jevois/Debug/Log.H>

// ####################################################################################################
ObjectRecognitionCIFAR::ObjectRecognitionCIFAR(std::string const & instance) :
    ObjectRecognition<tiny_dnn::sequential>(instance)
{ }

// ####################################################################################################
ObjectRecognitionCIFAR::~ObjectRecognitionCIFAR()
{
  // Nothing to do, base class destructor will de-allocate the network
}

// ####################################################################################################
void ObjectRecognitionCIFAR::define()
{
  typedef tiny_dnn::convolutional_layer<tiny_dnn::activation::identity> conv;
  typedef tiny_dnn::max_pooling_layer<tiny_dnn::activation::relu> pool;

  int const n_fmaps = 32; // number of feature maps for upper layer
  int const n_fmaps2 = 64; // number of feature maps for lower layer
  int const n_fc = 64; // number of hidden units in fully-connected layer

  (*net) << conv(32, 32, 5, 3, n_fmaps, tiny_dnn::padding::same)
         << pool(32, 32, n_fmaps, 2)
         << conv(16, 16, 5, n_fmaps, n_fmaps, tiny_dnn::padding::same)
         << pool(16, 16, n_fmaps, 2)
         << conv(8, 8, 5, n_fmaps, n_fmaps2, tiny_dnn::padding::same)
         << pool(8, 8, n_fmaps2, 2)
         << tiny_dnn::fully_connected_layer<tiny_dnn::activation::identity>(4 * 4 * n_fmaps2, n_fc)
         << tiny_dnn::fully_connected_layer<tiny_dnn::activation::softmax>(n_fc, 10);
}

// ####################################################################################################
void ObjectRecognitionCIFAR::train(std::string const & path)
{
  LINFO("Load training data from directory " << path);

  float learning_rate = 0.01F;
  
  // Load CIFAR dataset:
  std::vector<tiny_dnn::label_t> train_labels, test_labels;
  std::vector<tiny_dnn::vec_t> train_images, test_images;
  for (int i = 1; i <= 5; ++i)
    tiny_dnn::parse_cifar10(path + "/data_batch_" + std::to_string(i) + ".bin",
                            &train_images, &train_labels, -1.0, 1.0, 0, 0);

  tiny_dnn::parse_cifar10(path + "/test_batch.bin", &test_images, &test_labels, -1.0, 1.0, 0, 0);
  
  LINFO("Start training...");
  int const n_minibatch = 10;
  int const n_train_epochs = 30;

  tiny_dnn::timer t;
  
  // Create callbacks:
  auto on_enumerate_epoch = [&](){
    LINFO(t.elapsed() << "s elapsed.");
    tiny_dnn::result res = net->test(test_images, test_labels);
    LINFO(res.num_success << "/" << res.num_total << " success/total validation score so far");
    
    //disp.restart(train_images.size());
    t.restart();
  };

  auto on_enumerate_minibatch = [&](){
    //disp += n_minibatch;
  };

  // Training:
  tiny_dnn::adam optimizer;
  optimizer.alpha *= static_cast<tiny_dnn::float_t>(sqrt(n_minibatch) * learning_rate);
  net->train<tiny_dnn::cross_entropy>(optimizer, train_images, train_labels, n_minibatch, n_train_epochs,
                                      on_enumerate_minibatch, on_enumerate_epoch);

  LINFO("Training complete");

  // test and show results
  net->test(test_images, test_labels).print_detail(std::cout);
}

// ####################################################################################################
std::string const & ObjectRecognitionCIFAR::category(size_t idx) const
{
  static std::vector<std::string> const names =
    { "plane" /*"airplane"*/, "car" /*"automobile"*/, "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck" };

  if (idx >= names.size()) LFATAL("Category index out of bounds");
  
  return names[idx];
}
