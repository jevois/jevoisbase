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

#include <jevoisbase/Components/ObjectRecognition/ObjectRecognitionMNIST.H>
#include "tiny-dnn/tiny_dnn/tiny_dnn.h"
#include <jevois/Debug/Log.H>

// ####################################################################################################
ObjectRecognitionMNIST::ObjectRecognitionMNIST(std::string const & instance) :
    ObjectRecognition<tiny_dnn::sequential>(instance)
{
  // Note: base class constructor allocates net
}

// ####################################################################################################
ObjectRecognitionMNIST::~ObjectRecognitionMNIST()
{
  // Nothing to do, base class destructor will de-allocate the network
}

// ####################################################################################################
void ObjectRecognitionMNIST::define()
{
  // LeNet for MNIST handwritten digit recognition: 32x32 in, 10 classes out:
#define O true
#define X false
  static bool const tbl[] = {
    O, X, X, X, O, O, O, X, X, O, O, O, O, X, O, O,
    O, O, X, X, X, O, O, O, X, X, O, O, O, O, X, O,
    O, O, O, X, X, X, O, O, O, X, X, O, X, O, O, O,
    X, O, O, O, X, X, O, O, O, O, X, X, O, X, O, O,
    X, X, O, O, O, X, X, O, O, O, O, X, O, O, X, O,
    X, X, X, O, O, O, X, X, O, O, O, O, X, O, O, O
  };
#undef O
#undef X
  // by default will use backend_t::tiny_dnn unless you compiled
  // with -DUSE_AVX=ON and your device supports AVX intrinsics
  tiny_dnn::core::backend_t backend_type = tiny_dnn::core::default_engine();
    
  // Construct network:
// construct nets
  //
  // C : convolution
  // S : sub-sampling
  // F : fully connected
  // clang-format off
  using fc = tiny_dnn::layers::fc;
  using conv = tiny_dnn::layers::conv;
  using ave_pool = tiny_dnn::layers::ave_pool;
  using tanh = tiny_dnn::activation::tanh;

  using tiny_dnn::core::connection_table;
  using padding = tiny_dnn::padding;

  (*net) << conv(32, 32, 5, 1, 6, padding::valid, true, 1, 1, backend_type)    // C1, 1@32x32-in, 6@28x28-out
         << tanh()
         << ave_pool(28, 28, 6, 2)   // S2, 6@28x28-in, 6@14x14-out
         << tanh()
         << conv(14, 14, 5, 6, 16, connection_table(tbl, 6, 16),
                 padding::valid, true, 1, 1, backend_type)    // C3, 6@14x14-in, 16@10x10-out
         << tanh()
         << ave_pool(10, 10, 16, 2)  // S4, 16@10x10-in, 16@5x5-out
         << tanh()
         << conv(5, 5, 5, 16, 120, padding::valid, true, 1, 1, backend_type)  // C5, 16@5x5-in, 120@1x1-out
         << tanh()
         << fc(120, 10, true, backend_type)  // F6, 120-in, 10-out
         << tanh();
}

// ####################################################################################################
void ObjectRecognitionMNIST::train(std::string const & path)
{
  LINFO("Load training data from directory " << path);

  // Load MNIST dataset:
  std::vector<tiny_dnn::label_t> train_labels, test_labels;
  std::vector<tiny_dnn::vec_t> train_images, test_images;
  LINFO("Load training labels...");
  tiny_dnn::parse_mnist_labels(std::string(path) + "/train-labels.idx1-ubyte", &train_labels);
  LINFO("Load training images...");
  tiny_dnn::parse_mnist_images(std::string(path) + "/train-images.idx3-ubyte", &train_images, -1.0, 1.0, 2, 2);
  LINFO("Load test labels...");
  tiny_dnn::parse_mnist_labels(std::string(path) + "/t10k-labels.idx1-ubyte", &test_labels);
  LINFO("Load test images...");
  tiny_dnn::parse_mnist_images(std::string(path) + "/t10k-images.idx3-ubyte", &test_images, -1.0, 1.0, 2, 2);
  
  LINFO("Start training...");
  int minibatch_size = 10;
  int num_epochs = 30;
  tiny_dnn::timer t;
  
  // Create callbacks:
  auto on_enumerate_epoch = [&](){
    LINFO(t.elapsed() << "s elapsed.");
    tiny_dnn::result res = net->test(test_images, test_labels);
    LINFO(res.num_success << "/" << res.num_total << " success/total validation score so far");
    t.restart();
  };

  auto on_enumerate_minibatch = [&](){ };

  // Training:
  tiny_dnn::adagrad optimizer;
  optimizer.alpha *= static_cast<tiny_dnn::float_t>(std::sqrt(minibatch_size));

  net->train<tiny_dnn::mse>(optimizer, train_images, train_labels, minibatch_size, num_epochs,
                            on_enumerate_minibatch, on_enumerate_epoch);

  LINFO("Training complete");

  // Test and show results:
  net->test(test_images, test_labels).print_detail(std::cout);
}

// ####################################################################################################
std::string const & ObjectRecognitionMNIST::category(size_t idx) const
{
  static std::vector<std::string> const names = { "0", "1", "2", "3", "4", "5", "6", "7", "8", "9" };

  if (idx >= names.size()) LFATAL("Category index out of bounds");
  
  return names[idx];
}
