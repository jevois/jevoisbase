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

#include <jevoisbase/Components/ObjectRecognition/ObjectRecognitionILAB.H>
#include "tiny-dnn/tiny_dnn/tiny_dnn.h"
#include <jevois/Debug/Log.H>
#include <algorithm>
#include <fstream>

#include <opencv2/imgproc/imgproc.hpp>

#ifdef JEVOIS_PLATFORM
#include <opencv2/imgcodecs/imgcodecs.hpp> // for imread in opencv3.1
#else
#include <opencv2/highgui/highgui.hpp> // for imread and imshow in older opencv
#endif


// ####################################################################################################
ObjectRecognitionILAB::ObjectRecognitionILAB(std::string const & instance) :
    ObjectRecognition<tiny_dnn::sequential>(instance)
{ }

// ####################################################################################################
ObjectRecognitionILAB::~ObjectRecognitionILAB()
{
  // Nothing to do, base class destructor will de-allocate the network
}

// ####################################################################################################
void ObjectRecognitionILAB::define()
{
  using conv    = tiny_dnn::convolutional_layer;
  using pool    = tiny_dnn::max_pooling_layer;
  using fc      = tiny_dnn::fully_connected_layer;
  using relu    = tiny_dnn::relu_layer;
  using softmax = tiny_dnn::softmax_layer;

  const size_t n_fmaps  = 10;  ///< number of feature maps for upper layer
  const size_t n_fc = 64;  ///< number of hidden units in fully-connected layer
  int const n_categ = 5; // number of object categories

  (*net) << conv(32, 32, 5, 3, n_fmaps, tiny_dnn::padding::same)             // C1
         << pool(32, 32, n_fmaps, 2)                                         // P2
         << relu(16, 16, n_fmaps)                                            // activation
         << conv(16, 16, 5, n_fmaps, n_fmaps * 2, tiny_dnn::padding::same)   // C3
         << pool(16, 16, n_fmaps * 2, 2)                                     // P4
         << relu(8, 8, n_fmaps)                                              // activation
         << conv(8, 8, 5, n_fmaps * 2, n_fmaps * 4, tiny_dnn::padding::same) // C5
         << pool(8, 8, n_fmaps * 4, 2)                                       // P6
         << relu(4, 4, n_fmaps * 42)                                         // activation
         << fc(4 * 4 * n_fmaps * 4, n_fc)                                    // FC7
         << fc(n_fc, n_categ) << softmax(n_categ);                           // FC10
}

// ####################################################################################################
namespace
{
  void load_compiled(std::string const & fname, std::vector<tiny_dnn::label_t> & labels,
                     std::vector<tiny_dnn::vec_t> & images)
  {
    float const scale_min = -1.0F;
    float const scale_max = 1.0F;
    int const w = 32, h = 32; // image width and height

    std::ifstream ifs(fname, std::ios::in | std::ios::binary);
    if (ifs.is_open() == false) LFATAL("Failed to open load " << fname);

    // We need to randomize the order. To achieve this, we will here first compute a randomized set of indices, then
    // populate the data at those indices:
    size_t siz;
    {
      // get input file size:
      std::ifstream file(fname, std::ios::binary | std::ios::ate);
      siz = file.tellg() / (w * h * 3 + 1);
      LINFO("File has " << siz << " entries");
    }

    std::vector<size_t> idx; for (size_t i = 0; i < siz; ++i) idx.push_back(i);
    std::random_shuffle(idx.begin(), idx.end());
    labels.resize(siz); images.resize(siz);

    // Load the data:
    std::vector<unsigned char> buf(w * h * 3);

    for (size_t i = 0; i < siz; ++i)
    {
      unsigned char label; ifs.read((char *)(&label), 1);
      if (!ifs) LFATAL("Error reading " << fname);
      labels[idx[i]] = label;
      
      ifs.read((char *)(&buf[0]), buf.size());
      if (!ifs) LFATAL("Error reading " << fname);
      
      tiny_dnn::vec_t img;
      std::transform(buf.begin(), buf.end(), std::back_inserter(img),
                     [&](unsigned char c) { return scale_min + (scale_max - scale_min) * c / 255.0F; });
    
      images[idx[i]] = img;
    }
    ifs.close();
    LINFO("Loaded " << siz << " images and labels from file " << fname);
  }

  // ####################################################################################################
  void create_compiled(std::string const & fname, size_t startinst, size_t numinst)
  {
    static std::vector<std::string> const categs = { "car", "equip", "plane", "boat", "mil" }; // FIXME

    LINFO("Create " << fname << " using " << numinst << " instances starting at " << startinst);
    
    std::ofstream ofs(fname, std::ios::out | std::ios::binary);
    if (ofs.is_open() == false) LFATAL("Error trying to write file " << fname);

    for (unsigned char categ = 0; categ < categs.size(); ++categ)
      for (size_t inst = startinst; inst < startinst + numinst; ++inst)
      {
        // Create big image filename: eg, base/boat/boat-i0008-b0077-cropped.png
        char tmp[2048];
        snprintf(tmp, 2048, "/lab/tmp10b/u/iLab-20M-Cropped-Jiaping-Augments/%s/%s-i%04zu-b0000-cropped.png",
                 categs[categ].c_str(), categs[categ].c_str(), inst);
        LINFO("... adding 1320 images from " << tmp);
        
        // Load the big image:
        cv::Mat bigimg = cv::imread(tmp);

        // Images contain 44 (wide) x 30 (tall) = 1320 crops. Determine crop size:
        int const cw = bigimg.cols / 44;
        int const ch = bigimg.rows / 30;
        LINFO("cw="<<cw<<" ch="<<ch);
        // Extract the individual views: we have 44 views horizontally, in this loop order
        int x = 0, y = 0;
        for (int cam = 0; cam < 11; ++cam)
          for (int rot = 0; rot < 8; ++rot)
            for (int lig = 0; lig < 5; ++lig)
              for (int foc = 0; foc < 3; ++foc)
              {
                cv::Mat imgcrop = bigimg(cv::Rect(x, y, cw, ch));
                cv::Mat obj; cv::resize(imgcrop, obj, cv::Size(32, 32), 0, 0, cv::INTER_AREA);

#ifndef JEVOIS_PLATFORM
                cv::imshow("conversion", obj); cv::waitKey(1);
#endif
                cv::Mat rgbobj; cv::cvtColor(obj, rgbobj, cv::COLOR_BGR2RGB); // opencv reads BGR by default

                ofs.write((char const *)(&categ), 1);
                ofs.write((char const *)(rgbobj.data), 32*32*3);
                
                x += cw; if (x >= bigimg.cols) { x = 0; y += ch; }
              }
      }
  }
}

// ####################################################################################################
void ObjectRecognitionILAB::train(std::string const & path)
{
  LINFO("Load training data from directory " << path);

  float learning_rate = 0.01F;
  size_t const ntrain = 18; // number of objects to use for training
  size_t const ntest = 4; // number of objects to use for test
  
  // Load ILAB dataset:
  std::vector<tiny_dnn::label_t> train_labels, test_labels;
  std::vector<tiny_dnn::vec_t> train_images, test_images;

  // Try to load from pre-compiled:
  std::string const trainpath = path + "/ilab5-train.bin";
  std::string const testpath = path + "/ilab5-test.bin";

  std::ifstream ifs(trainpath);
  if (ifs.is_open() == false)
  {
    // Need to create the datasets from raw images
    create_compiled(trainpath, 1, ntrain);
    create_compiled(testpath, ntrain + 2, ntest);
  }

  // Ok, the datasets:
  load_compiled(trainpath, train_labels, train_images);
  load_compiled(testpath, test_labels, test_images);

  LINFO("Start training...");
  int const n_minibatch = 48;
  int const n_train_epochs = 100;

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
std::string const & ObjectRecognitionILAB::category(size_t idx) const
{
  static std::vector<std::string> const names = { "car", "equip", "plane", "boat", "mil" };

  if (idx >= names.size()) LFATAL("Category index out of bounds");
  
  return names[idx];
}
