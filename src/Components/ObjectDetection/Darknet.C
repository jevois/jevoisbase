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

#include <jevoisbase/Components/ObjectDetection/Darknet.H>
#include <jevois/Core/Module.H>
#include <opencv2/imgproc/imgproc.hpp>

// ####################################################################################################
Darknet::Darknet(std::string const & instance, bool show_detail_params) :
    jevois::Component(instance), net(nullptr), names(nullptr), classes(0), itsReady(false),
    itsShowDetailParams(show_detail_params), itsNeedReload(false)
{
  // Get NNPACK ready to rock:
#ifdef NNPACK
  nnp_initialize();
#endif

  // Possibly hide some parameters:
  if (itsShowDetailParams == false)
  {
    dataroot::freeze();
    datacfg::freeze();
    cfgfile::freeze();
    weightfile::freeze();
    namefile::freeze();
  }
}

// ####################################################################################################
Darknet::~Darknet()
{
#ifdef NNPACK
  nnp_deinitialize();
#endif
}

// ####################################################################################################
void Darknet::onParamChange(dknet::netw const & param, dknet::Net const & newval)
{
  if (itsShowDetailParams == false)
  {
    dataroot::unFreeze();
    datacfg::unFreeze();
    cfgfile::unFreeze();
    weightfile::unFreeze();
    namefile::unFreeze();
  }

  switch (newval)
  {
  case dknet::Net::Reference:
    dataroot::set(JEVOIS_SHARE_PATH "/darknet/single");
    datacfg::set("cfg/imagenet1k.data");
    cfgfile::set("cfg/darknet.cfg");
    weightfile::set("weights/darknet.weights");
    namefile::set("");
    break;

  case dknet::Net::Tiny:
    dataroot::set(JEVOIS_SHARE_PATH "/darknet/single");
    datacfg::set("cfg/imagenet1k.data");
    cfgfile::set("cfg/tiny.cfg");
    weightfile::set("weights/tiny.weights");
    namefile::set("");
    break;
    /*
  case dknet::Net::MobileNet:
    dataroot::set(JEVOIS_SHARE_PATH "/darknet/single");
    datacfg::set("cfg/imagenet1k.data");
    cfgfile::set("cfg/mobilenet.cfg");
    weightfile::set("weights/mobilenet.weights");
    namefile::set("");
    break;

  case dknet::Net::CompMobileNet:
    dataroot::set(JEVOIS_SHARE_PATH "/darknet/single");
    datacfg::set("cfg/imagenet1k.data");
    cfgfile::set("cfg/compressmobilenet_0.1.cfg");
    weightfile::set("weights/compressmobilenet_0.1.weights");
    namefile::set("");
    break;
    */
  }

  if (itsShowDetailParams == false)
  {
    dataroot::freeze();
    datacfg::freeze();
    cfgfile::freeze();
    weightfile::freeze();
    namefile::freeze();
  }
}

// ####################################################################################################
void Darknet::onParamChange(dknet::dataroot const & param, std::string const & newval) { itsNeedReload.store(true); }
void Darknet::onParamChange(dknet::datacfg const & param, std::string const & newval) { itsNeedReload.store(true); }
void Darknet::onParamChange(dknet::cfgfile const & param, std::string const & newval) { itsNeedReload.store(true); }
void Darknet::onParamChange(dknet::weightfile const & param, std::string const & newval) { itsNeedReload.store(true); }
void Darknet::onParamChange(dknet::namefile const & param, std::string const & newval) { itsNeedReload.store(true); }

// ####################################################################################################
void Darknet::postInit()
{
  // Start a thread to load the desired network:
  loadNet();
}

// ####################################################################################################
void Darknet::loadNet()
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

#ifdef DARKNET_NNPACK
  if (net && net->threadpool) pthreadpool_destroy(net->threadpool);
#endif
  if (net) { free_network(net); net = nullptr; }
  
  // Since loading big networks can take a while, do it in a thread so we can keep streaming video in the
  // meantime. itsReady will flip to true when the load is complete.
  itsReadyFut = std::async(std::launch::async, [&]() {
      std::string root = dataroot::get(); if (root.empty() == false) root += '/';
  
      // Note: darknet expects read/write pointers to the file names...
      std::string const datacf = absolutePath(root + datacfg::get());
      std::string const cfgfil = absolutePath(root + cfgfile::get());
      std::string const weightfil = absolutePath(root + weightfile::get());

      list * options = read_data_cfg(const_cast<char *>(datacf.c_str()));
      std::string name_list = namefile::get();
      if (name_list.empty()) name_list = absolutePath(root + option_find_str(options, "names", "data/names.list"));
      else name_list = absolutePath(root + name_list);

      LINFO("Using data config from " << datacf);
      LINFO("Using cfg from " << cfgfil);
      LINFO("Using weights from " << weightfil);
      LINFO("Using names from " << name_list);

      LINFO("Getting labels...");
      names = get_labels(const_cast<char *>(name_list.c_str()));
      LINFO("Parsing network and loading weights...");
      net = load_network(const_cast<char *>(cfgfil.c_str()), const_cast<char *>(weightfil.c_str()), 0);
      if (net == nullptr) LFATAL("Failed to load darknet network and/or weights -- ABORT");
      classes = option_find_int(options, "classes", 2);

      set_batch_network(net, 1);
      srand(2222222);

#ifdef DARKNET_NNPACK
      net->threadpool = pthreadpool_create(threads::get());
#endif
      LINFO("Darknet network ready");

      free_list(options);

      // We are ready to rock:
      itsReady.store(true);
    });
}

// ####################################################################################################
void Darknet::postUninit()
{
  try { itsReadyFut.get(); } catch (...) { }

  if (net)
  {
#ifdef DARKNET_NNPACK
    if (net->threadpool) pthreadpool_destroy(net->threadpool);
#endif
    free_network(net);
  }

  free_ptrs((void**)names, classes);
}

// ####################################################################################################
bool Darknet::ready() const
{ return (itsReady.load() == true && itsNeedReload.load() == false); }

// ####################################################################################################
float Darknet::predict(cv::Mat const & cvimg, std::vector<predresult> & results)
{
  if (itsNeedReload.load()) loadNet();
  if (itsReady.load() == false) throw std::logic_error("not ready yet...");
  if (cvimg.type() != CV_8UC3) LFATAL("cvimg must have type CV_8UC3 and RGB pixels");
  
  int const c = 3; // color channels
  int const w = cvimg.cols;
  int const h = cvimg.rows;
  image im = make_image(w, h, c);
  for (int k = 0; k < c; ++k)
    for (int j = 0; j < h; ++j)
      for (int i = 0; i < w; ++i)
      {
        int const dst_index = i + w*j + w*h*k;
        int const src_index = k + c*i + c*w*j;
        im.data[dst_index] = float(cvimg.data[src_index]) * (1.0F / 255.0F);
      }

  float predtime = predict(im, results);

  free_image(im);

  return predtime;
}

// ####################################################################################################
float Darknet::predict(image & im, std::vector<predresult> & results)
{
  if (itsNeedReload.load()) loadNet();
  if (itsReady.load() == false) throw std::logic_error("not ready yet...");
  int const topn = top::get();
  float const th = thresh::get();
  results.clear();

  // Resize the network if needed:
  resize_network(net, im.w, im.h);

  // Run the predictions:
  struct timeval start, stop;
  gettimeofday(&start, 0);
  float * predictions = network_predict(net, im.data);
  gettimeofday(&stop, 0);

  float predtime = (stop.tv_sec * 1000 + stop.tv_usec / 1000) - (start.tv_sec * 1000 + start.tv_usec / 1000);

  if (net->hierarchy) hierarchy_predictions(predictions, net->outputs, net->hierarchy, 1, 1);

  // Get the top scoring predictions and push them into results:
  int indexes[topn];
  top_k(predictions, net->outputs, topn, indexes);

  for (int i = 0; i < topn; ++i)
  {
    int const index = indexes[i];
    float const score = float(predictions[index] * 100);
    if (score >= th) results.push_back(std::make_pair(score, std::string(names[index])));
    else break;
  }

  return predtime;
}

// ####################################################################################################
void Darknet::resizeInDims(int w, int h)
{
  if (itsReady.load() == false) throw std::logic_error("not ready yet...");
  resize_network(net, w, h);
}

// ####################################################################################################
void Darknet::getInDims(int & w, int & h, int & c) const
{
  if (itsReady.load() == false) throw std::logic_error("not ready yet...");
  w = net->w; h = net->h; c = net->c;
}
