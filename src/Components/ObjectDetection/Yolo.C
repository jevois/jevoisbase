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

#include <jevoisbase/Components/ObjectDetection/Yolo.H>
#include <jevois/Core/Module.H>

// ####################################################################################################
Yolo::Yolo(std::string const & instance) : jevois::Component(instance), net(nullptr), names(nullptr), nboxes(0),
    dets(nullptr), classes(0), map(nullptr), itsReady(false)
{
  // Get NNPACK ready to rock:
#ifdef NNPACK
  nnp_initialize();
#endif
}

// ####################################################################################################
Yolo::~Yolo()
{
#ifdef NNPACK
  nnp_deinitialize();
#endif
}

// ####################################################################################################
void Yolo::postInit()
{
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
      
      char * mapf = option_find_str(options, "map", 0);
      if (mapf) map = read_map(mapf);

      LINFO("Parsing network and loading weights...");

      net = load_network(const_cast<char *>(cfgfil.c_str()), const_cast<char *>(weightfil.c_str()), 0);

      if (net == nullptr)
      {
	free_list(options);
	if (map) { free(map); map = nullptr; }
	LFATAL("Failed to load YOLO network and/or weights -- ABORT");
      }

      classes = option_find_int(options, "classes", 2);

      set_batch_network(net, 1);
      srand(2222222);
      LINFO("YOLO network ready");
  
#ifdef DARKNET_NNPACK
      net->threadpool = pthreadpool_create(threads::get());
#endif
      free_list(options);
      itsReady.store(true);
    });
}

// ####################################################################################################
void Yolo::postUninit()
{
  if (itsReadyFut.valid()) try { itsReadyFut.get(); } catch (...) { }

  if (dets) { free_detections(dets, nboxes); dets = nullptr; nboxes = 0; }

  if (map) { free(map); map = nullptr; }

  if (net)
  {
#ifdef DARKNET_NNPACK
    if (net->threadpool) pthreadpool_destroy(net->threadpool);
#endif
    free_network(net);
    net = nullptr;
  }

  free_ptrs((void**)names, classes);
  names = nullptr; classes = 0;
}

// ####################################################################################################
float Yolo::predict(cv::Mat const & cvimg)
{
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
        int dst_index = i + w*j + w*h*k;
        int src_index = k + c*i + c*w*j;
        im.data[dst_index] = float(cvimg.data[src_index]) / 255.0F;
      }

  float predtime = predict(im);

  free_image(im);

  return predtime;
}

// ####################################################################################################
float Yolo::predict(image & im)
{
  image sized; bool need_free = false;
  if (im.w == net->w && im.h == net->h) sized = im;
  else { sized = letterbox_image(im, net->w, net->h); need_free = true; }
      
  struct timeval start, stop;

  gettimeofday(&start, 0);
  network_predict(net, sized.data);
  gettimeofday(&stop, 0);

  float predtime = (stop.tv_sec * 1000 + stop.tv_usec / 1000) - (start.tv_sec * 1000 + start.tv_usec / 1000);

  if (need_free) free_image(sized);

  return predtime;
}

// ####################################################################################################
void Yolo::computeBoxes(int inw, int inh)
{
  layer & l = net->layers[net->n-1];

  if (dets) { free_detections(dets, nboxes); dets = nullptr; nboxes = 0; }

  dets = get_network_boxes(net, 1, 1, thresh::get() * 0.01F, hierthresh::get() * 0.01F, map, 0, &nboxes);

  float const nmsval = nms::get() * 0.01F;

  if (nmsval) do_nms_sort(dets, nboxes, l.classes, nmsval);
}

// ####################################################################################################
void Yolo::drawDetections(jevois::RawImage & outimg, int inw, int inh, int xoff, int yoff)
{
  float const thval = thresh::get();

  for (int i = 0; i < nboxes; ++i)
  {
    // For each detection, We need to get a list of labels and probabilities, sorted by score:
    std::vector<std::pair<float, std::string> > data;
  
    for (int j = 0; j < classes; ++j)
    {
      float const p = dets[i].prob[j] * 100.0F;
      if (p > thval) data.push_back(std::make_pair(p, names[j]));
    }

    // End here if nothing above threshold:
    if (data.empty()) continue;
    
    // Sort in ascending order:
    std::sort(data.begin(), data.end(), [](auto a, auto b) { return (a.first < b.first); });

    // Create our display label:
    std::string labelstr;
    for (auto itr = data.rbegin(); itr != data.rend(); ++itr)
    {
      if (labelstr.empty() == false) labelstr += ", ";
      labelstr += jevois::sformat("%s:%.1f", itr->second.c_str(), itr->first);
    }

    box const & b = dets[i].bbox;

    int const left = std::max(xoff, int(xoff + (b.x - b.w / 2.0F) * inw + 0.499F));
    int const bw = std::min(inw, int(b.w * inw + 0.499F));
    int const top = std::max(yoff, int(yoff + (b.y - b.h / 2.0F) * inh + 0.499F));
    int const bh = std::min(inh, int(b.h * inh + 0.499F));
    
    jevois::rawimage::drawRect(outimg, left, top, bw, bh, 2, jevois::yuyv::LightGreen);
    jevois::rawimage::writeText(outimg, labelstr,
				left + 4, top + 2, jevois::yuyv::LightGreen, jevois::rawimage::Font10x20);
  }
}

// ####################################################################################################
void Yolo::sendSerial(jevois::StdModule * mod, int inw, int inh)
{
  float const thval = thresh::get();

  for (int i = 0; i < nboxes; ++i)
  {
    // For each detection, We need to get a list of labels and probabilities, sorted by score:
    std::vector<std::pair<float, std::string> > data;
  
    for (int j = 0; j < classes; ++j)
    {
      float const p = dets[i].prob[j] * 100.0F;
      if (p > thval) data.push_back(std::make_pair(p, names[j]));
    }

    // End here if nothing above threshold:
    if (data.empty()) continue;
    
    // Sort in ascending order:
    std::sort(data.begin(), data.end(), [](auto a, auto b) { return (a.first < b.first); });

    // The last one will be the returned 
    auto const & last = data.back();
    std::string const name = jevois::sformat("%s:%.1f", last.second.c_str(), last.first);
    data.pop_back();

    std::string extra;
    for (auto itr = data.rbegin(); itr != data.rend(); ++itr)
      extra += jevois::sformat("%s:%.1f ", itr->second.c_str(), itr->first);
    
    box const & b = dets[i].bbox;

    int const left = (b.x - b.w / 2.0F) * inw;
    int const bw = b.w * inw;
    int const top = (b.y - b.h / 2.0F) * inh;
    int const bh = b.h * inh;

    mod->sendSerialImg2D(inw, inh, left, top, bw, bh, name, extra);
  }
}

// ####################################################################################################
void Yolo::resizeInDims(int w, int h)
{
  if (itsReady.load() == false) throw std::logic_error("not ready yet...");
  resize_network(net, w, h);
}

// ####################################################################################################
void Yolo::getInDims(int & w, int & h, int & c) const
{
  if (itsReady.load() == false) throw std::logic_error("not ready yet...");
  w = net->w; h = net->h; c = net->c;
}
