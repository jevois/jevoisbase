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

// ####################################################################################################
Yolo::Yolo(std::string const & instance) : jevois::Component(instance), itsReady(false)
{ }

// ####################################################################################################
Yolo::~Yolo()
{ }

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
      LINFO("Parsing network...");
      net = parse_network_cfg(const_cast<char *>(cfgfil.c_str()));
      LINFO("Loading weights...");
      load_weights(&net, const_cast<char *>(weightfil.c_str()));
      
      set_batch_network(&net, 1);
      srand(2222222);
      LINFO("YOLO network ready");
  
#ifdef NNPACK
      nnp_initialize();
      net.threadpool = pthreadpool_create(4);
#endif
      itsReady.store(true);
    });
}

// ####################################################################################################
void Yolo::postUninit()
{
  try { itsReadyFut.get(); } catch (...) { }
  
#ifdef NNPACK
  pthreadpool_destroy(net.threadpool);
  nnp_deinitialize();
#endif
  if (boxes) { free(boxes); boxes = nullptr; }
  if (probs) { layer & l = net.layers[net.n-1]; free_ptrs((void **)probs, l.w * l.h * l.n); probs = nullptr; }
}

// ####################################################################################################
float Yolo::predict(cv::Mat const & cvimg)
{
  if (itsReady.load() == false) throw std::runtime_error("not ready yet...");
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
  image sized = letterbox_image(im, net.w, net.h);
      
  struct timeval start, stop;
  float * X = sized.data;

  gettimeofday(&start, 0);
  network_predict(net, X);
  gettimeofday(&stop, 0);

  float predtime = (stop.tv_sec * 1000 + stop.tv_usec / 1000) - (start.tv_sec * 1000 + start.tv_usec / 1000);

  free_image(sized);

  return predtime;
}

// ####################################################################################################
void Yolo::computeBoxes(int inw, int inh)
{
  layer & l = net.layers[net.n-1];

  if (boxes == nullptr)
    boxes = (box *)calloc(l.w * l.h * l.n, sizeof(box));

  if (probs == nullptr)
  {
    probs = (float **)calloc(l.w * l.h * l.n, sizeof(float *));
    for (int j = 0; j < l.w * l.h * l.n; ++j) probs[j] = (float *)calloc(l.classes + 1, sizeof(float));
  }

  get_region_boxes(l, inw, inh, net.w, net.h, thresh::get(), probs, boxes, 0, 0, hierthresh::get(), 1);

  float const nmsval = nms::get();
  if (nmsval) do_nms_obj(boxes, probs, l.w * l.h * l.n, l.classes, nmsval);
}

// ####################################################################################################
void Yolo::drawDetections(jevois::RawImage & outimg, int inw, int inh, int xoff, int yoff)
{
  layer & l = net.layers[net.n-1];
  int const num = l.w * l.h * l.n;

  float const thval = thresh::get();
  float const hthval = hierthresh::get();
  
  for (int i = 0; i < num; ++i)
  {
    int cls = max_index(probs[i], l.classes);
    float prob = probs[i][cls];

    if (prob > thval)
    {
      box & b = boxes[i];

      int left = xoff + (b.x - b.w / 2.0F) * inw;
      int bw = b.w * inw;
      int top = yoff + (b.y - b.h / 2.0F) * inh;
      int bh = b.h * inh;

      jevois::rawimage::drawRect(outimg, left, top, bw, bh, 2, jevois::yuyv::LightGreen);
      jevois::rawimage::writeText(outimg, jevois::sformat("%s: %.2f", names[cls], prob),
                                  left, top - 22, jevois::yuyv::LightGreen, jevois::rawimage::Font10x20);
    }
  }
}