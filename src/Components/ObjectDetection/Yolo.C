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
Yolo::~Yolo()
{ }

// ####################################################################################################
void Yolo::postInit() override
{
  // Note: darknet expects read/write pointers to the file names...
  std::string datacfg = absolutePath("cfg/voc.data");
  std::string cfgfile = absolutePath("cfg/tiny-yolo-voc.cfg");
  std::string weightfile = absolutePath("tiny-yolo-voc.weights");

      list * options = read_data_cfg(const_cast<char *>(datacfg.c_str()));
      std::string name_list = absolutePath(option_find_str(options, "names", "data/names.list"));
      names = get_labels(const_cast<char *>(name_list.c_str()));

      /////alphabet = load_alphabet();
      net = parse_network_cfg(const_cast<char *>(cfgfile.c_str()));
      load_weights(&net, const_cast<char *>(weightfile.c_str()));
      
      set_batch_network(&net, 1);
      srand(2222222);
#ifdef NNPACK
      nnp_initialize();
      net.threadpool = pthreadpool_create(4);
#endif

    }

    // ####################################################################################################
    //! Virtual destructor for safe inheritance
    // ####################################################################################################
    virtual ~Yolo()
    { }

    void postUninit() override
    {
#ifdef NNPACK
      pthreadpool_destroy(net.threadpool);
      nnp_deinitialize();
#endif


    }

    // ####################################################################################################
    //! Processing function, no video output
    // ####################################################################################################
    virtual void process(jevois::InputFrame && inframe) override
    {
      // todo
    }

    // ####################################################################################################
    //! Processing function with video output to USB
    // ####################################################################################################
    virtual void process(jevois::InputFrame && inframe, jevois::OutputFrame && outframe) override
    {
      static jevois::Profiler prof("processing", 10, LOG_INFO);

      // Wait for next available camera image:
      jevois::RawImage const inimg = inframe.get();

      prof.start();
      
      // We only handle one specific pixel format, and any image size in this module:
      unsigned int const w = inimg.width, h = inimg.height;
      inimg.require("input", w, h, V4L2_PIX_FMT_YUYV);

      // While we process it, start a thread to wait for out frame and paste the input into it:
      jevois::RawImage outimg;
      auto paste_fut = std::async(std::launch::async, [&]() {
          outimg = outframe.get();
          outimg.require("output", w, h + 60, inimg.fmt);
          jevois::rawimage::paste(inimg, outimg, 0, 0);
          jevois::rawimage::writeText(outimg, "JeVois Darknet YOLO", 3, 3, jevois::yuyv::White);
          jevois::rawimage::drawFilledRect(outimg, 0, h, w, outimg.height-h, jevois::yuyv::Black);
        });

      prof.checkpoint("paste started");
      
      // Convert the image to RGB and process:
      cv::Mat cvimg = jevois::rawimage::convertToCvRGB(inimg);

      prof.checkpoint("convert to rgb");
      
      int const c = 3; // color channels
      image im = make_image(w, h, c);
      for (int k = 0; k < c; ++k)
        for (int j = 0; j < h; ++j)
          for (int i = 0; i < w; ++i)
          {
            int dst_index = i + w*j + w*h*k;
            int src_index = k + c*i + c*w*j;
            im.data[dst_index] = float(cvimg.data[src_index]) / 255.0F;
          }

      prof.checkpoint("rgb to float");
      
      image sized = letterbox_image(im, net.w, net.h);
      LINFO("sized image is " << sized.w << 'x' << sized.h);
      
      prof.checkpoint("letterbox");

      // Wait for paste to finish up:
      paste_fut.get();

      // Let camera know we are done processing the input image:
      inframe.done();

      float nms = .4;

      layer l = net.layers[net.n-1];

      box *boxes = (box *)calloc(l.w * l.h * l.n, sizeof(box));
      float **probs = (float **)calloc(l.w * l.h * l.n, sizeof(float *));
      for (int j = 0; j < l.w * l.h * l.n; ++j) probs[j] = (float *)calloc(l.classes + 1, sizeof(float)); // bugfix was float*

      float *X = sized.data;
      prof.checkpoint("paste done");
      network_predict(net, X);
      prof.checkpoint("nn done");
      
      float thresh = .24;
      float hier_thresh = .5;

      //get_region_boxes(l, im.w, im.h, net.w, net.h, thresh, probs, boxes, nullptr, 0, nullptr, hier_thresh, 1);
      get_region_boxes(l, im.w, im.h, net.w, net.h, thresh, probs, boxes, 0, 0, hier_thresh, 1);

      if (nms) do_nms_obj(boxes, probs, l.w*l.h*l.n, l.classes, nms);

      // Show all the results:
      drawDetections(outimg, l.w * l.h * l.n, thresh, boxes, probs, names, l.classes);

      // Show processing fps:
      //std::string const & fpscpu = timer.stop();
      //jevois::rawimage::writeText(outimg, fpscpu, 3, h - 13, jevois::yuyv::White);

      prof.checkpoint("draw done");
      
      // Send the output image with our processing results to the host over USB:
      outframe.send();

      // Cleanup:
      free_image(im);
      free_image(sized);
      free(boxes);
      free_ptrs((void **)probs, l.w*l.h*l.n);

      prof.stop();
    }

    // ####################################################################################################
  protected:

    void drawDetections(jevois::RawImage & im, int num, float thresh, box * boxes, float ** probs,
                         char ** names, int classes)
    {
      //LINFO("got " << num << " detections");
      // Adapted for YUYV image from draw_detections() in image.c of darknet
      for (int i = 0; i < num; ++i)
      {
        int cls = max_index(probs[i], classes);
        float prob = probs[i][cls];
        // LINFO("det " << i << " prob " << prob*100 << " name " << names[cls]);
        if (prob > thresh)
        {
          //printf("%d %s: %.0f%%\n", i, names[cls], prob*100);
          printf("%s: %.0f%%\n", names[cls], prob*100);
          box & b = boxes[i];

          int left = (b.x - b.w / 2.0F) * im.width;
          int bw = b.w * im.width;
          int top = (b.y - b.h / 2.0F) * im.height;
          int bh = b.h * im.height;

          jevois::rawimage::drawRect(im, left, top, bw, bh, 2, jevois::yuyv::LightGreen);
          jevois::rawimage::writeText(im, names[cls], left + 4, top + 4, jevois::yuyv::LightGreen);
        }
      }
    }
    
    network net;
    char **names;
 };

// Allow the module to be loaded as a shared object (.so) file:
JEVOIS_REGISTER_MODULE(Yolo);
