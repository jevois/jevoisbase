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

#include <jevois/Core/Module.H>
#include <jevois/Debug/Timer.H>
#include <jevois/Image/RawImageOps.H>
#include <opencv2/core/core.hpp>

#include <nnpack.h>

extern "C" {
#include <darknet.h>
}

//! Detect multiple objects in scenes using the Darknet YOLO deep neural network
/*! 
    @author Laurent Itti

    @displayname Darknet YOLO
    @videomapping NONE 0 0 0 YUYV 320 240 30.0 JeVois DarknetYOLO
    @videomapping YUYV 320 260 30.0 YUYV 320 240 30.0 JeVois DarknetYOLO
    @videomapping YUYV 640 500 20.0 YUYV 640 480 20.0 JeVois DarknetYOLO
    @email itti\@usc.edu
    @address University of Southern California, HNB-07A, 3641 Watt Way, Los Angeles, CA 90089-2520, USA
    @copyright Copyright (C) 2016 by Laurent Itti, iLab and the University of Southern California
    @mainurl http://jevois.org
    @supporturl http://jevois.org/doc
    @otherurl http://iLab.usc.edu
    @license GPL v3
    @distribution Unrestricted
    @restrictions None
    \ingroup modules */
class DarknetYOLO : public jevois::Module
{
  public: 
    // ####################################################################################################
    //! Constructor
    // ####################################################################################################
    DarknetYOLO(std::string const & instance) : jevois::Module(instance)
    {
    }

    void postInit() override
    {
      char * datacfg = "/jevois/modules/JeVois/DarknetYOLO/cfg/coco.data";//(char *)(absolutePath("cfg/coco.data").c_str());
      char * cfgfile = "/jevois/modules/JeVois/DarknetYOLO/cfg/tiny-yolo.cfg";//(char *)(absolutePath("cfg/tiny-yolo.cfg").c_str());
      char * weightfile = "/jevois/modules/JeVois/DarknetYOLO/tiny-yolo.weights";(char *)(absolutePath("tiny-yolo.weights").c_str());
      LINFO("datacfg=["<<datacfg<<']');
      
      list *options = read_data_cfg(datacfg);
      char *name_list = "/jevois/modules/JeVois/DarknetYOLO/data/coco.names";//(char *)(absolutePath(option_find_str(options, "names", "data/names.list")).c_str());
      names = get_labels(name_list);

      /////alphabet = load_alphabet();
      net = parse_network_cfg(cfgfile);
      load_weights(&net, weightfile);
      
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
    virtual ~DarknetYOLO()
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
      static jevois::Timer timer("processing", 100, LOG_DEBUG);

      // Wait for next available camera image:
      jevois::RawImage const inimg = inframe.get();

      timer.start();
      
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

      // Convert the image to RGB and process:
      cv::Mat cvimg = jevois::rawimage::convertToCvRGB(inimg);

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

      image sized = letterbox_image(im, net.w, net.h);

      float nms = .4;

      layer l = net.layers[net.n-1];

      box *boxes = (box *)calloc(l.w*l.h*l.n, sizeof(box));
      float **probs = (float **)calloc(l.w*l.h*l.n, sizeof(float *));
      for (int j = 0; j < l.w*l.h*l.n; ++j) probs[j] = (float *)calloc(l.classes + 1, sizeof(float *));

      float *X = sized.data;
      struct timeval start, stop;
      gettimeofday(&start, 0);
      network_predict(net, X);
      gettimeofday(&stop, 0);
      printf("Predicted in %ld ms.\n", (stop.tv_sec * 1000 + stop.tv_usec / 1000) - (start.tv_sec * 1000 + start.tv_usec / 1000));


      float thresh = .24;
      float hier_thresh = .5;

      get_region_boxes(l, im.w, im.h, net.w, net.h, thresh, probs, boxes, 0, 0, hier_thresh, 1);

      if (nms) do_nms_obj(boxes, probs, l.w*l.h*l.n, l.classes, nms);
      //else if (nms) do_nms_sort(boxes, probs, l.w*l.h*l.n, l.classes, nms);


      ////////draw_detections(im, l.w*l.h*l.n, thresh, boxes, probs, names, alphabet, l.classes);

      save_image(im, "predictions");

      free_image(im);
      free_image(sized);
      free(boxes);
      free_ptrs((void **)probs, l.w*l.h*l.n);
      
      // Wait for paste to finish up:
      paste_fut.get();

      // Let camera know we are done processing the input image:
      inframe.done();

      // Show all the results:
      /////itsArUco->drawDetections(outimg, 3, h+5, ids, corners, rvecs, tvecs);

      // Send serial output:
      ////itsArUco->sendSerial(this, ids, corners, w, h, rvecs, tvecs);

      // Show processing fps:
      std::string const & fpscpu = timer.stop();
      jevois::rawimage::writeText(outimg, fpscpu, 3, h - 13, jevois::yuyv::White);
    
      // Send the output image with our processing results to the host over USB:
      outframe.send();
    }

    // ####################################################################################################
  protected:
    network net;
    char **names;
    image **alphabet;
 };

// Allow the module to be loaded as a shared object (.so) file:
JEVOIS_REGISTER_MODULE(DarknetYOLO);
