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

#pragma once

#include <jevois/Component/Component.H>
#include <jevois/Image/RawImageOps.H>
#include <jevois/Types/Enum.H>
#include <atomic>

#include <nnpack.h>

extern "C" {
#include <darknet.h>
}

namespace dknet
{
  static jevois::ParameterCategory const ParamCateg("Darknet Options");

  //! Enum \relates Darknet
  JEVOIS_DEFINE_ENUM_CLASS(Net, (Reference) (Tiny) ) // NOT READY YET: (MobileNet) (CompMobileNet) )

  //! Parameter \relates Darknet
  JEVOIS_DECLARE_PARAMETER_WITH_CALLBACK(netw, Net, "Network to load. This meta-parameter sets parameters "
                                         "dataroot, datacfg, cfgfile, weightfile, and namefile for the chosen network.",
                                         Net::Tiny, Net_Values, ParamCateg);

  //! Parameter \relates Darknet
  JEVOIS_DECLARE_PARAMETER_WITH_CALLBACK(dataroot, std::string, "Root path for data, config, and weight files. "
                                         "If empty, use the module's path.",
                                         JEVOIS_SHARE_PATH "/darknet/single", ParamCateg);
  
  //! Parameter \relates Darknet
  JEVOIS_DECLARE_PARAMETER_WITH_CALLBACK(datacfg, std::string, "Data configuration file (if relative, relative to "
                                         "dataroot)",
                                         "cfg/imagenet1k.data", ParamCateg);

  //! Parameter \relates Darknet
  JEVOIS_DECLARE_PARAMETER_WITH_CALLBACK(cfgfile, std::string, "Network configuration file (if relative, relative to "
                                         "dataroot)",
                                         "cfg/tiny.cfg", ParamCateg);

  //! Parameter \relates Darknet
  JEVOIS_DECLARE_PARAMETER_WITH_CALLBACK(weightfile, std::string, "Network weights file (if relative, relative to "
                                         "dataroot)",
                                         "weights/tiny.weights", ParamCateg);

  //! Parameter \relates Darknet
  JEVOIS_DECLARE_PARAMETER_WITH_CALLBACK(namefile, std::string, "Category names file, or empty to fetch it from the "
                                         "network "
                                         "config file (if relative, relative to dataroot)",
                                         "", ParamCateg);

  //! Parameter \relates Darknet
  JEVOIS_DECLARE_PARAMETER(top, unsigned int, "Max number of top-scoring predictions that score above thresh to return",
                           5, ParamCateg);

  //! Parameter \relates Darknet
  JEVOIS_DECLARE_PARAMETER(thresh, float, "Threshold (in percent confidence) above which predictions will be reported",
                           20.0F, jevois::Range<float>(0.0F, 100.0F), ParamCateg);

  //! Parameter \relates Darknet
  JEVOIS_DECLARE_PARAMETER(threads, int, "Number of parallel computation threads",
                           6, jevois::Range<int>(1, 1024), ParamCateg);
}

//! Identify an object using Darknet deep neural network
/*! Darknet is a popular neural network framework. This component identifies the object in the given image crop. It
    returns the top scoring candidates.

    See https://pjreddie.com/darknet

    Darknet is a great, bare-metal deep learning and deep neural network framework. It is great for embedded systems
    like the small JeVois camera because it has a very small footprint and fewer dependencies than other deep neural
    network frameworks like Tensorflow, MXNet, Theano, Keras, PyTorch, etc. In addition, the port of Darknet to JeVois
    includes acceleration using the ARM NEON multimedia instructions through the popular NNPACK neural network
    acceleration package.

    \ingroup Components */
class Darknet : public jevois::Component,
                jevois::Parameter<dknet::netw, dknet::dataroot, dknet::datacfg, dknet::cfgfile,
                                  dknet::weightfile, dknet::namefile, dknet::top, dknet::thresh, dknet::threads>
{
  public: 
    //! Constructor
    /*! if show_detail_params is false, the parameters dataroot, datacfg, cfgfile, weightfile, and namefile are hidden
        and users can just use the parameter network to set various predefined networks. */
    Darknet(std::string const & instance, bool show_detail_params = false);

    //! Initialize, configure and load the network in a thread
    /*! Any call to process() will simply throw until the network is loaded and ready */
    void postInit() override;

    //! Virtual destructor for safe inheritance
    virtual ~Darknet();

    //! Un-initialize and free resources
    void postUninit() override;

    //! Typedef for one result: score, category name
    typedef std::pair<float, std::string> predresult;
    
    //! Processing function, results are stored internally in the underlying Darknet network object
    /*! This version expects an OpenCV RGB byte image which will be converted to float RGB planar, and which must
        already have the same dims as the input layer of the network (see indims() to help with that).  Returns the
        prediction time (neural net forward pass) in milliseconds. Throws std::logic_error if the network is still
        loading and not ready. */
    float predict(cv::Mat const & cvimg, std::vector<predresult> & results);
    
    //! Processing function, results are stored internally in the underlying Darknet network object
    /*! This version expects a Darknet image input, RGB float planar normalized to [0..1], with same dims as network
        input dims. Returns the prediction time (neural net forward pass) in milliseconds. Throws std::logic_error if
        the network is still loading and not ready. */
    float predict(image & im, std::vector<predresult> & results);

    //! Get input width, height, channels
    /*! Throws std::logic_error if the network is still loading and not ready. */
    void getInDims(int & w, int & h, int & c);
    
    // We leave these in the open in case one wants to access the probs, names, etc but just be careful with them
    network net;
    char ** names = nullptr;
    int classes;
    
  protected:
    void onParamChange(dknet::netw const & param, dknet::Net const & newval);
    void onParamChange(dknet::dataroot const & param, std::string const & newval);
    void onParamChange(dknet::datacfg const & param, std::string const & newval);
    void onParamChange(dknet::cfgfile const & param, std::string const & newval);
    void onParamChange(dknet::weightfile const & param, std::string const & newval);
    void onParamChange(dknet::namefile const & param, std::string const & newval);
    void loadNet();
    
    std::future<void> itsReadyFut;
    std::atomic<bool> itsReady;
    bool const itsShowDetailParams;
    std::atomic<bool> itsNeedReload;
 };
