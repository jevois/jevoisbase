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
#include <jevois/Types/ObjReco.H>
#include <atomic>

#include <tensorflow/lite/model.h>
#include <tensorflow/lite/interpreter.h>

namespace tflow
{
  static jevois::ParameterCategory const ParamCateg("TensorFlow Options");

  //! Parameter \relates TensorFlow
  JEVOIS_DECLARE_PARAMETER_WITH_CALLBACK(netdir, std::string, "Network to load. This should be the name of a "
					 "directory within JEVOIS:/share/tensorflow/ which should contain two files: "
					 "model.tflite and labels.txt",
                                         "mobilenet_v1_224_android_quant_2017_11_08", ParamCateg);

  //! Parameter \relates TensorFlow
  JEVOIS_DECLARE_PARAMETER_WITH_CALLBACK(dataroot, std::string, "Root path for data, config, and weight files. "
                                         "If empty, use the module's path.",
                                         JEVOIS_SHARE_PATH "/tensorflow", ParamCateg);
  
  //! Parameter \relates TensorFlow
  JEVOIS_DECLARE_PARAMETER(top, unsigned int, "Max number of top-scoring predictions that score above thresh to return",
                           5, ParamCateg);

  //! Parameter \relates TensorFlow
  JEVOIS_DECLARE_PARAMETER(thresh, float, "Threshold (in percent confidence) above which predictions will be reported",
                           20.0F, jevois::Range<float>(0.0F, 100.0F), ParamCateg);

  //! Parameter \relates TensorFlow
  JEVOIS_DECLARE_PARAMETER(threads, int, "Number of parallel computation threads, or 0 for auto",
                           4, jevois::Range<int>(0, 1024), ParamCateg);

  //! Parameter \relates TensorFlow
  JEVOIS_DECLARE_PARAMETER(scorescale, float, "Scaling factors applied to recognition scores, useful for InceptionV3",
                           1.0F, ParamCateg);
}

//! Identify an object using TensorFlow deep neural network
/*! TensorFlow is a popular neural network framework. This component identifies the object in the given image crop. It
    returns the top scoring candidates.

    See https://www.tensorflow.org

    TensorFlow is a great deep learning and deep neural network framework. TensorFlow Lite (used here) is optimized for
    embedded systems like the small JeVois camera because it has a very small footprint and fewer dependencies than
    other deep neural network frameworks like MXNet, Theano, Keras, PyTorch, etc. In addition, the port of TensorFlow to
    JeVois includes acceleration using the ARM NEON multimedia instructions.

    \ingroup Components */
class TensorFlow : public jevois::Component,
		   public jevois::Parameter<tflow::netdir, tflow::dataroot, tflow::top, tflow::thresh, tflow::threads,
					    tflow::scorescale>
{
  public: 
    //! Constructor
    TensorFlow(std::string const & instance);

    //! Initialize, configure and load the network in a thread
    /*! Any call to predict() will simply throw until the network is loaded and ready */
    void postInit() override;

    //! Virtual destructor for safe inheritance
    virtual ~TensorFlow();

    //! Un-initialize and free resources
    void postUninit() override;

    //! Processing function, results are stored internally in the underlying TensorFlow network object
    /*! Expects an OpenCV byte RGB (CV_8UC3) or Gray (CV_8UC1) image, which may be further converted to float if that is
        what the network wants as input. The image dims and number of channels must match the network's input dims and
        channels.  Returns the prediction time (neural net forward pass) in milliseconds. Throws std::logic_error if the
        network is still loading and not ready. */
    float predict(cv::Mat const & cvimg, std::vector<jevois::ObjReco> & results);
    
    //! Get input width, height, channels
    /*! Throws std::logic_error if the network is still loading and not ready. */
    void getInDims(int & w, int & h, int & c);
    
    // We leave these in the open in case one wants to access the probs, names, etc but just be careful with them
    std::vector<std::string> labels;
    size_t numlabels; // labels is padded to a multiple of 16; numlabels is the actual unpadded number of labels
    std::unique_ptr<tflite::FlatBufferModel> model;
    std::unique_ptr<tflite::Interpreter> interpreter;

  protected:
    void onParamChange(tflow::netdir const & param, std::string const & newval) override;
    void onParamChange(tflow::dataroot const & param, std::string const & newval) override;
    void loadNet();
    void readLabelsFile(std::string const & fname);

    template <class T>
    void get_top_n(T * prediction, int prediction_size, std::vector<jevois::ObjReco> & top_results,
		   bool input_floating);

    std::future<void> itsReadyFut;
    std::atomic<bool> itsReady;
    std::atomic<bool> itsNeedReload;

    struct JeVoisReporter : public tflite::ErrorReporter
    {
	int Report(char const * format, va_list args) override;
    };

    JeVoisReporter itsErrorReporter;
};
