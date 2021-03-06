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

#pragma once

#include <jevois/Component/Component.H>
#include <stdarg.h> // needed by tiny_dnn

// Defines used to optimize tiny-dnn:
#define CNN_USE_TBB
#undef CNN_USE_DOUBLE
//#define CNN_USE_NNPACK  // This still yieds compilation errors inside tiny-dnn and is not ready yet upstream

#include <tiny-dnn/tiny_dnn/config.h> // for float_t, etc. this does not include much code
#include <tiny-dnn/tiny_dnn/util/aligned_allocator.h> // for aligned_allocator
#include <tiny-dnn/tiny_dnn/util/util.h> // for index3d
#include <opencv2/core/core.hpp>

namespace tiny_dnn { template <typename NetType> class network; }

//! Abstract base class for an object recognition component
/*! This base class provides a framework to recognize objects using deep neural networks. The network is implemented
    using the tiny-dnn library, see https://github.com/tiny-dnn/tiny-dnn

    Derived classes implement a particular network by overriding the define() method of the base class. They also must
    implement the other abstract methods provided here.

    To create a new object recognition component, one would usually derive from the ObjectRecognition class template as
    opposed to deriving directly from ObjectRecognitionBase.

    Training
    --------

    Training is automatically launched if a pre-trained weight file is not found. Thus, typical workflow is, assuming
    \jvversion{1.4} or later:

    - create a new derived class and implement define(), train(), prcess(), etc in there. See ObjectRecognitionMNIST for
      example.

    - create a directory <b>/jevois/share/tiny-dnn/NAME/</b> where \a NAME is the instance name you give to your derived
      object.

    - copy the training data to that directory (as required by your implementation of train() in your derived class)

    - run `jevois-daemon` on host, make sure it has write permission to the directory you created. It will fail to load
      the pre-trained weights and will initiate training.

    - when training is complete, trained weights will be saved as <b>/jevois/share/tiny-dnn/NAME/weights.tnn</b> (for
      \jvversion{1.4} or earlier) or as <b>/jevois/share/tiny-dnn/NAME/weights.tnn.host</b> (for \jvversion{1.5} or
      later).

    Starting with \jvversion{1.5}, tiny-dnn has been updated to a recent version which uses \b cereal as a back-end to
    save networks and weights. This yields efficient loading of pre-trained networks but the binary archive format is
    not portable between Intel x64 hosts and ARM platform. Hence, you should proceed as follows:

    - run the above steps. Two files will be saved: \b weights.tnn.host (binary) and \b weights.tnn.json (portable text
      file).

    - copy \b weights.tnn.json to microSD in <b>JEVOIS:/share//tiny-dnn/NAME/</b> where \a NAME is the instance name you
      give to your derived object.

    - insert microSD into JeVois camera and launch the machine vision mode that uses your network.

    - ObjectRecognition::load() will fail to load the missing \b weights.tnn.platform (binary) and will thus revert to
      loading \b weights.tnn.json instead. It will then save \b weights.tnn.platform to microSD. Because this was done
      by the JeVois camera, \b weights.tnn.platform will now be in ARM binary format.

    - You can now copy \b weights.tnn.platform out of your microSD to <b>~/jevoisbase/share/tiny-dnn/NAME/</b> so that
      it will be flashed to microSD next time you make one using `jevois-flash-card` and you will avoid having to
      convert again next time JeVois starts.

    \ingroup components */
class ObjectRecognitionBase : public jevois::Component
{
  public:
    //! Type used by tiny-dnn for the results:
    typedef std::vector<tiny_dnn::float_t, tiny_dnn::aligned_allocator<tiny_dnn::float_t, 64> > vec_t;

    //! Constructor
    ObjectRecognitionBase(std::string const & instance);
    
    //! Virtual destructor for safe inheritance
    virtual ~ObjectRecognitionBase();

    //! Define the network structure
    /*! Derived classes must implement this function and load a network structure. */
    virtual void define() = 0;

    //! Get the input size for the current network, useful to prepare inputs to process()
    virtual tiny_dnn::index3d<size_t> insize() const = 0;
      
    //! Train the network
    /*! Derived classes must implement this function. */
    virtual void train(std::string const & path) = 0;

    //! Process an image, results are confidence for each category
    virtual vec_t process(cv::Mat const & img, bool normalize = true) = 0;

    //! Return the name of a given category (0-based index in the vector of results)
    virtual std::string const & category(size_t idx) const = 0;
};

//! Wrapper around a neural network implemented by with the tiny-dnn framework by Taiga Nomi
/*! Because tiny-dnn is an all-include package, we use the pimpl idiom here to avoid including all the tiny-dnn
    sources in the header file, and instead only include and compile them once in our ObjectRecognition.C file. 

    \ingroup components */
template <typename NetType>
class ObjectRecognition : public ObjectRecognitionBase
{
  public:
    //! Constructor allocates the (empty) network
    ObjectRecognition(std::string const & instance);
    
    //! Destructor
    virtual ~ObjectRecognition();
    
    //! Get the input size for the current network, useful to prepare inputs to process()
    virtual tiny_dnn::index3d<size_t> insize() const override;

    //! Process an image, results are confidence for each category
    vec_t process(cv::Mat const & img, bool normalize = true) override;

  protected:
    //! Initialize the network, required before one starts using it
    /*! First, we will call define(). Then, we will look in path for weights.tnn, and if not found, we will call
        train() to train the network using data in that path, and then we will save weights.tnn. Derived classes may
        implement a constructor that takes path and then calls init(path) after the base class has been constructed
        (e.g., in the body of the derived class constructor). */
    virtual void postInit() override;

    tiny_dnn::network<NetType> * net; // pointer here to avoid #include'ing tiny_dnn.h 
};
