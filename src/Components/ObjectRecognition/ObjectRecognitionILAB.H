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

#include <jevoisbase/src/Components/ObjectRecognition/ObjectRecognition.H>
#include "tiny-dnn/tiny_dnn/nodes.h"

//! Object recognition CNN for ILAB-10 dataset
/*! \ingroup components */
class ObjectRecognitionILAB : public ObjectRecognition<tiny_dnn::sequential>
{
  public:
    //! Constructor, loads the given CNN, its sizes must match our (fixed) internal network structure
    /*! All network data is assumed to be in the module's path plus "tiny-dnn/<instance>". In there, we will look for
        weights.tnn, and if not found, we will train the network using data in that path and then save weights.tnn. */
    ObjectRecognitionILAB(std::string const & instance);

    //! Destructor
    virtual ~ObjectRecognitionILAB();
    
    //! Define the network structure
    virtual void define() override;
    
    //! Train the network
    virtual void train(std::string const & path) override;
    
    //! Return the name of a given category (0-based index in the vector of results)
    virtual std::string const & category(size_t idx) const override;
};

