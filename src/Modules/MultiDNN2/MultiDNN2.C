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
#include <jevois/Image/RawImageOps.H>
#include <jevois/DNN/Pipeline.H>
#include <jevois/Util/Utils.H>
#include <jevois/Debug/Timer.H>

// icon from opencv

static jevois::ParameterCategory const ParamCateg("MultiDNN2 Options");

//! Parameter \relates MultiDNN2
JEVOIS_DECLARE_PARAMETER_WITH_CALLBACK(num, unsigned int, "Number of networks to run and display",
                                       4, jevois::Range<unsigned int>(1, 10), ParamCateg);

//! Run multiple neural networks in parallel with an overlapping display
/*! See \jvmod{DNN} for more details about the JeVois DNN framework.

    Edit the module's \b params.cfg file to select which models to run.

    You can load any model at runtime by setting the \p pipe parameter for each pipeline.

    @author Laurent Itti

    @displayname MultiDNN2
    @videomapping NONE 0 0 0.0 YUYV 640 480 15.0 JeVois MultiDNN2
    @videomapping YUYV 640 498 15.0 YUYV 640 480 15.0 JeVois MultiDNN2
    @email itti\@usc.edu
    @address University of Southern California, HNB-07A, 3641 Watt Way, Los Angeles, CA 90089-2520, USA
    @copyright Copyright (C) 2018 by Laurent Itti, iLab and the University of Southern California
    @mainurl http://jevois.org
    @supporturl http://jevois.org/doc
    @otherurl http://iLab.usc.edu
    @license GPL v3
    @distribution Unrestricted
    @restrictions None
    \ingroup modules */
class MultiDNN2 : public jevois::StdModule,
                 public jevois::Parameter<num>
{
  public: 
    // ####################################################################################################
    //! Constructor
    // ####################################################################################################
    using StdModule::StdModule;

    // ####################################################################################################
    //! Virtual destructor for safe inheritance
    // ####################################################################################################
    virtual ~MultiDNN2()
    { }

    // ####################################################################################################
    //! Processing function implementation
    // ####################################################################################################
    void doprocess(std::shared_ptr<jevois::dnn::Pipeline> pipe, jevois::InputFrame const & inframe,
                   jevois::RawImage * outimg, jevois::OptGUIhelper * helper, bool idle)
    {
      // If we have a second (scaled) image, assume this is the one we want to process:
      jevois::RawImage const & inimg = inframe.hasScaledImage() ? inframe.get2() : inframe.get();

      // Ok, process it:
      pipe->process(inimg, this, outimg, helper, idle);
    }
    
    // ####################################################################################################
    //! Processing function, no video output
    // ####################################################################################################
    virtual void process(jevois::InputFrame && inframe) override
    {
      LFATAL("todo");
    }
    
    // ####################################################################################################
    //! Processing function with video output to USB
    // ####################################################################################################
    virtual void process(jevois::InputFrame && inframe, jevois::OutputFrame && outframe) override
    {
      LFATAL("todo");
    }

#ifdef JEVOIS_PRO
    // ####################################################################################################
    //! Processing function with zero-copy and GUI on JeVois-Pro
    // ####################################################################################################
    virtual void process(jevois::InputFrame && inframe, jevois::GUIhelper & helper) override
    {
      // Compute overall frame rate, CPU usage, etc:
      static jevois::Timer timer("main", 20, LOG_DEBUG);
      std::string const & fpscpu = timer.stop();
      timer.start();

      // Start the display frame:
      unsigned short winw, winh;
      bool idle = helper.startFrame(winw, winh);
      
      // Display the camera input frame:
      int x = 0, y = 0; unsigned short w = 0, h = 0;
      helper.drawInputFrame("c", inframe, x, y, w, h, true);
        
      // Loop over all pipelines and run and display each:
      for (auto & pipe : itsPipes) doprocess(pipe, inframe, nullptr, &helper, idle);

      // Show overall frame rate and camera and display info:
      helper.iinfo(inframe, fpscpu, winw, winh);
     
      // Render the image and GUI:
      helper.endFrame();
    }
#endif
    
    // ####################################################################################################
  protected:
    void onParamChange(num const & JEVOIS_UNUSED_PARAM(param), unsigned int const & newval) override
    {
      // Try to preserve as many existing pipes as we can:
      while (itsPipes.size() < newval)
        itsPipes.emplace_back(addSubComponent<jevois::dnn::Pipeline>("p"+std::to_string(itsPipes.size())));

      while (newval < itsPipes.size()) { removeSubComponent(itsPipes.back()); itsPipes.pop_back(); }
    }

    std::vector<std::shared_ptr<jevois::dnn::Pipeline>> itsPipes;

};

// Allow the module to be loaded as a shared object (.so) file:
JEVOIS_REGISTER_MODULE(MultiDNN2);
