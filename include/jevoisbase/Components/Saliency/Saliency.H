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
#include <jevois/Image/RawImage.H>
#include <jevois/Debug/Profiler.H>

#include <opencv2/core/core.hpp>

#include <jevoisbase/src/Components/Saliency/env_image.h>
#include <jevoisbase/src/Components/Saliency/env_params.h>
#include <jevoisbase/src/Components/Saliency/env_math.h>
#include <jevoisbase/src/Components/Saliency/env_pyr.h>
#include <jevoisbase/src/Components/Saliency/env_motion_channel.h>

#include <mutex>
#include <condition_variable>
 
namespace saliency
{
  static jevois::ParameterCategory const ParamCateg("Saliency/Gist Options");

  //! Parameter \relates Saliency
  JEVOIS_DECLARE_PARAMETER(cweight, byte, "Color channel weight", 255, ParamCateg);

  //! Parameter \relates Saliency
  JEVOIS_DECLARE_PARAMETER(iweight, byte, "Intensity channel weight", 255, ParamCateg);

  //! Parameter \relates Saliency
  JEVOIS_DECLARE_PARAMETER(oweight, byte, "Orientation channel weight", 255, ParamCateg);

  //! Parameter \relates Saliency
  JEVOIS_DECLARE_PARAMETER(fweight, byte, "Flicker channel weight", 255, ParamCateg);

  //! Parameter \relates Saliency
  JEVOIS_DECLARE_PARAMETER(mweight, byte, "Motion channel weight", 255, ParamCateg);

  //! Parameter \relates Saliency
  JEVOIS_DECLARE_PARAMETER(centermin, size_t, "Lowest (finest) of the 3 center scales", 2, ParamCateg);

  //! Parameter \relates Saliency
  JEVOIS_DECLARE_PARAMETER(deltamin, size_t, "Lowest (finest) of the 2 center-surround delta scales", 3, ParamCateg);

  //! Parameter \relates Saliency
  JEVOIS_DECLARE_PARAMETER(smscale, size_t, "Scale of the saliency map", 4, ParamCateg);

  //! Parameter \relates Saliency
  JEVOIS_DECLARE_PARAMETER(mthresh, byte, "Motion threshold", 0, ParamCateg);

  //! Parameter \relates Saliency
  JEVOIS_DECLARE_PARAMETER(fthresh, byte, "Flicker threshold", 0, ParamCateg);

  //! Parameter \relates Saliency
  JEVOIS_DECLARE_PARAMETER(msflick, bool, "Use multiscale flicker computation", false, ParamCateg);
}

//! Simple wrapper class around Rob Peter's C-optimized, fixed-point-math visual saliency code
/*! This component implements a visual saliency detector using the algorithm from Itti et al., IEEE Trans PAMI,
    1998. The saliency map highlights locations in an video stream that stand out and would attract the attention of a
    human. In addition, this component computes the gist of the scene, a low-dimentional statistical summary of the
    whole scene that can be used for scene classification.
    
    Note that the Saliency component here imposes some restrictions on the saliency computations:
    
    - we enforce that always 6 feature maps are computed for each channel, which is obtained by using 3 center scales
      and 2 center-surround scale deltas. In this component, only the finest center scale and smallest delta can be
      specified, as well as the saliency map scale. This is so that the gist vector can have a fixed size.
    
    - number of orientations in the orientation channel is fixed at 4, number of directions in the motion channel is
      fixed at 4. This is again to obtain a fixed gist vector size.
    
    - we always consider all of C, I O, F and M channels as opposed to having a more dynamic collection of channels as
      done in other implementations of this model (see, e.g., http://iLab.usc.edu/toolkit/). This is again so that we
      have fixed gist size and available output maps. Note that some channels will not be computed if their weight is
      set to zero, and instead the maps will be empty and the gist entries will be zeroed out. 
    
    See the research paper at http://ilab.usc.edu/publications/doc/Itti_etal98pami.pdf
    \ingroup components*/
class Saliency : public jevois::Component,
                 public jevois::Parameter<saliency::cweight, saliency::iweight, saliency::oweight, saliency::fweight,
                                          saliency::mweight, saliency::centermin, saliency::deltamin, saliency::smscale,
                                          saliency::mthresh, saliency::fthresh, saliency::msflick>
{
  public:
    //! Constructor
    Saliency(std::string const & instance);
    
    //! Destructor
    virtual ~Saliency();
    
    //! Process a raw YUYV image. Results are stored in the Saliency class.
    void process(jevois::RawImage const & input, bool do_gist);

    //! Process an RGB image. Results are stored in the Saliency class.
    void process(cv::Mat const & input, bool do_gist);

    //! Wait until process() is done using the input image
    /*! This assumes that you are running process() in a different thread and here just want to wait until the initial
        processing that uses the input image is complete, so you can return that input image to the camera driver. */
    void waitUntilDoneWithInput() const;
    
    struct env_image salmap; //!< The saliency map
    
    //! Get location and value of max point in the saliency map
    void getSaliencyMax(int & x, int & y, intg32 & value);

    //! Inhibit the saliency map around a point, sigma is in pixels at the sacle of the map
    void inhibitionOfReturn(int const x, int const y, float const sigma);
    
    struct env_image intens;
    struct env_image color;
    struct env_image ori;
    struct env_image flicker;
    struct env_image motion;
    
    //! Gist vector has 1152 entries, 72 feature maps * 16 values/map
    /*! The 16 values are in raster order. If a feature weight is zero, the corresponding gist values are all
        zero. The feature maps are in order:
        
        red/green (6*16), blue/yellow (6*16), intens (6*16), ori (6*4*16), flicker (6*16), motion (6*4*16)
        
        so the offsets are:
        red/green:   offset 0 len 6*16
        blue/yellow: offset 1*6*16 len 6*16
        intens:      offset 2*6*16 len 6*16
        ori0:        offset 3*6*16 len 6*16
        ori1:        offset 4*6*16 len 6*16
        ori2:        offset 5*6*16 len 6*16
        ori3:        offset 6*6*16 len 6*16
        flicker:     offset 7*6*16 len 6*16
        motion0:     offset 8*6*16 len 6*16
        motion1:     offset 9*6*16 len 6*16
        motion2:     offset 10*6*16 len 6*16
        motion3:     offset 11*6*16 len 6*16  */
    unsigned char * gist;
    size_t const gist_size;
    
    // Helper struct for gist computation, of no use to end users
    struct visitor_data { unsigned char * gist; size_t gist_size; env_params * envp; };
    
  private:
    struct env_params envp;
    
    void combine_output(struct env_image* chanOut, const intg32 iweight, struct env_image* result);
    struct env_math imath;
    struct env_image prev_input;
    struct env_pyr prev_lowpass5;
    struct env_motion_channel motion_chan;
    std::mutex itsMtx;
    
    // locally rewritten to use our thread pool
    void env_mt_chan_orientation(const char* tagName, const struct env_image* img, env_chan_status_func* status_func,
                                 void* status_userdata, struct env_image* result);
    
    // locally rewritten to use our thread pool
    void env_mt_motion_channel_input(struct env_motion_channel* chan, const char* tagName,
                                     const struct env_dims inputdims, struct env_pyr* lowpass5,
                                     env_chan_status_func* status_func, void* status_userdata,
                                     struct env_image* result);
    
    void processStart(struct env_dims const & dims, bool do_gist);
    
    visitor_data itsVisitorData;
    jevois::Profiler itsProfiler;

    //! A mutex used to signal when the raw image is not needed anymore by process() (RawImage version)
    mutable std::mutex itsRawImageMtx;
    
    //! A condition variable that gets notified during process(RawImage...) when raw image not needed anymore
    mutable std::condition_variable itsRawImageCond;
    mutable bool itsInputDone;
};

//! Draw a saliency map or feature map in a YUYV image
/*! \relates Saliency */
void drawMap(jevois::RawImage & img, env_image const * fmap, unsigned int xoff, unsigned int yoff, unsigned int scale);

//! Draw a saliency map or feature map in a YUYV image, applying some right bitshift to values
/*! \relates Saliency */
void drawMap(jevois::RawImage & img, env_image const * fmap, unsigned int xoff, unsigned int yoff, unsigned int scale,
             unsigned int bitshift);

//! Draw a gist vector in a YUYV image as a rectangle of width width*scale and correct height
/*! \relates Saliency */
void drawGist(jevois::RawImage & img, unsigned char const * gist, size_t gistsize, unsigned int xoff, unsigned int yoff,
              unsigned int width, unsigned int scale);
    
