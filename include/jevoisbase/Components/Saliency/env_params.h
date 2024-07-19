/*!@file env_params.h */

// //////////////////////////////////////////////////////////////////// //
// The iLab Neuromorphic Vision C++ Toolkit - Copyright (C) 2000-2005   //
// by the University of Southern California (USC) and the iLab at USC.  //
// See http://iLab.usc.edu for information about this project.          //
// //////////////////////////////////////////////////////////////////// //
// Major portions of the iLab Neuromorphic Vision Toolkit are protected //
// under the U.S. patent ``Computation of Intrinsic Perceptual Saliency //
// in Visual Environments, and Applications'' by Christof Koch and      //
// Laurent Itti, California Institute of Technology, 2001 (patent       //
// pending; application number 09/912,225 filed July 23, 2001; see      //
// http://pair.uspto.gov/cgi-bin/final/home.pl for current status).     //
// //////////////////////////////////////////////////////////////////// //
// This file is part of the iLab Neuromorphic Vision C++ Toolkit.       //
//                                                                      //
// The iLab Neuromorphic Vision C++ Toolkit is free software; you can   //
// redistribute it and/or modify it under the terms of the GNU General  //
// Public License as published by the Free Software Foundation; either  //
// version 2 of the License, or (at your option) any later version.     //
//                                                                      //
// The iLab Neuromorphic Vision C++ Toolkit is distributed in the hope  //
// that it will be useful, but WITHOUT ANY WARRANTY; without even the   //
// implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR      //
// PURPOSE.  See the GNU General Public License for more details.       //
//                                                                      //
// You should have received a copy of the GNU General Public License    //
// along with the iLab Neuromorphic Vision C++ Toolkit; if not, write   //
// to the Free Software Foundation, Inc., 59 Temple Place, Suite 330,   //
// Boston, MA 02111-1307 USA.                                           //
// //////////////////////////////////////////////////////////////////// //
//
// Primary maintainer for this file: Rob Peters <rjpeters at usc dot edu>
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/Envision/env_params.h $
// $Id: env_params.h 9830 2008-06-18 18:50:22Z lior $
//

#pragma once

#include <jevoisbase/Components/Saliency/env_config.h>
#include <jevoisbase/Components/Saliency/env_types.h>
#include <jevoisbase/Components/Saliency/env_image.h>

struct env_params
{
    enum env_maxnorm_type maxnorm_type;
    intg32 range_thresh;
    env_size_t scale_bits;
    env_size_t num_motion_directions;
    byte motion_thresh;
    byte flicker_thresh;
    byte multiscale_flicker;
    env_size_t num_orientations;  //!< number of Gabor subchannels
    env_size_t cs_lev_min;
    env_size_t cs_lev_max;
    env_size_t cs_del_min;
    env_size_t cs_del_max;
    env_size_t output_map_level;  ///< the pyramid level at which the feature map is taken
    byte chan_i_weight;
    byte chan_c_weight;
    byte chan_o_weight;
    byte chan_f_weight;
    byte chan_m_weight;

    int (*submapPreProc)(const char* tagName,
                         env_size_t clev, env_size_t slev,
                         struct env_image* submap,
                         const struct env_image* center,
                         const struct env_image* surround,
                         void * user_data);
    int (*submapPostNormProc)(const char* tagName,
                              env_size_t clev, env_size_t slev,
                              struct env_image *submap,
                              const struct env_image* center,
                              const struct env_image* surround,
                              void * user_data);
    int (*submapPostProc)(const char* tagName,
                          struct env_image *cmap,
                          void * user_data);
    void * user_data_preproc;
    void * user_data_postnorm;
    void * user_data_postproc;
};

#ifdef __cplusplus
extern "C"
{
#endif
  
  void env_params_set_defaults(struct env_params* envp);
  
  env_size_t env_max_cs_index(const struct env_params* envp);
  
  env_size_t env_max_pyr_depth(const struct env_params* envp);
  
  intg32 env_total_weight(const struct env_params* envp);
  
  void env_params_validate(const struct env_params* envp);
  
#ifdef __cplusplus
}
#endif
