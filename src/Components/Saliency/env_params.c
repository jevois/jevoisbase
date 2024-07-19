/*!@file env_params.c */

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
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/Envision/env_params.c $
// $Id: env_params.c 9830 2008-06-18 18:50:22Z lior $
//

#include <jevoisbase/Components/Saliency/env_params.h>

#include <jevoisbase/Components/Saliency/env_log.h>

// ######################################################################
void env_params_set_defaults(struct env_params* envp)
{
  envp->maxnorm_type = ENV_VCXNORM_MAXNORM;
  envp->range_thresh = 0;
  envp->scale_bits = 16; /// WAS 30;
  envp->num_motion_directions = 4;
  envp->motion_thresh = 0;
  envp->flicker_thresh = 0;
  envp->multiscale_flicker = 0;
  envp->num_orientations = 4;
  envp->cs_lev_min = 2;
  envp->cs_lev_max = 4;
  envp->cs_del_min = 3;
  envp->cs_del_max = 4;
  envp->output_map_level = 4;
  envp->chan_i_weight = 255;
  envp->chan_c_weight = 255;
  envp->chan_o_weight = 255;
  envp->chan_f_weight = 255;
  envp->chan_m_weight = 255;
  
  envp->submapPreProc = 0;
  envp->submapPostNormProc = 0;
  envp->submapPostProc = 0;
  envp->user_data_preproc = 0;
  envp->user_data_postnorm = 0;
  envp->user_data_postproc = 0;
}

// ######################################################################
env_size_t env_max_cs_index(const struct env_params* envp)
{
  return (envp->cs_del_max - envp->cs_del_min + 1) * (envp->cs_lev_max - envp->cs_lev_min + 1);
}

// ######################################################################
env_size_t env_max_pyr_depth(const struct env_params* envp)
{
  return (envp->cs_lev_max + envp->cs_del_max + 1);
}

// ######################################################################
intg32 env_total_weight(const struct env_params* envp)
{
  return
    ((intg32) envp->chan_c_weight)
    + ((intg32) envp->chan_i_weight)
    + ((intg32) envp->chan_o_weight)
    + ((intg32) envp->chan_f_weight)
    + ((intg32) envp->chan_m_weight)
    ;
}

// ######################################################################
void env_params_validate(const struct env_params* envp)
{
  ENV_ASSERT(envp->maxnorm_type == ENV_VCXNORM_NONE || envp->maxnorm_type == ENV_VCXNORM_MAXNORM);
  ENV_ASSERT((envp->scale_bits+1) < (8*sizeof(intg32)));
  ENV_ASSERT(envp->num_motion_directions <= 99);
  ENV_ASSERT(envp->num_orientations <= 99);
  
  ENV_ASSERT(envp->cs_lev_max >= envp->cs_lev_min);
  ENV_ASSERT(envp->cs_del_max >= envp->cs_del_min);
  
  ENV_ASSERT(env_total_weight(envp) > 0);
}
