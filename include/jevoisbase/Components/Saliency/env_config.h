/*!@file env_config.h */

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
// $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/src/Envision/env_config.h $
// $Id: env_config.h 8101 2007-03-13 06:07:28Z lior $
//

#pragma once

// HACK
#define INT_IS_32_BITS

#if defined(LONG_IS_32_BITS)
#  define ENV_INTG32_TYPE long
#elif defined(INT_IS_32_BITS)
#  define ENV_INTG32_TYPE int
#else
#  error no 32-bit integer type available
#endif

#if defined(LONG_LONG_IS_64_BITS)
#  define ENV_INTG64_TYPE long long
#else
// just leave ENV_INTG64_TYPE undefined and we'll get along without a 64-bit type
#endif

#include <stdlib.h>
inline void * env_allocate(unsigned long nbytes) { return malloc(nbytes); }
inline void env_deallocate(void * mem) { free(mem); }
