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

#include <sstream>
#include <jevois/Debug/Log.H>

// OpenGL-ES seems to be available both for platform and regular host PCs
#include <EGL/egl.h>
#include <EGL/eglext.h>
#include <GLES2/gl2.h>
#include <GLES2/gl2ext.h>

namespace jevoisbase
{
  //! Function to write OpenGL errors in clear witout requiring GLUT
  inline std::string opengl_error(int err)
  {
    switch (err)
    {
    case GL_INVALID_ENUM: return "GL_INVALID_ENUM";
    case GL_INVALID_VALUE: return "GL_INVALID_VALUE";
    case GL_INVALID_OPERATION: return "GL_INVALID_OPERATION";
    case GL_INVALID_FRAMEBUFFER_OPERATION: return "GL_INVALID_FRAMEBUFFER_OPERATION";
    case GL_OUT_OF_MEMORY: return "GL_OUT_OF_MEMORY";
    default: std::ostringstream ss; ss << "UNKNOWN(0x" << std::hex << err << '0'; return ss.str();
    }
  }
} // namespace jevoisbase

//! Simple macro to check for OpenGL errors
/*! Inspired from here:
    http://stackoverflow.com/questions/11256470/define-a-macro-to-facilitate-opengl-command-debugging */
#define GL_CHECK(stmt) do { stmt; GLenum _err = glGetError(); \
    if (_err != GL_NO_ERROR) LFATAL("Error " << jevoisbase::opengl_error(_err) <<" in: " << #stmt); } while (0)

//! Simple macro to check for OpenGL errors when a boolean result is expected
#define GL_CHECK_BOOL(stmt) do { int _result = stmt; if (_result == 0) LERROR("Failed: " #stmt); \
    GLenum _err = glGetError();                                         \
    if (_err != GL_NO_ERROR) LFATAL("Error " << jevoisbase::opengl_error(_err) <<" in: " << #stmt); } while (0)
