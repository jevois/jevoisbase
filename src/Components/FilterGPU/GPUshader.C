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

#include <jevoisbase/Components/FilterGPU/GPUshader.H>

// ####################################################################################################
GPUshader::GPUshader() :
    Src(nullptr), Id(0)
{ }

// ####################################################################################################
GPUshader::~GPUshader()
{
  if (Id) glDeleteShader(Id);
  if (Src) delete[] Src;
}

// ####################################################################################################
GLuint GPUshader::id() const
{ return Id; }

// ####################################################################################################
void GPUshader::load(char const * filename, GLuint type)
{
  if (Src) delete [] Src;
  if (Id) glDeleteShader(Id);
  
  // Cheeky bit of code to read the whole file into memory:
  FILE * f = fopen(filename, "rb"); if (f == nullptr) PLFATAL("Failed to read file " << filename);
  fseek(f, 0, SEEK_END);
  int sz = ftell(f);
  fseek(f, 0, SEEK_SET);
  Src = new GLchar[sz+1];
  fread(Src, 1, sz, f);
  Src[sz] = 0; // null terminate it
  fclose(f);

  // now create and compile the shader
  GL_CHECK(Id = glCreateShader(type));
  GL_CHECK(glShaderSource(Id, 1, (const GLchar**)&Src, 0));
  GL_CHECK(glCompileShader(Id));

  // Compilation check
  GLint compiled; glGetShaderiv(Id, GL_COMPILE_STATUS, &compiled);
  if (compiled == 0)
  {
    GLint loglen = 2048; char log[loglen];
    glGetShaderInfoLog(Id, loglen, &loglen, &log[0]); log[loglen] = '\0';
    LERROR("Failed to compile shader from file " << filename << ", Log: " << &log[0]);
    glDeleteShader(Id);
  }
  LINFO("Compiled shader from file " << filename);
}
