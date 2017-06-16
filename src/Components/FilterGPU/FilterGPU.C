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

#include <jevoisbase/Components/FilterGPU/FilterGPU.H>
#include <jevoisbase/Components/FilterGPU/OpenGL.H>

// ####################################################################################################
FilterGPU::FilterGPU(std::string const & instance) :
    jevois::Component(instance), itsProgramChanged(false), itsQuadVertexBuffer(0), itsDisplay(EGL_NO_DISPLAY),
    itsConfig(0), itsContext(0), itsSurface(0), itsFramebufferId(0), itsRenderbufferId(0),
    itsRenderWidth(0), itsRenderHeight(0), itsRenderType(0)
{ }

// ####################################################################################################
void FilterGPU::initDisplay()
{
  // Get an EGL display connection:
  GL_CHECK(itsDisplay = eglGetDisplay(EGL_DEFAULT_DISPLAY));
  if (itsDisplay == EGL_NO_DISPLAY) LFATAL("Could not get an OpenGL display");

  // Initialize the EGL display connection:
  EGLint major, minor;
  GL_CHECK_BOOL(eglInitialize(itsDisplay, &major, &minor););
  LINFO("Initialized OpenGL-ES v" << major << '.' << minor);
  
  // Get an appropriate EGL configuration:
  EGLint num_config;
  static EGLint const cfg_attr[] =
    { EGL_SURFACE_TYPE, EGL_WINDOW_BIT, EGL_RENDERABLE_TYPE, EGL_OPENGL_ES2_BIT, EGL_NONE };
  GL_CHECK_BOOL(eglChooseConfig(itsDisplay, cfg_attr, &itsConfig, 1, &num_config));
  if (num_config < 1) LFATAL("Could not find a suitable OpenGL config");

  // Create a pbuffer surface:
  GL_CHECK(itsSurface = eglCreatePbufferSurface(itsDisplay, itsConfig, NULL));
  
  // Bind to OpenGL-ES API:
  GL_CHECK_BOOL(eglBindAPI(EGL_OPENGL_ES_API));

  // Create an EGL rendering context:
  static EGLint const ctx_attr[] = { EGL_CONTEXT_CLIENT_VERSION, 2, EGL_NONE };
  GL_CHECK(itsContext = eglCreateContext(itsDisplay, itsConfig, EGL_NO_CONTEXT, ctx_attr));
  if (itsContext == EGL_NO_CONTEXT) LFATAL("Failed to create OpenGL context");

  // Bind the context to the surface:
  GL_CHECK(eglMakeCurrent(itsDisplay, itsSurface, itsSurface, itsContext)); 
}

// ####################################################################################################
FilterGPU::~FilterGPU()
{
  // Kill our member variables before we close down OpenGL:
  itsProgram.reset(); itsSrcTex.reset();

  if (itsRenderbufferId) glDeleteRenderbuffers(1, &itsRenderbufferId);
  if (itsFramebufferId) glDeleteFramebuffers(1, &itsFramebufferId);

  glDeleteBuffers(1, &itsQuadVertexBuffer);

  // Delete surface, context, etc and close down:
  eglMakeCurrent(itsDisplay, EGL_NO_SURFACE, EGL_NO_SURFACE, EGL_NO_CONTEXT);
  if (itsSurface) eglDestroySurface(itsDisplay, itsSurface);
  eglDestroyContext(itsDisplay, itsContext);
  eglTerminate(itsDisplay);
}
  
// ####################################################################################################
void FilterGPU::setProgram(std::string const & vertex_shader, std::string const & frag_shader)
{
  // We cannot load the program here as our display may not have been initialized yet, and we may be in a different
  // thread. Let's just remember the file names and we will do the work in process():
  std::lock_guard<std::mutex> _(itsMutex);
  itsVshader = absolutePath(vertex_shader);
  itsFshader = absolutePath(frag_shader);
  itsProgramChanged = true;
  itsProgramParams.clear();
}

// ####################################################################################################
void FilterGPU::setProgramParam2f(std::string name, float val1, float val2)
{
  std::lock_guard<std::mutex> _(itsMutex);
  itsProgramParams[name] = { F2, { val1, val2 } };
  itsProgramChanged = true;
}

// ####################################################################################################
void FilterGPU::setProgramParam1f(std::string name, float val)
{
   std::lock_guard<std::mutex> _(itsMutex);
   itsProgramParams[name] = { F1, { val, 0.0F } };
  itsProgramChanged = true;
}

// ####################################################################################################
void FilterGPU::setProgramParam2i(std::string name, int val1, int val2)
{
  std::lock_guard<std::mutex> _(itsMutex);
  itsProgramParams[name] = { I2, { float(val1), float(val2) } };
  itsProgramChanged = true;
}

// ####################################################################################################
void FilterGPU::setProgramParam1i(std::string name, int val)
{
  std::lock_guard<std::mutex> _(itsMutex);
  itsProgramParams[name] = { I2, { float(val), 0.0F } };
  itsProgramChanged = true;
}

// ####################################################################################################
void FilterGPU::process(cv::Mat const & src, cv::Mat & dst)
{
  // We init the display here so that it is in the same thread as the subsequent processing, as OpenGL is not very
  // thread-friendly. Yet, see here for an alternative, which is basically to create some sort of shadow context in the
  // process() thread after having created the context in the constructor or init() thread:
  // http://stackoverflow.com/questions/11726650/egl-can-context-be-shared-between-threads
  if (itsDisplay == EGL_NO_DISPLAY) initDisplay();
  
  if (src.type() != CV_8UC1 && src.type() != CV_8UC4) LFATAL("Source pixel format must be CV_8UC1 or CV_8UC4");
  if (dst.type() != CV_8UC2 && dst.type() != CV_8UC4) LFATAL("Dest pixel format must be CV_8UC2 or CV_8UC4");
  GLuint const srcformat = (src.channels() == 4 ? GL_RGBA : GL_LUMINANCE); // GL_ALPHA also works
  GLuint const dstformat = (dst.channels() == 4 ? GL_RGBA4 : GL_RGB565);

  // Allocate our textures if needed, eg, first time we are called or sizes have changed:
  if (!itsSrcTex || itsSrcTex->Width != src.cols || itsSrcTex->Height != src.rows || itsSrcTex->Format != srcformat)
  {
    itsSrcTex.reset(new GPUtexture(src.cols, src.rows, srcformat, false));
    LINFO("Input texture " << itsSrcTex->Width << 'x' << itsSrcTex->Height << ' ' <<
          (srcformat == GL_RGBA ? "RGBA" : "LUMINANCE") << " ready.");
  }
  
  // Create our framebuffer and renderbuffer if needed:
  if (itsRenderbufferId == 0 || itsRenderWidth != dst.cols || itsRenderHeight != dst.rows ||
      itsRenderType != dst.type())
  {
    if (itsRenderbufferId) glDeleteRenderbuffers(1, &itsRenderbufferId);
    if (itsFramebufferId) glDeleteFramebuffers(1, &itsFramebufferId);
    
    GL_CHECK(glGenFramebuffers(1, &itsFramebufferId));
    GL_CHECK(glBindFramebuffer(GL_FRAMEBUFFER, itsFramebufferId));

	GL_CHECK(glGenRenderbuffers(1, &itsRenderbufferId));
	GL_CHECK(glBindRenderbuffer(GL_RENDERBUFFER, itsRenderbufferId));
	GL_CHECK(glRenderbufferStorage(GL_RENDERBUFFER, dstformat, dst.cols, dst.rows));
    GL_CHECK(glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_RENDERBUFFER, itsRenderbufferId));

    if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
      LFATAL("Framebuffer creation failed");

    itsRenderWidth = dst.cols; itsRenderHeight = dst.rows; itsRenderType = dst.type();
    LINFO("Render buffer " << itsRenderWidth << 'x' << itsRenderHeight << ' ' <<
          (dst.channels() == 2 ? "RGB565" : "RGBA") << " ready.");
  }

  // Create our vertex buffer if needed:
  if (itsQuadVertexBuffer == 0)
  {
    // Create an ickle vertex buffer:
    static GLfloat const qv[] =
      { 0.0f, 0.0f, 1.0f, 1.0f,    1.0f, 0.0f, 1.0f, 1.0f,    0.0f, 1.0f, 1.0f, 1.0f,    1.0f, 1.0f, 1.0f, 1.0f };
    
    GL_CHECK(glGenBuffers(1, &itsQuadVertexBuffer));
    glBindBuffer(GL_ARRAY_BUFFER, itsQuadVertexBuffer);
    glBufferData(GL_ARRAY_BUFFER, sizeof(qv), qv, GL_STATIC_DRAW);
    GL_CHECK(glBindBuffer(GL_ARRAY_BUFFER, 0));
  }

  // Set background color and clear buffers:
  glClearColor(0.15f, 0.25f, 0.35f, 1.0f);
  glClear(GL_COLOR_BUFFER_BIT);

  // Copy source pixel data to source texture:
  itsSrcTex->setPixels(src.data);
  
  // Tell OpenGL to render into our destination framebuffer:
  GL_CHECK(glBindFramebuffer(GL_FRAMEBUFFER, itsFramebufferId));
  GL_CHECK(glViewport(0, 0, itsRenderWidth, itsRenderHeight));

  // Load the shader program if needed:
  if (itsProgramChanged)
  {
    std::lock_guard<std::mutex> _(itsMutex);
    
    // Nuke any old program first:
    itsProgram.reset();
    
    // Create, load and compile the program:
    itsProgram.reset(new GPUprogram(itsVshader.c_str(), itsFshader.c_str()));
    itsProgramChanged = false;
    
    // Tell OpenGL to use our program:
    glUseProgram(itsProgram->id());
    
    // Set all the parameters:
    GLuint i = itsProgram->id();
    for (auto const & p : itsProgramParams)
    {
      char const * n = p.first.c_str();
      float const * v = &p.second.val[0];
      switch (p.second.type)
      {
      case F2: GL_CHECK(glUniform2f(glGetUniformLocation(i, n), v[0], v[1])); break;
      case F1: GL_CHECK(glUniform1f(glGetUniformLocation(i, n), v[0])); break;
      case I2: GL_CHECK(glUniform2i(glGetUniformLocation(i, n), int(v[0]), int(v[1]))); break;
      case I1: GL_CHECK(glUniform1i(glGetUniformLocation(i, n), int(v[0]))); break;
      default: LFATAL("Unsupported GPU program parameter type " << p.second.type);
      }
    }
  }

  if (!itsProgram) LFATAL("You need to set a program before processing frames");
  
  // Set program parameters that all programs should always use, ignore any error:
  glUniform2f(glGetUniformLocation(itsProgram->id(), "texelsize"), 1.0f / dst.cols, 1.0f / dst.rows);
  glUniform1i(glGetUniformLocation(itsProgram->id(), "tex"), 0);

  // Draw the texture onto the triangle strip, applying the program:
  GL_CHECK(glBindBuffer(GL_ARRAY_BUFFER, itsQuadVertexBuffer));
  GL_CHECK(glActiveTexture(GL_TEXTURE0));
  GL_CHECK(glBindTexture(GL_TEXTURE_2D, itsSrcTex->Id));

  GLuint loc = glGetAttribLocation(itsProgram->id(), "vertex");
  GL_CHECK(glVertexAttribPointer(loc, 4, GL_FLOAT, 0, 16, 0));
  GL_CHECK(glEnableVertexAttribArray(loc));
  GL_CHECK(glDrawArrays(GL_TRIANGLE_STRIP, 0, 4));
  
  // Copy the rendered pixels to destination image:
  GL_CHECK(glBindFramebuffer(GL_FRAMEBUFFER, itsFramebufferId));
  if (dstformat == GL_RGBA4) GL_CHECK(glReadPixels(0, 0, dst.cols, dst.rows, GL_RGBA, GL_UNSIGNED_BYTE, dst.data));
  else GL_CHECK(glReadPixels(0, 0, dst.cols, dst.rows, GL_RGB, GL_UNSIGNED_SHORT_5_6_5, dst.data));
  GL_CHECK(glBindFramebuffer(GL_FRAMEBUFFER, 0));
}

