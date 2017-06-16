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
#include <jevois/Image/RawImageOps.H>

#include <jevoisbase/Components/FilterGPU/GPUtexture.H>
#include <jevoisbase/Components/FilterGPU/GPUprogram.H>

#include <map>

//! Simple image filtering using OpenGL-ES on the GPU
/*! This class first copies the given input image into an OpenGL texture, then applies OpenGL-ES vertex and fragment
    shaders to achieve some image processing, and finally gets the resulting pixels back into an image.
    
    We support two types of input:

    - CV_8UC1 greyscale image. This is uploaded to GPU as a luminance texture. In the shaders, R=G=B and alpha=1.
    - CV-8UC4 color images. This is uploaded to GPU as an RGBA image. In the shaders, R,G,B,A are as in the input.

    We support two types of output:

    - CV_8UC2 RGB565 images. This is useful to avoid any conversion of color results. GPU results are directly rendered
      into an RGB565 color buffer, which may directly be the buffer of an RGB565 output RawImage, i.e., the pixel memory
      was allocated and "belongs to" the USB gadget driver. Thus, the rendered pixels can be directly read back from the
      GPU into the RawImage and send over USB with no additional copy or conversion. See DemoGPU for an example of this.
    - CV-8UC4 RGBA images. This may be used either for higher-accuracy color output, but may require conversion since we
      do not support sending RGBA images over the USB link. Using jevois::rawimage::unpackCvRGBAtoGrayRawImage() one may
      unpack such a packed RGBA image into 4 single-channel planes. This then makes it possible to run one RGBA shader
      that produces 4 different filtering effects in each of the 4 color/alpha channels, and to then unpack the results
      into a 4-up greyscale RawImage buffer for sending over USB. DemoCPUGPU uses this, uploading a greyscale input
      image and then running 4 different filters to create the R, G, B, and A outputs, finally unpacking those into 4
      greyscale images sent over USB.

    Shader conventions:

    - input texture is in "tex"
    - current coordinates within texture are in "tcoord"
    - size of one texture pixel is set in "texelsize" (uniform 2 float)
    - see DemoGPU and DemoCPUGPU for example shaders

    This code is inspired from the tutorial and code examples found on this page:
    http://robotblogging.blogspot.com/2013/10/gpu-accelerated-camera-processing-on.html

    on host, mesa may be in the way of other drivers, eg nvidia. You need to get them out. 
    \code
    sudo find / -name libEGL*
    sudo find / -name libGLESv2*
    
    cd /usr/lib/x86_64-linux-gnu
    ls -l | grep mesa # we will fix those with what we found above
    sudo ln -fs /usr/lib/nvidia-352-updates/libEGL.so libEGL.so
    sudo ln -fs /usr/lib/nvidia-352-updates/libGLESv2.so libGLESv2.so
    sudo ln -fs /usr/lib/nvidia-352-updates/libGL.so libGL.so 
    sudo mv mesa-egl/ld.so.conf mesa-egl/ld.so.conf.disabled
    sudo ldconfig
    
    # back to jevoisbase:
    ldd hbuild/libjevoisbase.so |grep GL # there should not be any mesa in there
    \endcode

    FIXME: yet, this code still does not run on the host. Maybe OpenCV which is using highgui on the host which uses Qt
    which uses OpenGL (not the ES version) is the issue.  \ingroup components */
class FilterGPU : public jevois::Component
{
  public:
    //! Constructor
    FilterGPU(std::string const & instance);

    //! Destructor
    ~FilterGPU();

    //! Set an image processing program
    void setProgram(std::string const & vertex_shader, std::string const & frag_shader);

    //! Set some program parameters, 2 float version
    /*! OpenGL is not very thread-friendly. Hence the program chosen using setProgram() will actually be instantiated
        inside the process() function so everything happens in one thread. Here, users can set program parameters, those
        are just memorized and will be set in process() as well. Each time setProgram() is called, all cached parameters
        are cleared and they should be set again. */
    void setProgramParam2f(std::string name, float val1, float val2);
    
    //! Set some program parameters, 1 float version
    void setProgramParam1f(std::string name, float val);

    //! Set some program parameters, 2 int version
    void setProgramParam2i(std::string name, int val1, int val2);

    //! Set some program parameters, 1 int version
    void setProgramParam1i(std::string name, int val);

    //! Process an image. The dst image should be allocated with correct image size and pixel type
    void process(cv::Mat const & src, cv::Mat & dst);

  private:
    void initDisplay(); // OpenGL is not thread-safe, so we init in the process() thread rather than at construction
    std::string itsVshader, itsFshader;
    volatile bool itsProgramChanged;
    
    std::shared_ptr<GPUtexture> itsSrcTex;
    std::shared_ptr<GPUprogram> itsProgram;
    GLuint itsQuadVertexBuffer;
    EGLDisplay itsDisplay;
    EGLConfig itsConfig;
    EGLContext itsContext;
    EGLSurface itsSurface;
    GLuint itsFramebufferId; // output frame buffer
    GLuint itsRenderbufferId; // output render buffer
    GLuint itsRenderWidth, itsRenderHeight;
    int itsRenderType;
    enum paramtype { F2, I2, F1, I1 };
    struct param { paramtype type; float val[2]; };
        
    std::map<std::string, param> itsProgramParams;
    
    mutable std::mutex itsMutex;
};

