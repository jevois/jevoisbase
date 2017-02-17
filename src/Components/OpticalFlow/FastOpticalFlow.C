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

#include <jevoisbase/src/Components/OpticalFlow/FastOpticalFlow.H>

// This is a simplified version of the main() function of the code here: git clone
// https://github.com/tikroeger/OF_DIS.git

// See jevoisbase/Contrib for full information including copyright and for the other source files.

#include <opencv2/imgproc/imgproc.hpp>
#include <jevoisbase/Contrib/OF_DIS/oflow.h>

// ##############################################################################################################
FastOpticalFlow::FastOpticalFlow(std::string const & instance) :
    jevois::Component::Component(instance), itsProfiler("FastOpticalFlow"), itsNuke(true),
    img_bo_pyr(nullptr), img_bo_dx_pyr(nullptr), img_bo_dy_pyr(nullptr), img_bo_fmat_pyr(nullptr),
    img_bo_dx_fmat_pyr(nullptr), img_bo_dy_fmat_pyr(nullptr), itsPyrDepth(0)
{ }

// ##############################################################################################################
FastOpticalFlow::~FastOpticalFlow()
{
  for (int i = 0; i < itsPyrDepth; ++i)
  {
    img_bo_fmat_pyr[i].release();
    img_bo_dx_fmat_pyr[i].release();
    img_bo_dy_fmat_pyr[i].release();
  }

  img_bo_mat.release();
  img_bo_fmat.release();

  if (itsPyrDepth)
  {
    delete [] img_bo_pyr;
    delete [] img_bo_dx_pyr;
    delete [] img_bo_dy_pyr;
    delete [] img_bo_fmat_pyr;
    delete [] img_bo_dx_fmat_pyr;
    delete [] img_bo_dy_fmat_pyr;
  }
}

// ##############################################################################################################
void FastOpticalFlow::onParamChange(fastopticalflow::opoint const & JEVOIS_UNUSED_PARAM(param), int const & val)
{
  std::lock_guard<std::mutex> _(itsMtx); // prevent changes during critical section of process()
  itsNuke = true;
}

// ##############################################################################################################
void FastOpticalFlow::onParamChange(fastopticalflow::thetasf const & JEVOIS_UNUSED_PARAM(param), int const & val)
{
  std::lock_guard<std::mutex> _(itsMtx); // prevent changes during critical section of process()
  itsNuke = true;
}

// ##############################################################################################################
void FastOpticalFlow::onParamChange(fastopticalflow::thetaps const & JEVOIS_UNUSED_PARAM(param), int const & val)
{
  std::lock_guard<std::mutex> _(itsMtx); // prevent changes during critical section of process()
  itsNuke = true;
}

// ##############################################################################################################
void FastOpticalFlow::onParamChange(fastopticalflow::thetaov const & JEVOIS_UNUSED_PARAM(param), float const & val)
{
  std::lock_guard<std::mutex> _(itsMtx); // prevent changes during critical section of process()
  itsNuke = true;
}

// ##############################################################################################################
namespace
{
  void ConstructImgPyramide(const cv::Mat & img_ao_fmat, cv::Mat * img_ao_fmat_pyr, cv::Mat * img_ao_dx_fmat_pyr,
                            cv::Mat * img_ao_dy_fmat_pyr, const float ** img_ao_pyr, const float ** img_ao_dx_pyr,
                            const float ** img_ao_dy_pyr, const int lv_f, const int lv_l, const int rpyrtype,
                            const bool getgrad, const int imgpadding, const int padw, const int padh)
  {
    for (int i=0; i<=lv_f; ++i)  // Construct image and gradient pyramides
    {
      if (i==0) // At finest scale: copy directly, for all other: downscale previous scale by .5
      {
#if (SELECTCHANNEL==1 | SELECTCHANNEL==3)  // use RGB or intensity image directly
        img_ao_fmat_pyr[i] = img_ao_fmat.clone();
#elif (SELECTCHANNEL==2)   // use gradient magnitude image as input
        cv::Mat dx,dy,dx2,dy2,dmag;
        cv::Sobel( img_ao_fmat, dx, CV_32F, 1, 0, 1, 1, 0, cv::BORDER_DEFAULT );
        cv::Sobel( img_ao_fmat, dy, CV_32F, 0, 1, 1, 1, 0, cv::BORDER_DEFAULT );
        dx2 = dx.mul(dx);
        dy2 = dy.mul(dy);
        dmag = dx2+dy2;
        cv::sqrt(dmag,dmag);
        img_ao_fmat_pyr[i] = dmag.clone();
#endif
      }
      else
        cv::resize(img_ao_fmat_pyr[i-1], img_ao_fmat_pyr[i], cv::Size(), .5, .5, cv::INTER_LINEAR);
      
      img_ao_fmat_pyr[i].convertTo(img_ao_fmat_pyr[i], rpyrtype);
      
      if ( getgrad ) 
      {
        cv::Sobel( img_ao_fmat_pyr[i], img_ao_dx_fmat_pyr[i], CV_32F, 1, 0, 1, 1, 0, cv::BORDER_DEFAULT );
        cv::Sobel( img_ao_fmat_pyr[i], img_ao_dy_fmat_pyr[i], CV_32F, 0, 1, 1, 1, 0, cv::BORDER_DEFAULT );
        img_ao_dx_fmat_pyr[i].convertTo(img_ao_dx_fmat_pyr[i], CV_32F);
        img_ao_dy_fmat_pyr[i].convertTo(img_ao_dy_fmat_pyr[i], CV_32F);
      }
    }
    
    // pad images
    for (int i=0; i<=lv_f; ++i)  // Construct image and gradient pyramides
    {
      copyMakeBorder(img_ao_fmat_pyr[i],img_ao_fmat_pyr[i],imgpadding,imgpadding,imgpadding,
                     imgpadding,cv::BORDER_REPLICATE);  // Replicate border for image padding
      img_ao_pyr[i] = (float*)img_ao_fmat_pyr[i].data;
      
      if ( getgrad ) 
      {
        copyMakeBorder(img_ao_dx_fmat_pyr[i],img_ao_dx_fmat_pyr[i],imgpadding,imgpadding,imgpadding,imgpadding,
                       cv::BORDER_CONSTANT , 0); // Zero padding for gradients
        copyMakeBorder(img_ao_dy_fmat_pyr[i],img_ao_dy_fmat_pyr[i],imgpadding,imgpadding,imgpadding,imgpadding,
                       cv::BORDER_CONSTANT , 0);
        
        img_ao_dx_pyr[i] = (float*)img_ao_dx_fmat_pyr[i].data;
        img_ao_dy_pyr[i] = (float*)img_ao_dy_fmat_pyr[i].data;      
      }
    }
  }
  
  int AutoFirstScaleSelect(int imgwidth, int fratio, int patchsize)
  {
    return std::max(0,(int)std::floor(log2((2.0f*(float)imgwidth) / ((float)fratio * (float)patchsize))));
  }
  
}

// ##############################################################################################################
void FastOpticalFlow::process(cv::Mat const & img_ao_mat, cv::Mat & dst)
{
  itsProfiler.start();
  
  // Prevent param changes while we are running:
  std::unique_lock<std::mutex> lck(itsMtx);

  if (img_ao_mat.type() != CV_8UC1) LFATAL("Input images must have same size and be CV_8UC1 grayscale");
  if (dst.cols != img_ao_mat.cols || dst.rows != img_ao_mat.rows * 2 || dst.type() != CV_8UC1)
    LFATAL("dst image must have same width as inputs, be 2x taller, and have CV_8UC1 pixels");
  
  // Nuke all caches if input size changed:
  if (img_ao_mat.cols != itsWidth || img_ao_mat.rows != itsHeight) itsNuke = true;
  itsWidth = img_ao_mat.cols; itsHeight = img_ao_mat.rows;
  
  int rpyrtype, nochannels;
#if (SELECTCHANNEL==1 | SELECTCHANNEL==2) // use Intensity or Gradient image      
  rpyrtype = CV_32FC1;
  nochannels = 1;
#elif (SELECTCHANNEL==3) // use RGB image
  rpyrtype = CV_32FC3;
  nochannels = 3;      
#endif
  cv::Mat img_ao_fmat;
  cv::Size sz = img_ao_mat.size();
  int width_org = sz.width;   // unpadded original image size
  int height_org = sz.height;  // unpadded original image size 
  
  // *** Parse rest of parameters, See oflow.h for definitions.
  int lv_f, lv_l, maxiter, miniter, patchsz, patnorm, costfct, tv_innerit, tv_solverit, verbosity;
  float mindprate, mindrrate, minimgerr, poverl, tv_alpha, tv_gamma, tv_delta, tv_sor;
  bool usefbcon, usetvref;
  
  mindprate = 0.05; mindrrate = 0.95; minimgerr = 0.0;    
  usefbcon = 0; patnorm = 1; costfct = 0; 
  tv_alpha = 10.0; tv_gamma = 10.0; tv_delta = 5.0;
  tv_innerit = 1; tv_solverit = 3; tv_sor = 1.6;
  verbosity = 0; // Default: 2 = Plot detailed timings
  
  int fratio = 5; // For automatic selection of coarsest scale: 1/fratio * width = maximum expected motion magnitude in
                  // image. Set lower to restrict search space.
  
  switch (opoint::get())
  {
  case 1:
    patchsz = 8; poverl = 0.3; 
    lv_f = AutoFirstScaleSelect(width_org, fratio, patchsz);
    lv_l = std::max(lv_f-2,0); maxiter = 16; miniter = 16; 
    usetvref = 0; 
    break;
  case 3:
    patchsz = 12; poverl = 0.75; 
    lv_f = AutoFirstScaleSelect(width_org, fratio, patchsz);
    lv_l = std::max(lv_f-4,0); maxiter = 16; miniter = 16; 
    usetvref = 1; 
    break;
  case 4:
    patchsz = 12; poverl = 0.75; 
    lv_f = AutoFirstScaleSelect(width_org, fratio, patchsz);
    lv_l = std::max(lv_f-5,0); maxiter = 128; miniter = 128; 
    usetvref = 1; 
    break;        
  case 2:
  default:
    patchsz = 8; poverl = 0.4; 
    lv_f = AutoFirstScaleSelect(width_org, fratio, patchsz);
    lv_l = std::max(lv_f-2,0); maxiter = 12; miniter = 12; 
    usetvref = 1; 
    break;
  }

  // Nuke any old pyramid data and allocate some new one?
  if (itsNuke)
  {
    for (int i = 0; i < itsPyrDepth; ++i)
    {
      img_bo_fmat_pyr[i].release();
      img_bo_dx_fmat_pyr[i].release();
      img_bo_dy_fmat_pyr[i].release();
    }
    img_bo_mat.release();
    img_bo_fmat.release();

    if (itsPyrDepth)
    {
      delete [] img_bo_pyr;
      delete [] img_bo_dx_pyr;
      delete [] img_bo_dy_pyr;
      delete [] img_bo_fmat_pyr;
      delete [] img_bo_dx_fmat_pyr;
      delete [] img_bo_dy_fmat_pyr;
    }
    
    img_bo_pyr = new const float*[lv_f + 1];
    img_bo_dx_pyr = new const float*[lv_f + 1];
    img_bo_dy_pyr = new const float*[lv_f + 1];
    img_bo_fmat_pyr = new cv::Mat[lv_f + 1];
    img_bo_dx_fmat_pyr = new cv::Mat[lv_f + 1];
    img_bo_dy_fmat_pyr = new cv::Mat[lv_f + 1];
  }
  
  // Remember pyramid depth so we can free them later:
  itsPyrDepth = lv_f + 1;
  
  // Possibly override some of the default values obtained by setting an operating point:
  int par = thetasf::get(); if (par != -1) { lv_f = par; lv_l = std::max(0, par - 2); }
  par = thetait::get(); if (par != -1) { maxiter = par; miniter = par; }
  par = thetaps::get(); if (par != -1) patchsz = par;
  float fpar = thetaov::get(); if (fpar != -1.0F) poverl = fpar;
  usetvref = usevref::get() ? 1 : 0;

  // keep track of whether we should nuke the caches or not before we unlock:
  bool nuke = itsNuke; itsNuke = false;
  lck.unlock();
  
  // *** Pad image such that width and height are restless divisible on all scales (except last)
  int padw=0, padh=0;
  int scfct = pow(2,lv_f); // enforce restless division by this number on coarsest scale
  int div = sz.width % scfct;
  if (div>0) padw = scfct - div;
  div = sz.height % scfct;
  if (div>0) padh = scfct - div;          
  if (padh>0 || padw>0)
    copyMakeBorder(img_ao_mat,img_ao_mat,floor((float)padh/2.0f),ceil((float)padh/2.0f),
                   floor((float)padw/2.0f),ceil((float)padw/2.0f),cv::BORDER_REPLICATE);
  sz = img_ao_mat.size();  // padded image size, ensures divisibility by 2 on all scales (except last)
  
  //  *** Generate scale pyramides
  img_ao_mat.convertTo(img_ao_fmat, CV_32F); // convert to float

  itsProfiler.checkpoint("Converted");

  const float* img_ao_pyr[lv_f+1];
  const float* img_ao_dx_pyr[lv_f+1];
  const float* img_ao_dy_pyr[lv_f+1];
  
  cv::Mat img_ao_fmat_pyr[lv_f+1];
  cv::Mat img_ao_dx_fmat_pyr[lv_f+1];
  cv::Mat img_ao_dy_fmat_pyr[lv_f+1];

  ConstructImgPyramide(img_ao_fmat, img_ao_fmat_pyr, img_ao_dx_fmat_pyr, img_ao_dy_fmat_pyr, img_ao_pyr,
                       img_ao_dx_pyr, img_ao_dy_pyr, lv_f, lv_l, rpyrtype, 1, patchsz, padw, padh);

  // Copy ao to bo if first frame after a nuke:
  if (nuke)
  {
    img_bo_mat = img_ao_mat;
    img_bo_fmat = img_ao_fmat;

    for (int i = 0; i < itsPyrDepth; ++i)
    {
      img_bo_fmat_pyr[i] = img_ao_fmat_pyr[i]; img_bo_pyr[i] = (float*)img_bo_fmat_pyr[i].data;
      img_bo_dx_fmat_pyr[i] = img_ao_dx_fmat_pyr[i]; img_bo_dx_pyr[i] = (float*)img_bo_dx_fmat_pyr[i].data;
      img_bo_dy_fmat_pyr[i] = img_ao_dy_fmat_pyr[i]; img_bo_dy_pyr[i] = (float*)img_bo_dy_fmat_pyr[i].data;
    }
  }

  itsProfiler.checkpoint("Pyramid");
  
  //  *** Run main optical flow / depth algorithm
  float sc_fct = pow(2,lv_l);
#if (SELECTMODE==1)
  cv::Mat flowout(sz.height / sc_fct , sz.width / sc_fct, CV_32FC2); // Optical Flow
#else
  cv::Mat flowout(sz.height / sc_fct , sz.width / sc_fct, CV_32FC1); // Depth
#endif       
  
  OFC::OFClass ofc(img_ao_pyr, img_ao_dx_pyr, img_ao_dy_pyr, 
                   img_bo_pyr, img_bo_dx_pyr, img_bo_dy_pyr, 
                   patchsz,  // extra image padding to avoid border violation check
                   (float*)flowout.data,   // pointer to n-band output float array
                   nullptr,  // pointer to n-band input float array of size of first (coarsest) scale, or nullptr
                   sz.width, sz.height, 
                   lv_f, lv_l, maxiter, miniter, mindprate, mindrrate, minimgerr, patchsz, poverl, 
                   usefbcon, costfct, nochannels, patnorm, 
                   usetvref, tv_alpha, tv_gamma, tv_delta, tv_innerit, tv_solverit, tv_sor,
                   verbosity);    
  itsProfiler.checkpoint("Flow");
  
  // *** Resize to original scale, if not run to finest level
  if (lv_l != 0)
  {
    flowout *= sc_fct;
    cv::resize(flowout, flowout, cv::Size(), sc_fct, sc_fct , cv::INTER_LINEAR);
  }
  
  // If image was padded, remove padding before returning:
  flowout = flowout(cv::Rect((int)floor((float)padw/2.0f),(int)floor((float)padh/2.0f),width_org,height_org));

  itsProfiler.checkpoint("Resized");
  
  // flowout is CV_32FC2, we want 2-up CV_8UC1 for our final output:
  size_t const npix = width_org * height_org;
  unsigned char * vxptr = dst.data;
  unsigned char * vyptr = dst.data + npix;
  float const * fdata = reinterpret_cast<float const *>(flowout.data);
  float const fac = factor::get();
  
  for (int i = 0; i < npix; ++i)
  {
    *vxptr++ = (unsigned char)(128.0F + std::max(-128.0F, std::min(127.0F, fdata[0] * fac)));
    *vyptr++ = (unsigned char)(128.0F + std::max(-128.0F, std::min(127.0F, fdata[1] * fac)));
    fdata += 2;
  }

  itsProfiler.checkpoint("Output formatted");

  // Get ready for next frame:
  img_bo_mat = img_ao_mat;
  img_bo_fmat = img_ao_fmat;

  for (int i = 0; i < itsPyrDepth; ++i)
  {
    img_bo_fmat_pyr[i] = img_ao_fmat_pyr[i]; img_bo_pyr[i] = (float*)img_bo_fmat_pyr[i].data;
    img_bo_dx_fmat_pyr[i] = img_ao_dx_fmat_pyr[i]; img_bo_dx_pyr[i] = (float*)img_bo_dx_fmat_pyr[i].data;
    img_bo_dy_fmat_pyr[i] = img_ao_dy_fmat_pyr[i]; img_bo_dy_pyr[i] = (float*)img_bo_dy_fmat_pyr[i].data;
  }    

  itsProfiler.stop();
}

