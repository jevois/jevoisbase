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

#include <jevoisbase/src/Components/Saliency/Saliency.H>

#include <jevoisbase/src/Components/Saliency/env_config.h>
#include <jevoisbase/src/Components/Saliency/env_c_math_ops.h>
#include <jevoisbase/src/Components/Saliency/env_image.h>
#include <jevoisbase/src/Components/Saliency/env_image_ops.h>
#include <jevoisbase/src/Components/Saliency/env_log.h>
#include <jevoisbase/src/Components/Saliency/env_params.h>

#include <jevois/Core/VideoBuf.H>
#include <jevois/Debug/Log.H>
#include <jevois/Debug/Timer.H>
#include <jevois/Image/RawImageOps.H>
#include <jevois/Image/ColorConversion.h>

#include <cstdlib>
#include <future>
#include <functional> // for placeholders

#define WEIGHT_SCALEBITS ((env_size_t) 8)

// ##############################################################################################################
static int computeGist(const char * tagName, env_size_t clev, env_size_t slev, struct env_image* submap,
                       const struct env_image * JEVOIS_UNUSED_PARAM(center),
                       const struct env_image * JEVOIS_UNUSED_PARAM(surround), void * vdata)
{
  //LINFO(tagName << ": cs=" << clev << '-' << slev << " submap " << submap->dims.w << 'x' << submap->dims.h);

  // We should be called 72 times, once for each of our 72 feature maps. Our gist contains 16 values (4x4 grid) for each
  // of the maps, so 72*16=1152 values in total. Each single channel has 6 maps. First, parse the tagname to get the
  // offset in our gist vector at the single channel level:

  Saliency::visitor_data * vd = reinterpret_cast<Saliency::visitor_data *>(vdata);
  
  env_size_t const onechan = 6 * 16;
  env_size_t offset; unsigned int bitshift = 0;
  switch (tagName[0])
  {
  case 'r':
    if (tagName[2] == 'd') { offset = 0; bitshift = 6; } // red/green
    else { offset = (std::atoi(tagName + 10) + 6) * onechan; bitshift = 4; } // reichardt(%d/%d) with number 1-based
    break;
  case 'b': offset = onechan; bitshift = 6; break; // blue/yellow
  case 'i': offset = 2 * onechan; bitshift = 7; break; // intensity
  case 's': offset = (std::atoi(tagName + 10) + 2) * onechan; bitshift = 0; break; // steerable(%d/%d),  1-based
  case 'f': offset = 7 * onechan; bitshift = 5; break; // flicker
  default: LFATAL("Unknown channel " << tagName);
  }

  // Now we need to compute an additional offset between 0 and 5 depending on clev and slev:
  // clev in { 2, 3, 4 }, delta = slev - clev in { 3, 4 }
  // FIXME: use envp to get the levels, etc
  env_size_t const delta = slev - clev;
  offset += ((delta - vd->envp->cs_del_min) * 3 + clev - vd->envp->cs_lev_min) * 16;
  if (offset + 16 > vd->gist_size) LFATAL("gist offset " << offset << " out of range");

  // All right, fill in the data:
  env_grid_average(submap, vd->gist + offset, bitshift, 4, 4);
  
  return 0;
}

// ##############################################################################################################
Saliency::Saliency(std::string const & instance) :
    jevois::Component(instance), gist_size(72 * 16), itsProfiler("Saliency", 100, LOG_DEBUG), itsInputDone(true)
{
  env_params_set_defaults(&envp);

  env_init_integer_math(&imath, &envp);
  
  env_img_init_empty(&prev_input);
  env_pyr_init(&prev_lowpass5, 0);
  env_motion_channel_init(&motion_chan, &envp);

  salmap = env_img_initializer;
  intens = env_img_initializer;
  color = env_img_initializer;
  ori = env_img_initializer;
  flicker = env_img_initializer;
  motion = env_img_initializer;
  gist = new unsigned char[gist_size];

  itsVisitorData.gist = gist;
  itsVisitorData.gist_size = gist_size;
  itsVisitorData.envp = &envp;
}

// ##############################################################################################################
Saliency::~Saliency()
{
  delete [] gist;
  env_img_make_empty(&prev_input);
  env_pyr_make_empty(&prev_lowpass5);
  env_motion_channel_destroy(&motion_chan);
}

// ##############################################################################################################
void Saliency::combine_output(struct env_image* chanOut, const intg32 iweight, struct env_image* result)
{
  if (!env_img_initialized(chanOut)) return;
  
  intg32* const sptr = env_img_pixelsw(chanOut);
  const env_size_t sz = env_img_size(chanOut);

  // Lock here so that we combine one channel at a time:
  std::lock_guard<std::mutex> _(itsMtx);
  
  if (!env_img_initialized(result))
  {
    env_img_resize_dims(result, chanOut->dims);
    intg32* const dptr = env_img_pixelsw(result);
    for (env_size_t i = 0; i < sz; ++i)
    {
      sptr[i] = (sptr[i] >> WEIGHT_SCALEBITS) * iweight;
      dptr[i] = sptr[i];
    }
  }
  else
  {
    ENV_ASSERT(env_dims_equal(chanOut->dims, result->dims));
    intg32* const dptr = env_img_pixelsw(result);
    const env_size_t sz = env_img_size(result);
    for (env_size_t i = 0; i < sz; ++i)
    {
      sptr[i] = (sptr[i] >> WEIGHT_SCALEBITS) * iweight;
      dptr[i] += sptr[i];
    }
  }
}

#define SALUPDATE(envval, param) \
  prev = envp.envval; envp.envval = saliency::param::get(); if (envp.envval != prev) nuke = true;

// ##############################################################################################################
void Saliency::processStart(struct env_dims const & dims, bool do_gist)
{
  // Mark our input image as being processed:
  {
    std::unique_lock<std::mutex> ulck(itsRawImageMtx);
    itsInputDone = false;
  }
  
  bool nuke = false; size_t prev;
  
  // Update envp with our current parameters:
  envp.chan_c_weight = saliency::cweight::get();
  envp.chan_i_weight = saliency::iweight::get();
  envp.chan_o_weight = saliency::oweight::get();
  envp.chan_f_weight = saliency::fweight::get();
  envp.chan_m_weight = saliency::mweight::get();

  SALUPDATE(cs_lev_min, centermin);
  envp.cs_lev_max = envp.cs_lev_min + 2;

  SALUPDATE(cs_del_min, deltamin);
  envp.cs_del_max = envp.cs_del_min + 1;

  SALUPDATE(output_map_level, smscale);

  envp.motion_thresh = saliency::mthresh::get();

  envp.flicker_thresh = saliency::fthresh::get();

  prev = envp.multiscale_flicker;
  envp.multiscale_flicker = saliency::msflick::get() ? 1 : 0;
  if (envp.multiscale_flicker != prev) nuke = true;
  
  env_params_validate(&envp);

  // Zero-out all our internals:
  env_img_make_empty(&salmap);
  env_img_make_empty(&intens);
  env_img_make_empty(&color);
  env_img_make_empty(&ori);
  env_img_make_empty(&flicker);
  env_img_make_empty(&motion);
  memset(gist, 0, gist_size);

  // Reject bad images:
  if (dims.w < 32 || dims.h < 32) LFATAL("input dims " << dims.w << 'x' << dims.h << " too small -- REJECTED");
  if (dims.w > 2048 || dims.h > 2048) LFATAL("input dims " << dims.w << 'x' << dims.h << " too large -- REJECTED");

  // Check whether the input size or critical params just changed, and if so invalidate our previous stored data:
  if (env_img_initialized(&prev_input) && (prev_input.dims.w != dims.w || prev_input.dims.h != dims.h)) nuke = true;

  if (nuke)
  {
    env_img_make_empty(&prev_input);
    env_pyr_make_empty(&prev_lowpass5);
    env_motion_channel_destroy(&motion_chan);
    env_motion_channel_init(&motion_chan, &envp);
  }
  
  // Install hook for gist computation, if desired:
  if (do_gist) { envp.user_data_preproc = &itsVisitorData; envp.submapPreProc = &computeGist; }
  else { envp.user_data_preproc = nullptr; envp.submapPreProc = nullptr; }
}

// ##############################################################################################################
void Saliency::waitUntilDoneWithInput() const
{
  std::unique_lock<std::mutex> ulck(itsRawImageMtx);
  if (itsInputDone) return; // we are done already
  itsRawImageCond.wait(ulck, [&]() { return itsInputDone; } );
}

// ##############################################################################################################
void Saliency::process(cv::Mat const & input, bool do_gist)
{
  static env_chan_status_func * statfunc = nullptr;
  static void * statdata = nullptr;

  // We here do what env_mt_visual_cortex_inut used to do in the original envision code, but using lambdas instead of
  // the c-based jobs:
  struct env_dims dims = { (env_size_t)input.cols, (env_size_t)input.rows };
  processStart(dims, do_gist);
  struct env_rgb_pixel * inpixels = reinterpret_cast<struct env_rgb_pixel *>(input.data);

  const intg32 total_weight = env_total_weight(&envp);
  ENV_ASSERT(total_weight > 0);
  
  /* We want to compute
     
   *                 weight
   *        img * ------------
   *              total_weight
   *
   *
   *        To do that without overflowing, we compute it as
   *
   *
   *                 weight      256
   *        img * ------------ * ---
   *              total_weight   256
   *
   *            img       weight * 256
   *        = ( --- ) * ( ------------ )
   *            256       total_weight
   *
   * where 256 is an example of (1<<WEIGHT_SCALEBITS) for
   * WEIGHT_SCALEBITS=8.
   */
  
  // We can get the color channel started right away:
  std::future<void> colorfut;
  if (envp.chan_c_weight > 0)
    colorfut = std::async(std::launch::async, [&](){
        env_chan_color("color", &envp, &imath, inpixels, dims, statfunc, statdata, &color);
        combine_output(&color, envp.chan_c_weight * (1<<WEIGHT_SCALEBITS) / total_weight, &salmap);
      });

  // Compute luminance image:
  struct env_image bwimg; env_img_init(&bwimg, dims);
  env_c_luminance_from_byte(inpixels, dims.w * dims.h, imath.nbits, env_img_pixelsw(&bwimg));
  
  // Notify anyone that was waiting to free the raw input that we are done with it:
  itsInputDone = true; itsRawImageCond.notify_all();

  // Compute a luminance pyramid:
  struct env_pyr lowpass5; env_pyr_init(&lowpass5, env_max_pyr_depth(&envp));
  env_pyr_build_lowpass_5(&bwimg, envp.cs_lev_min, &imath, &lowpass5);
  
  // Now parallelize the other channels:
  std::future<void> motfut;
  if (envp.chan_m_weight > 0)
    motfut = std::async(std::launch::async, [&](){
        env_mt_motion_channel_input(&motion_chan, "motion", bwimg.dims, &lowpass5, statfunc, statdata, &motion);
        combine_output(&motion, envp.chan_m_weight * (1<<WEIGHT_SCALEBITS) / total_weight, &salmap);
      });

  std::future<void> orifut;
  if (envp.chan_o_weight > 0)
    orifut = std::async(std::launch::async, [&](){
        env_mt_chan_orientation("orientation", &bwimg, statfunc, statdata, &ori);
        combine_output(&ori, envp.chan_o_weight * (1<<WEIGHT_SCALEBITS) / total_weight, &salmap);
      });
  
  std::future<void> flickfut;
  if (envp.chan_f_weight > 0)
    flickfut = std::async(std::launch::async, [&](){
        if (envp.multiscale_flicker)
          env_chan_msflicker("flicker", &envp, &imath, bwimg.dims, &prev_lowpass5, &lowpass5,
                             statfunc, statdata, &flicker);
        else
          env_chan_flicker("flicker", &envp, &imath, &prev_input, &bwimg, statfunc, statdata, &flicker);
        
        combine_output(&flicker, envp.chan_f_weight * (1<<WEIGHT_SCALEBITS) / total_weight, &salmap);
        
        if (envp.multiscale_flicker) env_pyr_copy_src_dst(&lowpass5, &prev_lowpass5);
        else env_pyr_make_empty(&prev_lowpass5);
      });
  
  // Intensity is the fastest one and we here just run it in the current thread:
  if (envp.chan_i_weight > 0)
  {
    env_chan_intensity("intensity", &envp, &imath, bwimg.dims, &lowpass5, 1, statfunc, statdata, &intens);
    combine_output(&intens, envp.chan_i_weight * (1<<WEIGHT_SCALEBITS) / total_weight, &salmap);
  }

  // Wait for all channels to finish up:
  if (colorfut.valid()) colorfut.get();
  if (orifut.valid()) orifut.get();
  if (flickfut.valid()) flickfut.get();
  if (motfut.valid()) motfut.get();

  // Cleanup and get ready for next frame:
  if (!envp.multiscale_flicker) env_img_swap(&prev_input, &bwimg); else env_img_make_empty(&prev_input);

  if (statfunc) (*statfunc)(statdata, "saliency", &salmap);

  // We transfer our lowpass5 to the motion channel as unshifted prev:
  env_pyr_swap(&lowpass5, &motion_chan.unshifted_prev);
  env_pyr_make_empty(&lowpass5);

  env_img_make_empty(&bwimg);
  /*
  env_visual_cortex_rescale_ranges(&salmap, &intens, &color, &ori, &flicker, &motion);
  */
}

// ##############################################################################################################
void Saliency::process(jevois::RawImage const & input, bool do_gist)
{
  itsProfiler.start();

  static env_chan_status_func * statfunc = nullptr;
  static void * statdata = nullptr;

  // We here do what env_mt_visual_cortex_inut used to do in the original envision code, but using lambdas instead of
  // the c-based jobs:
  struct env_dims dims = { input.width, input.height };
  processStart(dims, do_gist);
  itsProfiler.checkpoint("processStart");
  
  // Compute Lum, RG, BY, parallelizing over rows:
  const intg32 lumthresh = (3*255) / 10;
  struct env_image rgimg; env_img_init(&rgimg, dims);
  struct env_image byimg; env_img_init(&byimg, dims);
  struct env_image bwimg; env_img_init(&bwimg, dims);

  int const nthreads = 4;
  int hh = dims.h / nthreads;
  std::vector<std::future<void> > rgbyfut;
  unsigned char const * inpix = input.pixels<unsigned char>();
  intg32 * rgpix = env_img_pixelsw(&rgimg);
  intg32 * bypix = env_img_pixelsw(&byimg);
  intg32 * bwpix = env_img_pixelsw(&bwimg);
  for (int i = 0; i < nthreads-1; ++i)
    rgbyfut.push_back(std::async(std::launch::async, [&](int ii) {    
          int offset = dims.w * hh * ii;
          convertYUYVtoRGBYL(dims.w, hh, inpix + offset*2, rgpix + offset, bypix + offset, bwpix + offset,
                             lumthresh, imath.nbits);
        }, i));

  // Do the last bit in the current thread:
  int offset = dims.w * hh * (nthreads - 1);
  convertYUYVtoRGBYL(dims.w, dims.h - hh * (nthreads-1), inpix + offset*2, rgpix + offset, bypix + offset,
                     bwpix + offset, lumthresh, imath.nbits);

  const intg32 total_weight = env_total_weight(&envp);
  ENV_ASSERT(total_weight > 0);
  
  // We can get the color channels started right away. Here we split rg and by into two threads then combine later in a
  // manner similar to what env_chan_color_rgby() does:
  const env_size_t firstlevel = envp.cs_lev_min;
  const env_size_t depth = env_max_pyr_depth(&envp);
  std::future<void> rgfut, byfut;
  struct env_image byOut = env_img_initializer;

  // Wait for rgbylum computation to be complete:
  for (auto & f : rgbyfut) f.get();
  rgbyfut.clear();
  itsProfiler.checkpoint("rgby");

  // Notify anyone that was waiting to free the raw input that we are done with it:
  itsInputDone = true; itsRawImageCond.notify_all();
  
  // Launch RG and BY in threads:
  if (envp.chan_c_weight > 0)
  {
    rgfut = std::async(std::launch::async, [&]() {
          struct env_pyr rgpyr;
          env_pyr_init(&rgpyr, depth);
          env_pyr_build_lowpass_5(&rgimg, firstlevel, &imath, &rgpyr);
          env_chan_intensity("red/green", &envp, &imath, rgimg.dims, &rgpyr, 0, statfunc, statdata, &color);
          env_pyr_make_empty(&rgpyr);
      });

    byfut = std::async(std::launch::async, [&]() {
          struct env_pyr bypyr;
          env_pyr_init(&bypyr, depth);
          env_pyr_build_lowpass_5(&byimg, firstlevel, &imath, &bypyr);
          env_chan_intensity("blue/yellow", &envp, &imath, byimg.dims, &bypyr, 0, statfunc, statdata, &byOut);
          env_pyr_make_empty(&bypyr);
      });
  }
  
  // Compute a luminance pyramid:
  struct env_pyr lowpass5; env_pyr_init(&lowpass5, env_max_pyr_depth(&envp));
  env_pyr_build_lowpass_5(&bwimg, envp.cs_lev_min, &imath, &lowpass5);

  itsProfiler.checkpoint("lowpass pyr");
  
  // Now parallelize the other channels:
  std::future<void> motfut;
  if (envp.chan_m_weight > 0)
    motfut = std::async(std::launch::async, [&]() {
        env_mt_motion_channel_input(&motion_chan, "motion", bwimg.dims, &lowpass5, statfunc, statdata, &motion);
        combine_output(&motion, envp.chan_m_weight * (1<<WEIGHT_SCALEBITS) / total_weight, &salmap);
      });

  std::future<void> orifut;
  if (envp.chan_o_weight > 0)
    orifut = std::async(std::launch::async, [&]() {
        env_mt_chan_orientation("orientation", &bwimg, statfunc, statdata, &ori);
        combine_output(&ori, envp.chan_o_weight * (1<<WEIGHT_SCALEBITS) / total_weight, &salmap);
      });
  
  std::future<void> flickfut;
  if (envp.chan_f_weight > 0)
    flickfut = std::async(std::launch::async, [&]() {
        if (envp.multiscale_flicker)
          env_chan_msflicker("flicker", &envp, &imath, bwimg.dims, &prev_lowpass5, &lowpass5,
                             statfunc, statdata, &flicker);
        else
          env_chan_flicker("flicker", &envp, &imath, &prev_input, &bwimg, statfunc, statdata, &flicker);
        
        combine_output(&flicker, envp.chan_f_weight * (1<<WEIGHT_SCALEBITS) / total_weight, &salmap);
        
        if (envp.multiscale_flicker) env_pyr_copy_src_dst(&lowpass5, &prev_lowpass5);
        else env_pyr_make_empty(&prev_lowpass5);
      });
  
  // Intensity is the fastest one and we here just run it in the current thread:
  if (envp.chan_i_weight > 0)
  {
    env_chan_intensity("intensity", &envp, &imath, bwimg.dims, &lowpass5, 1, statfunc, statdata, &intens);
    combine_output(&intens, envp.chan_i_weight * (1<<WEIGHT_SCALEBITS) / total_weight, &salmap);
  }
  itsProfiler.checkpoint("intens");
  
  // Wait for all channels to finish up:
  if (rgfut.valid()) rgfut.get();
  itsProfiler.checkpoint("red-green");

  if (byfut.valid())
  {
    byfut.get();
    
    // Finish up the color channel by combining rg and by:
    const intg32 * const byptr = env_img_pixels(&byOut);
    intg32 * const dptr = env_img_pixelsw(&color);
    const env_size_t sz = env_img_size(&color);
    for (env_size_t i = 0; i < sz; ++i) dptr[i] = (dptr[i] + byptr[i]) >> 1;

    env_max_normalize_inplace(&color, INTMAXNORMMIN, INTMAXNORMMAX, envp.maxnorm_type, envp.range_thresh);

    if (statfunc) (*statfunc)(statdata, "color", &color);
    env_img_make_empty(&byOut);

    // Add color channel to the saliency map:
    combine_output(&color, envp.chan_c_weight * (1<<WEIGHT_SCALEBITS) / total_weight, &salmap);
  }
  itsProfiler.checkpoint("blue-yellow");

  if (orifut.valid()) orifut.get();
  itsProfiler.checkpoint("orientation");

  if (flickfut.valid()) flickfut.get();
  itsProfiler.checkpoint("flicker");

  if (motfut.valid()) motfut.get();
  itsProfiler.checkpoint("motion");

  // Cleanup and get ready for next frame:
  if (!envp.multiscale_flicker) env_img_swap(&prev_input, &bwimg); else env_img_make_empty(&prev_input);

  if (statfunc) (*statfunc)(statdata, "saliency", &salmap);

  // We transfer our lowpass5 to the motion channel as unshifted prev:
  env_pyr_swap(&lowpass5, &motion_chan.unshifted_prev);
  env_pyr_make_empty(&lowpass5);

  env_img_make_empty(&bwimg);
  env_img_make_empty(&rgimg);
  env_img_make_empty(&byimg);
  /*
  env_visual_cortex_rescale_ranges(&salmap, &intens, &color, &ori, &flicker, &motion);
  */
  itsProfiler.stop();
}

// ##############################################################################################################
void Saliency::env_mt_chan_orientation(const char* tagName, const struct env_image* img,
                                               env_chan_status_func* status_func, void* status_userdata,
                                               struct env_image* result)
{
  env_img_make_empty(result);
  
  if (envp.num_orientations == 0) return;
  
  struct env_pyr hipass9;
  env_pyr_init(&hipass9, env_max_pyr_depth(&envp));
  env_pyr_build_hipass_9(img, envp.cs_lev_min, &imath, &hipass9);
  
  char buf[17] = {
    's', 't', 'e', 'e', 'r', 'a', 'b', 'l', 'e', // 0--8
    '(', '_', '_', // 9--11
    '/', '_', '_', ')', '\0' // 12--16
  };
  
  ENV_ASSERT(envp.num_orientations <= 99);
  
  buf[13] = '0' + (envp.num_orientations / 10);
  buf[14] = '0' + (envp.num_orientations % 10);

  std::vector<std::future<void> > fut;
  std::mutex mtx;
  for (env_size_t i = 0; i < envp.num_orientations; ++i)
    fut.push_back(std::async(std::launch::async, [&](env_size_t ii) {
          struct env_image chanOut; env_img_init_empty(&chanOut);

          char tagname[17]; memcpy(tagname, buf, 17);
          tagname[10] = '0' + ((ii+1) / 10);
          tagname[11] = '0' + ((ii+1) % 10);
         
          // theta = (180.0 * i) / envp.num_orientations + 90.0, where ENV_TRIG_TABSIZ is equivalent to 360.0 or 2*pi
          const env_size_t thetaidx = (ENV_TRIG_TABSIZ * ii) / (2 * envp.num_orientations) + (ENV_TRIG_TABSIZ / 4);
          ENV_ASSERT(thetaidx < ENV_TRIG_TABSIZ);
    
          env_chan_steerable(tagname, &envp, &imath, img->dims, &hipass9, thetaidx,
                             status_func, status_userdata, &chanOut);

          // Access result image one thread at a time:
          std::lock_guard<std::mutex> _(mtx);
          if (!env_img_initialized(result))
          {
            env_img_resize_dims(result, chanOut.dims);
            env_c_image_div_scalar(env_img_pixels(&chanOut), env_img_size(&chanOut), (intg32)envp.num_orientations,
                                   env_img_pixelsw(result));
          }
          else
          {
            ENV_ASSERT(env_dims_equal(chanOut.dims, result->dims));
            env_c_image_div_scalar_accum(env_img_pixels(&chanOut), env_img_size(&chanOut),
                                         (intg32)envp.num_orientations, env_img_pixelsw(result));
          }
          env_img_make_empty(&chanOut);
        }, i));

  // Wait for all the jobs to complete:
  for (std::future<void> & f : fut) f.get();
  
  env_pyr_make_empty(&hipass9);
  
  if (env_img_initialized(result))
    env_max_normalize_inplace(result, INTMAXNORMMIN, INTMAXNORMMAX, envp.maxnorm_type, envp.range_thresh);
  
  if (status_func) (*status_func)(status_userdata, tagName, result);
}

// ##############################################################################################################
void Saliency::env_mt_motion_channel_input(struct env_motion_channel* chan, const char* tagName,
                                                   const struct env_dims inputdims, struct env_pyr* unshiftedCur,
                                                   env_chan_status_func* status_func, void* status_userdata,
                                                   struct env_image* result)
{
  env_img_make_empty(result);
  
  if (chan->num_directions != envp.num_motion_directions)
  {
    env_motion_channel_destroy(chan);
    env_motion_channel_init(chan, &envp);
  }

  if (chan->num_directions == 0) return;
  
  char buf[17] =
    {
      'r', 'e', 'i', 'c', 'h', 'a', 'r', 'd', 't', // 0--8
      '(', '_', '_', // 9--11
      '/', '_', '_', ')', '\0' // 12--16
    };
  
  ENV_ASSERT(chan->num_directions <= 99);
  
  buf[13] = '0' + (chan->num_directions / 10);
  buf[14] = '0' + (chan->num_directions % 10);
  
  // compute Reichardt motion detection into several directions
  std::vector<std::future<void> > fut;
  std::mutex mtx;
  for (env_size_t dir = 0; dir < chan->num_directions; ++dir)
    fut.push_back(std::async(std::launch::async, [&](env_size_t d) {
          struct env_image chanOut; env_img_init_empty(&chanOut);

          char tagname[17]; memcpy(tagname, buf, 17);
          tagname[10] = '0' + ((d+1) / 10);
          tagname[11] = '0' + ((d+1) % 10);

          const env_size_t firstlevel = envp.cs_lev_min;
          const env_size_t depth = env_max_pyr_depth(&envp);
  
          // theta = (360.0 * i) / chan->num_directions;
          const env_size_t thetaidx = (d * ENV_TRIG_TABSIZ) / chan->num_directions;
          ENV_ASSERT(thetaidx < ENV_TRIG_TABSIZ);
  
          // create an empty pyramid:
          struct env_pyr shiftedCur; env_pyr_init(&shiftedCur, depth);
  
          // fill the empty pyramid with the shifted version
          for (env_size_t i = firstlevel; i < depth; ++i)
          {
            env_img_resize_dims(env_pyr_imgw(&shiftedCur, i), env_pyr_img(unshiftedCur, i)->dims);
            env_shift_image(env_pyr_img(unshiftedCur, i), imath.costab[thetaidx], -imath.sintab[thetaidx],
                            ENV_TRIG_NBITS, env_pyr_imgw(&shiftedCur, i));
          }
  
          env_chan_direction(tagname, &envp, &imath, inputdims, &chan->unshifted_prev, unshiftedCur,
                             &chan->shifted_prev[d], &shiftedCur, status_func, status_userdata, &chanOut);
  
          env_pyr_swap(&chan->shifted_prev[d], &shiftedCur);
          env_pyr_make_empty(&shiftedCur);

          // Access result image one thread at a time:
          std::lock_guard<std::mutex> _(mtx);
          if (env_img_initialized(&chanOut))
          {
            if (!env_img_initialized(result))
            {
              env_img_resize_dims(result, chanOut.dims);
              env_c_image_div_scalar(env_img_pixels(&chanOut), env_img_size(&chanOut), (intg32)chan->num_directions,
                                     env_img_pixelsw(result));
            }
            else
            {
              ENV_ASSERT(env_dims_equal(chanOut.dims, result->dims));
              env_c_image_div_scalar_accum(env_img_pixels(&chanOut), env_img_size(&chanOut),
                                           (intg32)chan->num_directions, env_img_pixelsw(result));
            }
          }
          env_img_make_empty(&chanOut);
        }, dir));

  // Wait for all the jobs to complete:
  for (std::future<void> & f : fut) f.get();
        
  if (env_img_initialized(result))
    env_max_normalize_inplace(result, INTMAXNORMMIN, INTMAXNORMMAX, envp.maxnorm_type, envp.range_thresh);

  if (status_func) (*status_func)(status_userdata, tagName, result);
}

// ##############################################################################################################
void Saliency::getSaliencyMax(int & x, int & y, intg32 & value)
{
  if (env_img_initialized(&salmap) == false) LFATAL("Saliency map has not yet been computed");

  intg32 *sm = salmap.pixels; int const smw = int(salmap.dims.w), smh = int(salmap.dims.h);

  value = *sm;

  for (int j = 0; j < smh; ++j)
    for (int i = 0; i < smw; ++i)
      if (*sm > value) { value = *sm++; x = i; y = j; } else ++sm;
}

// ##############################################################################################################
void Saliency::inhibitionOfReturn(int const x, int const y, float const sigma)
{
  if (env_img_initialized(&salmap) == false) LFATAL("Saliency map has not yet been computed");

  intg32 *sm = salmap.pixels; int const smw = int(salmap.dims.w), smh = int(salmap.dims.h);
  float const sigsq = sigma * sigma;
  
  for (int j = 0; j < smh; ++j)
    for (int i = 0; i < smw; ++i)
    {
      float const distsq = (i-x)*(i-x) + (j-y)*(j-y);
      if (distsq < sigsq)
        *sm++ = 0; // hard kill in a disk up to sigma
      else
      {
        float val = *sm;
        val *= 1.0F - expf( -0.5F * (distsq - sigsq ) / sigsq); // smooth decay
        *sm++ = static_cast<intg32>(val + 0.4999F);
      }
    }
}

// ####################################################################################################
void drawMap(jevois::RawImage & img, env_image const * fmap, unsigned int xoff, unsigned int yoff,
             unsigned int scale)
{
  unsigned int const imgw = img.width;
  unsigned short * d = img.pixelsw<unsigned short>() + xoff + yoff * imgw;
  intg32 *s = fmap->pixels;
  const env_size_t w = fmap->dims.w, h = fmap->dims.h;
  const env_size_t ws = w * scale;

  for (env_size_t jj = 0; jj < h; ++jj)
  {
    unsigned short const * dd = d;

    // Copy and scale the first row one pixel at a time:
    for (env_size_t ii = 0; ii < w; ++ii)
    {
      intg32 v = *s++;
      unsigned short const val = 0x8000 | v;
      for (env_size_t k = 0; k < scale; ++k) *d++ = val;
    }
    d += imgw - ws;

    // Then just use memcpy to duplicate it to achieve the scaling factor vertically:
    for (env_size_t k = 1; k < scale; ++k) { memcpy(d, dd, ws * 2); d += imgw; }
  }

  // Draw a rectangle to delinate the map:
  jevois::rawimage::drawRect(img, xoff, yoff, scale * w, scale * h, 0x80a0);
}

// ####################################################################################################
void drawMap(jevois::RawImage & img, env_image const * fmap, unsigned int xoff, unsigned int yoff,
             unsigned int scale, unsigned int bitshift)
{
  unsigned int const imgw = img.width;
  unsigned short * d = img.pixelsw<unsigned short>() + xoff + yoff * imgw;
  intg32 *s = fmap->pixels;
  const env_size_t w = fmap->dims.w, h = fmap->dims.h;
  const env_size_t ws = w * scale;

  for (env_size_t jj = 0; jj < h; ++jj)
  {
    unsigned short const * dd = d;

    // Copy and scale the first row one pixel at a time:
    for (env_size_t ii = 0; ii < w; ++ii)
    {
      intg32 v = (*s++) >> bitshift; if (v > 255) v = 255;
      unsigned short const val = 0x8000 | v;
      for (env_size_t k = 0; k < scale; ++k) *d++ = val;
    }
    d += imgw - ws;

    // Then just use memcpy to duplicate it to achieve the scaling factor vertically:
    for (env_size_t k = 1; k < scale; ++k) { memcpy(d, dd, ws * 2); d += imgw; }
  }

  // Draw a rectangle to delinate the map:
  jevois::rawimage::drawRect(img, xoff, yoff, scale * w, scale * h, 0x80a0);
}

// ####################################################################################################
void drawGist(jevois::RawImage & img, unsigned char const * gist, size_t gistsize, unsigned int xoff,
              unsigned int yoff, unsigned int width, unsigned int scale)
{
  unsigned int const height = gistsize / width;
  unsigned int const imgw = img.width;
  unsigned short * d = img.pixelsw<unsigned short>() + xoff + yoff * imgw;
  unsigned char const * const dataend = gist + gistsize;
  unsigned int const ws = width * scale;

  for (env_size_t jj = 0; jj < height; ++jj)
  {
    unsigned short const * dd = d;

    // Copy and scale the first row one pixel at a time:
    for (env_size_t ii = 0; ii < width; ++ii)
    {
      intg32 v = gist >= dataend ? 0 : *gist++;
      unsigned short const val = 0x8000 | v;
      for (env_size_t k = 0; k < scale; ++k) *d++ = val;
    }
    d += imgw - ws;

    // Then just use memcpy to duplicate it to achieve the scaling factor vertically:
    for (env_size_t k = 1; k < scale; ++k) { memcpy(d, dd, ws * 2); d += imgw; }
  }

  // Draw a rectangle to delinate the map:
  //jevois::rawimage::drawRect(img, xoff, yoff, scale * width, scale * height, 0x80a0);
}

