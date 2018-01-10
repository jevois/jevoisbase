######################################################################################################################
#
# JeVois Smart Embedded Machine Vision Toolkit - Copyright (C) 2016 by Laurent Itti, the University of Southern
# California (USC), and iLab at USC. See http://iLab.usc.edu and http://jevois.org for information about this project.
#
# This file is part of the JeVois Smart Embedded Machine Vision Toolkit.  This program is free software; you can
# redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software
# Foundation, version 2.  This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY;
# without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public
# License for more details.  You should have received a copy of the GNU General Public License along with this program;
# if not, write to the Free Software Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301, USA.
#
# Contact information: Laurent Itti - 3641 Watt Way, HNB-07A - Los Angeles, BA 90089-2520 - USA.
# Tel: +1 213 740 3527 - itti@pollux.usc.edu - http://iLab.usc.edu - http://jevois.org
######################################################################################################################

# CMake build rules for JeVois Base library and executables

# You may provide the installed JeVois config root and version as:
#   cmake -DJEVOIS_CONFIG=/jevois/config ..

cmake_minimum_required(VERSION 3.1)

# Set vendor name, our modules will be placed in a directory by that name under /jevois/modules:
set(JEVOIS_VENDOR "JeVois")
set(JEVOIS_CONFIG "/jevois/config" CACHE STRING "Path to JeVois config to use")

# Include our helper functions, config, etc from the JeVois install:
set(CMAKE_MODULE_PATH ${JEVOIS_CONFIG})
include(jevois_config)
include(JeVois)

# Set project name, detects compiler (which has been set by our helper module). Then set some complation flags:
project(jevoisbase)
jevois_project_set_flags()

set(JVB "${CMAKE_CURRENT_SOURCE_DIR}")

# Allow #include <jevoibase/x/y> in our components and modules:
include_directories("include")

# Allow #include <jevoibase/Contrib/x/y> in our components and modules:
include_directories("..")

# Setup our library using the source files in src/Components and call it jevoisbase with version 1.6.3:
jevois_setup_library(src/Components jevoisbase 1.6.3)

# Setup our modules that are in src/Modules, make them depend on jevoisbase:
jevois_setup_modules(src/Modules jevoisbase)

# Add includes to our install / distribution package:
add_subdirectory(include)

# Now setup any additional includes and sources that we will add to the jevoisbase library for use by our individual
# modules, typically those are in the Contrib/ drectory:

########################################################################################################################
# OpenGL-ES libraries on platform (and they also exist on host):
target_link_libraries(jevoisbase GLESv2 EGL)

########################################################################################################################
# Add any needed boost libraries that are not already pulled in by libjevois:
target_link_libraries(jevoisbase boost_regex)

########################################################################################################################
# tiny-dnn support:
include_directories(Contrib)
include_directories(Contrib/tiny-dnn)

include_directories(Contrib/pthreadpool/include)

target_sources(jevoisbase PRIVATE Contrib/pthreadpool/src/threadpool-pthreads.c)

########################################################################################################################
# darknet with NNPACK support: (on top of things already pulled in by tiny-dnn and NNPACK)

include_directories(Contrib/NNPACK/include)
include_directories(Contrib/NNPACK/src)
include_directories(Contrib/FP16/include)
include_directories(Contrib/FXdiv/include)
include_directories(Contrib/psimd/include)

# We use NNPACK-darknet fork so we can compile darknet, but it seems behind quite a bit in terms of neon support
set(NNPS "${JVB}/Contrib/NNPACK/src")

# NNPACK: Core
target_sources(jevoisbase PRIVATE ${NNPS}/init.c)
target_sources(jevoisbase PRIVATE ${NNPS}/convolution-output.c)
target_sources(jevoisbase PRIVATE ${NNPS}/convolution-input-gradient.c)
target_sources(jevoisbase PRIVATE ${NNPS}/convolution-kernel-gradient.c)
target_sources(jevoisbase PRIVATE ${NNPS}/convolution-inference.c)
target_sources(jevoisbase PRIVATE ${NNPS}/fully-connected-output.c)
target_sources(jevoisbase PRIVATE ${NNPS}/fully-connected-inference.c)
target_sources(jevoisbase PRIVATE ${NNPS}/pooling-output.c)
target_sources(jevoisbase PRIVATE ${NNPS}/softmax-output.c)
target_sources(jevoisbase PRIVATE ${NNPS}/relu-output.c)
target_sources(jevoisbase PRIVATE ${NNPS}/relu-input-gradient.c)

if (JEVOIS_PLATFORM)
  # Transformations
  target_sources(jevoisbase PRIVATE ${NNPS}/psimd/2d-fourier-8x8.c)
  target_sources(jevoisbase PRIVATE ${NNPS}/psimd/2d-fourier-16x16.c)
  target_sources(jevoisbase PRIVATE ${NNPS}/neon/2d-winograd-8x8-3x3.c)
  target_sources(jevoisbase PRIVATE ${NNPS}/neon/2d-winograd-8x8-3x3-fp16.c)
  # ReLU and Softmax
  target_sources(jevoisbase PRIVATE ${NNPS}/neon/relu.c)
  target_sources(jevoisbase PRIVATE ${NNPS}/psimd/softmax.c)
  # FFT block accumulation
  #target_sources(jevoisbase PRIVATE ${NNPS}/psimd/fft-block-mac.c) ###
  # Tuple GEMM
  target_sources(jevoisbase PRIVATE ${NNPS}/neon/blas/h4gemm.c)
  target_sources(jevoisbase PRIVATE ${NNPS}/neon/blas/s4gemm.c)
  target_sources(jevoisbase PRIVATE ${NNPS}/neon/blas/c4gemm.c)
  target_sources(jevoisbase PRIVATE ${NNPS}/neon/blas/s4c2gemm.c)
  target_sources(jevoisbase PRIVATE ${NNPS}/neon/blas/c4gemm-conjb.c)
  target_sources(jevoisbase PRIVATE ${NNPS}/neon/blas/s4c2gemm-conjb.c)
  target_sources(jevoisbase PRIVATE ${NNPS}/neon/blas/c4gemm-conjb-transc.c)
  target_sources(jevoisbase PRIVATE ${NNPS}/neon/blas/s4c2gemm-conjb-transc.c)
  # Direct convolution
  target_sources(jevoisbase PRIVATE ${NNPS}/neon/blas/conv1x1.c)
  # BLAS microkernels
  target_sources(jevoisbase PRIVATE ${NNPS}/neon/blas/sgemm.c)
  target_sources(jevoisbase PRIVATE ${NNPS}/neon/blas/sdotxf.c)
  target_sources(jevoisbase PRIVATE ${NNPS}/psimd/blas/shdotxf.c)
  # FFT and Winograd
  target_sources(jevoisbase PRIVATE ${NNPS}/psimd/fft-aos.c)
  target_sources(jevoisbase PRIVATE ${NNPS}/psimd/fft-soa.c)
  target_sources(jevoisbase PRIVATE ${NNPS}/psimd/fft-real.c)
  target_sources(jevoisbase PRIVATE ${NNPS}/psimd/fft-dualreal.c)
  target_sources(jevoisbase PRIVATE ${NNPS}/neon/winograd-f6k3.c)
  target_sources(jevoisbase PRIVATE ${NNPS}/psimd/exp.c)

  add_definitions(-DARM_NEON)

else (JEVOIS_PLATFORM)

  # NOTE: on host we just use scalar as opposed to x86 SIMD extensions
  # Transformations
  target_sources(jevoisbase PRIVATE ${NNPS}/scalar/2d-fourier-8x8.c)
  target_sources(jevoisbase PRIVATE ${NNPS}/scalar/2d-fourier-16x16.c)
  target_sources(jevoisbase PRIVATE ${NNPS}/scalar/2d-winograd-8x8-3x3.c)
  # ReLU and Softmax
  target_sources(jevoisbase PRIVATE ${NNPS}/scalar/relu.c)
  target_sources(jevoisbase PRIVATE ${NNPS}/scalar/softmax.c)
  # Tuple GEMM
  target_sources(jevoisbase PRIVATE ${NNPS}/scalar/blas/s2gemm.c)
  target_sources(jevoisbase PRIVATE ${NNPS}/scalar/blas/s2gemm-transc.c)
  target_sources(jevoisbase PRIVATE ${NNPS}/scalar/blas/cgemm.c)
  target_sources(jevoisbase PRIVATE ${NNPS}/scalar/blas/cgemm-conjb.c)
  target_sources(jevoisbase PRIVATE ${NNPS}/scalar/blas/cgemm-conjb-transc.c)
  # Direct convolution
  target_sources(jevoisbase PRIVATE ${NNPS}/scalar/blas/conv1x1.c)
  # BLAS microkernels
  target_sources(jevoisbase PRIVATE ${NNPS}/scalar/blas/sgemm.c)
  target_sources(jevoisbase PRIVATE ${NNPS}/scalar/blas/sdotxf.c)
  target_sources(jevoisbase PRIVATE ${NNPS}/scalar/blas/shdotxf.c)
  # FFT and Winograd
  target_sources(jevoisbase PRIVATE ${NNPS}/scalar/fft-aos.c)
  target_sources(jevoisbase PRIVATE ${NNPS}/scalar/fft-soa.c)
  target_sources(jevoisbase PRIVATE ${NNPS}/scalar/fft-real.c)
  target_sources(jevoisbase PRIVATE ${NNPS}/scalar/fft-dualreal.c)
  target_sources(jevoisbase PRIVATE ${NNPS}/scalar/winograd-f6k3.c) ####

  add_definitions(-DNNP_BACKEND_SCALAR=1)

endif(JEVOIS_PLATFORM)

add_definitions(-DNNPACK)

# Now darknet-nnpack proper:
include_directories(Contrib/darknet-nnpack/include)
set(DKNS "${JVB}/Contrib/darknet-nnpack/src")
add_definitions(-DDARKNET_NNPACK)

# Or use the baseline darknet instead (much slower):
#include_directories(Contrib/darknet/include)
#set(DKNS "${JVB}/Contrib/darknet/src")

target_sources(jevoisbase PRIVATE ${DKNS}/gemm.c ${DKNS}/utils.c ${DKNS}/cuda.c ${DKNS}/deconvolutional_layer.c
  ${DKNS}/convolutional_layer.c ${DKNS}/list.c ${DKNS}/image.c ${DKNS}/activations.c ${DKNS}/im2col.c ${DKNS}/col2im.c
  ${DKNS}/blas.c ${DKNS}/crop_layer.c ${DKNS}/dropout_layer.c ${DKNS}/maxpool_layer.c ${DKNS}/softmax_layer.c
  ${DKNS}/data.c ${DKNS}/matrix.c ${DKNS}/network.c ${DKNS}/connected_layer.c ${DKNS}/cost_layer.c ${DKNS}/parser.c
  ${DKNS}/option_list.c ${DKNS}/detection_layer.c ${DKNS}/route_layer.c ${DKNS}/box.c ${DKNS}/normalization_layer.c
  ${DKNS}/avgpool_layer.c ${DKNS}/layer.c ${DKNS}/local_layer.c ${DKNS}/shortcut_layer.c ${DKNS}/activation_layer.c
  ${DKNS}/rnn_layer.c ${DKNS}/gru_layer.c ${DKNS}/crnn_layer.c ${DKNS}/demo.c ${DKNS}/batchnorm_layer.c
  ${DKNS}/region_layer.c ${DKNS}/reorg_layer.c ${DKNS}/tree.c ${DKNS}/lstm_layer.c)

########################################################################################################################
# ZZBar barcode / QR-code source files:

include_directories(Contrib/ZBar/include Contrib/ZBar/zbar)
set(ZBS "${JVB}/Contrib/ZBar/zbar")
target_sources(jevoisbase PRIVATE ${ZBS}/processor.c ${ZBS}/scanner.c ${ZBS}/symbol.c ${ZBS}/img_scanner.c
  ${ZBS}/qrcode/rs.c ${ZBS}/qrcode/isaac.c ${ZBS}/qrcode/util.c ${ZBS}/qrcode/qrdectxt.c ${ZBS}/qrcode/bch15_5.c
  ${ZBS}/qrcode/binarize.c ${ZBS}/qrcode/qrdec.c ${ZBS}/config.c ${ZBS}/error.c ${ZBS}/processor/posix.c
  ${ZBS}/processor/lock.c ${ZBS}/processor/null.c ${ZBS}/convert.c ${ZBS}/decoder/i25.c ${ZBS}/decoder/qr_finder.c
  ${ZBS}/decoder/code128.c ${ZBS}/decoder/codabar.c ${ZBS}/decoder/code39.c ${ZBS}/decoder/databar.c
  ${ZBS}/decoder/ean.c ${ZBS}/decoder/code93.c ${ZBS}/image.c ${ZBS}/refcnt.c ${ZBS}/decoder.c)
# ${ZBS}/decoder/pdf417.c

# FIXME need to debug zbar as it gives some warnings
add_definitions(-Wparentheses -w)

########################################################################################################################
# cvEyeTracker eye-tracking

target_sources(jevoisbase PRIVATE ${JVB}/Contrib/cvEyeTracker-1.2.5/ransac_ellipse.cpp
  ${JVB}/Contrib/cvEyeTracker-1.2.5/remove_corneal_reflection.cpp ${JVB}/Contrib/cvEyeTracker-1.2.5/svd.c)

########################################################################################################################
# Neon-accelerated NE10 support:
include_directories(Contrib/Ne10/inc)

target_sources(jevoisbase PRIVATE ${JVB}/Contrib/Ne10/modules/imgproc/NE10_boxfilter.c)
if (JEVOIS_PLATFORM)
  target_sources(jevoisbase PRIVATE ${JVB}/Contrib/Ne10/modules/imgproc/NE10_boxfilter.neon.c)
endif (JEVOIS_PLATFORM)

########################################################################################################################
# VLfeat support:
include_directories(Contrib/vlfeat/vl)

# Add VLfeat sources shared among various algorithms:
target_sources(jevoisbase PRIVATE ${JVB}/Contrib/vlfeat/vl/host.c ${JVB}/Contrib/vlfeat/vl/generic.c
  ${JVB}/Contrib/vlfeat/vl/imopv.c)

# The source code for SSE2 convolution seems to be missing...
add_definitions(-DVL_DISABLE_SSE2)

# Other defs to make VLfeat comile:
add_definitions(-DVL_COMPILER_GNUC -DVL_ARCH_LITTLE_ENDIAN)

# Add VLfeat sources used by DenseSift module:
target_sources(jevoisbase PRIVATE ${JVB}/Contrib/vlfeat/vl/dsift.c ${JVB}/Contrib/vlfeat/vl/sift.c)

########################################################################################################################
# Fast optical flow
include_directories(Contrib/OF_DIS)
target_sources(jevoisbase PRIVATE ${JVB}/Contrib/OF_DIS/oflow.cpp ${JVB}/Contrib/OF_DIS/refine_variational.cpp
  ${JVB}/Contrib/OF_DIS/patchgrid.cpp ${JVB}/Contrib/OF_DIS/patch.cpp ${JVB}/Contrib/OF_DIS/FDF1.0.1/image.c
  ${JVB}/Contrib/OF_DIS/FDF1.0.1/opticalflow_aux.c ${JVB}/Contrib/OF_DIS/FDF1.0.1/solver.c)

# Select mode 1 (optical flow) and 1 channel (grayscale):
add_definitions(-DSELECTMODE=1 -DSELECTCHANNEL=1)

# Fix path assumptions in the optical flow code for Eigen:
if (JEVOIS_PLATFORM)
  add_definitions(-I${JEVOIS_BUILDROOT_BASE}/host/usr/arm-buildroot-linux-gnueabihf/sysroot/usr/include/eigen3)
else (JEVOIS_PLATFORM)
  add_definitions(-I/usr/include/eigen3)
endif (JEVOIS_PLATFORM)
  
# Enable OpenMP, which can accelerate the fast optical flow code. NOTE: With small images and when running on a fast
# Intel-based host, this actually slows down the code by 10x or so, probably the parallelism overhead is not worth it
# for small images. Yet, we enable it here as it seems to help on the platform:
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fopenmp")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -fopenmp")
add_definitions(-DWITH_OPENMP -DUSE_PARALLEL_ON_FLOWAGGR)

########################################################################################################################
# ARtoolkit
# When installing ARtoolkit from scratch (already done in jevoisbase):
# run the configure, enable V4L2 and select as default, disable all others

# In jevoisbase, we compile these files:
# find Contrib -name *.c | grep -v VideoLinuxV4L|grep -v 1394|grep -v Eden|grep -v Gl|grep -v examples|grep -v GStrea|grep -v calib|grep -v util|xargs
target_sources(jevoisbase PRIVATE Contrib/ARToolKit5arm/lib/SRC/VideoImage/videoImage.c Contrib/ARToolKit5arm/lib/SRC/KPM/FreakMatcher/unsupported/test/mpreal/dlmalloc.c Contrib/ARToolKit5arm/lib/SRC/KPM/kpmFopen.c Contrib/ARToolKit5arm/lib/SRC/ARICP/icpCalibStereo.c Contrib/ARToolKit5arm/lib/SRC/ARICP/icpCore.c Contrib/ARToolKit5arm/lib/SRC/ARICP/icpUtil.c Contrib/ARToolKit5arm/lib/SRC/ARICP/icpStereoHandle.c Contrib/ARToolKit5arm/lib/SRC/ARICP/icpPointRobust.c Contrib/ARToolKit5arm/lib/SRC/ARICP/icpHandle.c Contrib/ARToolKit5arm/lib/SRC/ARICP/icpStereoPoint.c Contrib/ARToolKit5arm/lib/SRC/ARICP/icpPoint.c Contrib/ARToolKit5arm/lib/SRC/ARICP/icpStereoPointRobust.c Contrib/ARToolKit5arm/lib/SRC/AR2/tracking.c Contrib/ARToolKit5arm/lib/SRC/AR2/featureSet.c Contrib/ARToolKit5arm/lib/SRC/AR2/imageSet.c Contrib/ARToolKit5arm/lib/SRC/AR2/searchPoint.c Contrib/ARToolKit5arm/lib/SRC/AR2/marker.c Contrib/ARToolKit5arm/lib/SRC/AR2/handle.c Contrib/ARToolKit5arm/lib/SRC/AR2/matching2.c Contrib/ARToolKit5arm/lib/SRC/AR2/jpeg.c Contrib/ARToolKit5arm/lib/SRC/AR2/selectTemplate.c Contrib/ARToolKit5arm/lib/SRC/AR2/coord.c Contrib/ARToolKit5arm/lib/SRC/AR2/matching.c Contrib/ARToolKit5arm/lib/SRC/AR2/featureMap.c Contrib/ARToolKit5arm/lib/SRC/AR2/tracking2d.c Contrib/ARToolKit5arm/lib/SRC/AR2/surface.c Contrib/ARToolKit5arm/lib/SRC/AR2/template.c Contrib/ARToolKit5arm/lib/SRC/VideoDummy/videoDummy.c Contrib/ARToolKit5arm/lib/SRC/AR/paramDistortion.c Contrib/ARToolKit5arm/lib/SRC/AR/arCreateHandle.c Contrib/ARToolKit5arm/lib/SRC/AR/vHouse.c Contrib/ARToolKit5arm/lib/SRC/AR/arFilterTransMat.c Contrib/ARToolKit5arm/lib/SRC/AR/vTridiag.c Contrib/ARToolKit5arm/lib/SRC/AR/paramClear.c Contrib/ARToolKit5arm/lib/SRC/AR/arDetectMarker.c Contrib/ARToolKit5arm/lib/SRC/AR/arGetTransMat.c Contrib/ARToolKit5arm/lib/SRC/AR/arPattGetID.c Contrib/ARToolKit5arm/lib/SRC/AR/mDet.c Contrib/ARToolKit5arm/lib/SRC/AR/arLabeling.c Contrib/ARToolKit5arm/lib/SRC/AR/ar3DCreateHandle.c Contrib/ARToolKit5arm/lib/SRC/AR/mFree.c Contrib/ARToolKit5arm/lib/SRC/AR/mTrans.c Contrib/ARToolKit5arm/lib/SRC/AR/arPattCreateHandle.c Contrib/ARToolKit5arm/lib/SRC/AR/mDup.c Contrib/ARToolKit5arm/lib/SRC/AR/arPattSave.c Contrib/ARToolKit5arm/lib/SRC/AR/ar3DUtil.c Contrib/ARToolKit5arm/lib/SRC/AR/arGetTransMatStereo.c Contrib/ARToolKit5arm/lib/SRC/AR/vDisp.c Contrib/ARToolKit5arm/lib/SRC/AR/paramFile.c Contrib/ARToolKit5arm/lib/SRC/AR/mInv.c Contrib/ARToolKit5arm/lib/SRC/AR/vFree.c Contrib/ARToolKit5arm/lib/SRC/AR/mAllocMul.c Contrib/ARToolKit5arm/lib/SRC/AR/arPattAttach.c Contrib/ARToolKit5arm/lib/SRC/AR/paramGetPerspective.c Contrib/ARToolKit5arm/lib/SRC/AR/arImageProc.c Contrib/ARToolKit5arm/lib/SRC/AR/arPattLoad.c Contrib/ARToolKit5arm/lib/SRC/AR/arDetectMarker2.c Contrib/ARToolKit5arm/lib/SRC/AR/mPCA.c Contrib/ARToolKit5arm/lib/SRC/AR/mAlloc.c Contrib/ARToolKit5arm/lib/SRC/AR/mAllocTrans.c Contrib/ARToolKit5arm/lib/SRC/AR/paramDisp.c Contrib/ARToolKit5arm/lib/SRC/AR/vAlloc.c Contrib/ARToolKit5arm/lib/SRC/AR/paramChangeSize.c Contrib/ARToolKit5arm/lib/SRC/AR/mAllocDup.c Contrib/ARToolKit5arm/lib/SRC/AR/arGetLine.c Contrib/ARToolKit5arm/lib/SRC/AR/mAllocInv.c Contrib/ARToolKit5arm/lib/SRC/AR/mAllocUnit.c Contrib/ARToolKit5arm/lib/SRC/AR/mUnit.c Contrib/ARToolKit5arm/lib/SRC/AR/mDisp.c Contrib/ARToolKit5arm/lib/SRC/AR/paramLT.c Contrib/ARToolKit5arm/lib/SRC/AR/mMul.c Contrib/ARToolKit5arm/lib/SRC/AR/mSelfInv.c Contrib/ARToolKit5arm/lib/SRC/AR/vInnerP.c Contrib/ARToolKit5arm/lib/SRC/AR/arGetMarkerInfo.c Contrib/ARToolKit5arm/lib/SRC/AR/paramDecomp.c Contrib/ARToolKit5arm/lib/SRC/AR/arLabelingSub/arLabelingSubEBI3C565.c Contrib/ARToolKit5arm/lib/SRC/AR/arLabelingSub/arLabelingSubEWIYC.c Contrib/ARToolKit5arm/lib/SRC/AR/arLabelingSub/arLabelingSubDBI3CA4444.c Contrib/ARToolKit5arm/lib/SRC/AR/arLabelingSub/arLabelingSubEBR3C565.c Contrib/ARToolKit5arm/lib/SRC/AR/arLabelingSub/arLabelingSubDWI3C.c Contrib/ARToolKit5arm/lib/SRC/AR/arLabelingSub/arLabelingSubEBICY.c Contrib/ARToolKit5arm/lib/SRC/AR/arLabelingSub/arLabelingSubEWRCY.c Contrib/ARToolKit5arm/lib/SRC/AR/arLabelingSub/arLabelingSubEWI3C.c Contrib/ARToolKit5arm/lib/SRC/AR/arLabelingSub/arLabelingSubEWI3CA5551.c Contrib/ARToolKit5arm/lib/SRC/AR/arLabelingSub/arLabelingSubDBR3CA4444.c Contrib/ARToolKit5arm/lib/SRC/AR/arLabelingSub/arLabelingSubDBR3CA5551.c Contrib/ARToolKit5arm/lib/SRC/AR/arLabelingSub/arLabelingSubDBRA3C.c Contrib/ARToolKit5arm/lib/SRC/AR/arLabelingSub/arLabelingSubEBRC.c Contrib/ARToolKit5arm/lib/SRC/AR/arLabelingSub/arLabelingSubEBR3C.c Contrib/ARToolKit5arm/lib/SRC/AR/arLabelingSub/arLabelingSubDBICY.c Contrib/ARToolKit5arm/lib/SRC/AR/arLabelingSub/arLabelingSubDWRC.c Contrib/ARToolKit5arm/lib/SRC/AR/arLabelingSub/arLabelingSubDBR3C565.c Contrib/ARToolKit5arm/lib/SRC/AR/arLabelingSub/arLabelingSubEWRYC.c Contrib/ARToolKit5arm/lib/SRC/AR/arLabelingSub/arLabelingSubDWI3CA5551.c Contrib/ARToolKit5arm/lib/SRC/AR/arLabelingSub/arLabelingSubEWRC.c Contrib/ARToolKit5arm/lib/SRC/AR/arLabelingSub/arLabelingSubEBRA3C.c Contrib/ARToolKit5arm/lib/SRC/AR/arLabelingSub/arLabelingSubEBI3C.c Contrib/ARToolKit5arm/lib/SRC/AR/arLabelingSub/arLabelingSubDWIYC.c Contrib/ARToolKit5arm/lib/SRC/AR/arLabelingSub/arLabelingSubEWI3C565.c Contrib/ARToolKit5arm/lib/SRC/AR/arLabelingSub/arLabelingSubEWR3C565.c Contrib/ARToolKit5arm/lib/SRC/AR/arLabelingSub/arLabelingSubDWR3CA5551.c Contrib/ARToolKit5arm/lib/SRC/AR/arLabelingSub/arLabelingSubDBI3CA5551.c Contrib/ARToolKit5arm/lib/SRC/AR/arLabelingSub/arLabelingSubEWR3CA5551.c Contrib/ARToolKit5arm/lib/SRC/AR/arLabelingSub/arLabelingSubEBR3CA5551.c Contrib/ARToolKit5arm/lib/SRC/AR/arLabelingSub/arLabelingSubEWR3CA.c Contrib/ARToolKit5arm/lib/SRC/AR/arLabelingSub/arLabelingSubEWR3CA4444.c Contrib/ARToolKit5arm/lib/SRC/AR/arLabelingSub/arLabelingSubEBZ.c Contrib/ARToolKit5arm/lib/SRC/AR/arLabelingSub/arLabelingSubDBI3C565.c Contrib/ARToolKit5arm/lib/SRC/AR/arLabelingSub/arLabelingSubEBI3CA.c Contrib/ARToolKit5arm/lib/SRC/AR/arLabelingSub/arLabelingSubDBZ.c Contrib/ARToolKit5arm/lib/SRC/AR/arLabelingSub/arLabelingSubDBIC.c Contrib/ARToolKit5arm/lib/SRC/AR/arLabelingSub/arLabelingSubDWI3C565.c Contrib/ARToolKit5arm/lib/SRC/AR/arLabelingSub/arLabelingSubDWR3CA.c Contrib/ARToolKit5arm/lib/SRC/AR/arLabelingSub/arLabelingSubEBIC.c Contrib/ARToolKit5arm/lib/SRC/AR/arLabelingSub/arLabelingSubDWI3CA4444.c Contrib/ARToolKit5arm/lib/SRC/AR/arLabelingSub/arLabelingSubEBRCY.c Contrib/ARToolKit5arm/lib/SRC/AR/arLabelingSub/arLabelingSubEBR3CA4444.c Contrib/ARToolKit5arm/lib/SRC/AR/arLabelingSub/arLabelingSubEWIC.c Contrib/ARToolKit5arm/lib/SRC/AR/arLabelingSub/arLabelingSubDBR3CA.c Contrib/ARToolKit5arm/lib/SRC/AR/arLabelingSub/arLabelingSubDWIA3C.c Contrib/ARToolKit5arm/lib/SRC/AR/arLabelingSub/arLabelingSubDBI3CA.c Contrib/ARToolKit5arm/lib/SRC/AR/arLabelingSub/arLabelingSubDBRYC.c Contrib/ARToolKit5arm/lib/SRC/AR/arLabelingSub/arLabelingSubDWI3CA.c Contrib/ARToolKit5arm/lib/SRC/AR/arLabelingSub/arLabelingSubDBIA3C.c Contrib/ARToolKit5arm/lib/SRC/AR/arLabelingSub/arLabelingSubEBIYC.c Contrib/ARToolKit5arm/lib/SRC/AR/arLabelingSub/arLabelingSubDBRC.c Contrib/ARToolKit5arm/lib/SRC/AR/arLabelingSub/arLabelingSubEBI3CA4444.c Contrib/ARToolKit5arm/lib/SRC/AR/arLabelingSub/arLabelingSubEWZ.c Contrib/ARToolKit5arm/lib/SRC/AR/arLabelingSub/arLabelingSubDWR3CA4444.c Contrib/ARToolKit5arm/lib/SRC/AR/arLabelingSub/arLabelingSubDBR3C.c Contrib/ARToolKit5arm/lib/SRC/AR/arLabelingSub/arLabelingSubDWRCY.c Contrib/ARToolKit5arm/lib/SRC/AR/arLabelingSub/arLabelingSubEBI3CA5551.c Contrib/ARToolKit5arm/lib/SRC/AR/arLabelingSub/arLabelingSubDWRA3C.c Contrib/ARToolKit5arm/lib/SRC/AR/arLabelingSub/arLabelingSubEWRA3C.c Contrib/ARToolKit5arm/lib/SRC/AR/arLabelingSub/arLabelingSubDWRYC.c Contrib/ARToolKit5arm/lib/SRC/AR/arLabelingSub/arLabelingSubDWR3C565.c Contrib/ARToolKit5arm/lib/SRC/AR/arLabelingSub/arLabelingSubEWICY.c Contrib/ARToolKit5arm/lib/SRC/AR/arLabelingSub/arLabelingSubDBI3C.c Contrib/ARToolKit5arm/lib/SRC/AR/arLabelingSub/arLabelingSubDWIC.c Contrib/ARToolKit5arm/lib/SRC/AR/arLabelingSub/arLabelingSubDWR3C.c Contrib/ARToolKit5arm/lib/SRC/AR/arLabelingSub/arLabelingSubDWICY.c Contrib/ARToolKit5arm/lib/SRC/AR/arLabelingSub/arLabelingSubEWR3C.c Contrib/ARToolKit5arm/lib/SRC/AR/arLabelingSub/arLabelingSubDBIYC.c Contrib/ARToolKit5arm/lib/SRC/AR/arLabelingSub/arLabelingSubDWZ.c Contrib/ARToolKit5arm/lib/SRC/AR/arLabelingSub/arLabelingSubEBIA3C.c Contrib/ARToolKit5arm/lib/SRC/AR/arLabelingSub/arLabelingSubDBRCY.c Contrib/ARToolKit5arm/lib/SRC/AR/arLabelingSub/arLabelingSubEWI3CA.c Contrib/ARToolKit5arm/lib/SRC/AR/arLabelingSub/arLabelingSubEWI3CA4444.c Contrib/ARToolKit5arm/lib/SRC/AR/arLabelingSub/arLabelingSubEBR3CA.c Contrib/ARToolKit5arm/lib/SRC/AR/arLabelingSub/arLabelingSubEWIA3C.c Contrib/ARToolKit5arm/lib/SRC/AR/arLabelingSub/arLabelingSubEBRYC.c Contrib/ARToolKit5arm/lib/SRC/AR/arUtil.c Contrib/ARToolKit5arm/lib/SRC/ARMulti/arMultiReadConfigFile.c Contrib/ARToolKit5arm/lib/SRC/ARMulti/arMultiGetTransMat.c Contrib/ARToolKit5arm/lib/SRC/ARMulti/arMultiGetTransMatStereo.c Contrib/ARToolKit5arm/lib/SRC/ARMulti/arMultiFreeConfig.c Contrib/ARToolKit5arm/lib/SRC/Video/video2.c Contrib/ARToolKit5arm/lib/SRC/Video/videoAspectRatio.c Contrib/ARToolKit5arm/lib/SRC/Video/video.c Contrib/ARToolKit5arm/lib/SRC/Video/videoSaveImage.c Contrib/ARToolKit5arm/lib/SRC/Util/profile.c Contrib/ARToolKit5arm/lib/SRC/Util/thread_sub.c Contrib/ARToolKit5arm/lib/SRC/ARWrapper/trackingSub.c)

include_directories(Contrib/ARToolKit5arm/include)

########################################################################################################################
# Link to OpenCV libraries for superpixels, aruco, and others (from pkg-config --libs opencv and deleting a few):

target_link_libraries(jevoisbase ${JEVOIS_OPENCV_LIBS} opencv_stitching opencv_superres opencv_videostab
  opencv_aruco opencv_bgsegm opencv_bioinspired opencv_ccalib opencv_dnn opencv_dpm opencv_fuzzy
  opencv_line_descriptor opencv_optflow opencv_plot opencv_reg opencv_saliency opencv_stereo opencv_structured_light
  opencv_rgbd opencv_surface_matching opencv_tracking opencv_datasets opencv_text opencv_face opencv_xfeatures2d
  opencv_shape opencv_video opencv_ximgproc opencv_calib3d opencv_features2d opencv_flann opencv_xobjdetect
  opencv_objdetect opencv_ml opencv_xphoto opencv_highgui opencv_videoio opencv_imgcodecs opencv_photo
  opencv_imgproc opencv_core)

########################################################################################################################
# Install shared resources (cascade classifiers, neural network weights, etc):

install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/share"
  DESTINATION "${JEVOIS_INSTALL_ROOT}" COMPONENT bin)

########################################################################################################################
# Documentation:

add_custom_target(doc COMMAND doxygen "${CMAKE_CURRENT_SOURCE_DIR}/doc/doxygen.cfg"
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})

# This is to install the doc of jevois and jevoisbase to jevois.org, only works in ilab:
if ($ENV{JEVOIS_ILAB})
  add_custom_target(docweb
    COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/scripts/docinstall.sh
    DEPENDS doc
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
endif ($ENV{JEVOIS_ILAB})

# Documentation files for our install / distribution package
set(DOC_FILES README INSTALL COPYING)
if (JEVOIS_PLATFORM)
  set(DOC_PATH "share/doc/jevoisbase-platform")
else (JEVOIS_PLATFORM)
  set(DOC_PATH "share/doc/jevoisbase-host")
endif (JEVOIS_PLATFORM)
install(FILES ${DOC_FILES} DESTINATION ${DOC_PATH} COMPONENT bin)

########################################################################################################################
# Debian packaging:

# Create packages (Debian, RPM): in hbuild/ or pbuild/, just type 'sudo cpack' to create the package.
# To list the files created in a package, run: dpkg --contents <package.deb>
set(CPACK_PACKAGE_DESCRIPTION "JeVois Smart Machine Vision Base Modules")
set(CPACK_PACKAGE_DESCRIPTION_SUMMARY "JeVois Smart Embedded Machine Vision Toolkit, Base Modules")
set(CPACK_PACKAGE_CONTACT "Laurent Itti <jevois.org@gmail.com>")
set(CPACK_DEBIAN_PACKAGE_MAINTAINER ${CPACK_PACKAGE_CONTACT})
set(CPACK_DEBIAN_PACKAGE_SECTION "universe")
set(CPACK_PACKAGE_VENDOR "iLab at the University of Southern California")
set(CPACK_PACKAGE_VERSION_MAJOR "${JEVOIS_VERSION_MAJOR}")  # Note: jevoisbase version tracks jevois version
set(CPACK_PACKAGE_VERSION_MINOR "${JEVOIS_VERSION_MINOR}")
set(CPACK_PACKAGE_VERSION_PATCH "${JEVOIS_VERSION_PATCH}")
set(JEVOIS_PACKAGE_RELEASE "1") # packager revision number

if (JEVOIS_PLATFORM)
  set(JEVOIS_DEPEND "jevois-platform (>=${JEVOIS_VERSION_MAJOR}.${JEVOIS_VERSION_MINOR}.${JEVOIS_VERSION_PATCH})")
else (JEVOIS_PLATFORM)
  set(JEVOIS_DEPEND "jevois-host (>=${JEVOIS_VERSION_MAJOR}.${JEVOIS_VERSION_MINOR}.${JEVOIS_VERSION_PATCH})")
endif (JEVOIS_PLATFORM)

set(CPACK_DEBIAN_PACKAGE_DEPENDS "${JEVOIS_DEPEND}, libgles2-mesa, libgles2-mesa-dev")

# Use helper from JeVois.cmake for all other settings:
jevois_setup_cpack("jevoisbase")


########################################################################################################################
# Wrapping up:
jevois_project_finalize()  # call this last in your CMakeLists.txt, will un-mount the live microSD if applicable
