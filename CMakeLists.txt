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
if (JEVOIS_HARDWARE STREQUAL "PRO")
  set(JEVOIS_CONFIG "/jevoispro/config" CACHE STRING "Path to JeVois config to use")
else()
  set(JEVOIS_CONFIG "/jevois/config" CACHE STRING "Path to JeVois config to use")
endif()

# Include our helper functions, config, etc from the JeVois install:
set(CMAKE_MODULE_PATH ${JEVOIS_CONFIG})
include(jevois_config)
include(JeVois)

# Set project name, detects compiler (which has been set by our helper module). Then set some complation flags:
project(${JEVOIS}base CXX C ASM) # NNPACK has a few assembly sources, darknet and others are C
jevois_project_set_flags()

set(JVB "${CMAKE_CURRENT_SOURCE_DIR}")

# Allow #include <jevoibase/x/y> in our components and modules:
include_directories("include")

# Allow #include <jevoibase/Contrib/x/y> in our components and modules:
include_directories("..")

# Setup our library using the source files in src/Components and call it jevoisbase with the current version number:
set(JEVOISBASE_SOVERSION "1.18.0")
jevois_setup_library(src/Components ${JEVOIS}base ${JEVOISBASE_SOVERSION})

# Check that we are not trying to compile a newer jevoisbase against an older jevois-sdk, which would typically fail:
jevois_check_sdk_version(${JEVOISBASE_SOVERSION})

# Setup our modules that are in src/Modules, make them depend on jevoisbase:
jevois_setup_modules(src/Modules ${JEVOIS}base)

# Add includes to our install / distribution package:
add_subdirectory(include)

# Now setup any additional includes and sources that we will add to the jevoisbase library for use by our individual
# modules, typically those are in the Contrib/ drectory:

########################################################################################################################
# Enable address sanitizer (to debug illegal memory accesses and such; should normally be commented out):
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=address")


########################################################################################################################
# Find any include files installed by JeVois into /${JEVOIS}/include:
include_directories(${JEVOIS_MODULES_ROOT}/include)

########################################################################################################################
# OpenVX and ovxlib support
if (JEVOIS_PLATFORM)
  include_directories(${JEVOIS_BUILD_BASE}/usr/include/ovxlib)
  include_directories(${JEVOIS_BUILD_BASE}/usr/include/VX)
else ()
  include_directories(/usr/include/ovxlib)
  include_directories(/usr/include/VX)
endif ()

########################################################################################################################
# OpenGL-ES libraries on platform (and they also exist on host):
target_link_libraries(${JEVOIS}base GLESv2 EGL)

########################################################################################################################
# Add any needed boost libraries that are not already pulled in by libjevois:
target_link_libraries(${JEVOIS}base boost_regex)

########################################################################################################################
# tiny-dnn support:
include_directories(Contrib)
include_directories(Contrib/tiny-dnn)

include_directories(Contrib/pthreadpool/include)

target_sources(${JEVOIS}base PRIVATE Contrib/pthreadpool/src/threadpool-pthreads.c)
add_definitions(-DTBB_SUPPRESS_DEPRECATED_MESSAGES) # avoid message about deprecated code in tbb.h
if (JEVOIS_PRO)
  if (JEVOIS_PLATFORM)
    add_definitions(-D__ARM_NEON__) # Allow tiny-dnn to detect that we support neon instructions
  endif ()
endif ()

########################################################################################################################
# darknet with NNPACK support: (on top of things already pulled in by tiny-dnn)

include_directories(Contrib/NNPACK/include)
include_directories(Contrib/NNPACK/src)
include_directories(Contrib/FP16/include)
include_directories(Contrib/FXdiv/include)
include_directories(Contrib/psimd/include)

# Library cpuinfo (dependency for NNPACK):
include_directories(Contrib/cpuinfo/include)
include_directories(Contrib/cpuinfo/deps/clog/include)
include_directories(Contrib/cpuinfo/src)

set(CPUI "${JVB}/Contrib/cpuinfo/src")

target_sources(${JEVOIS}base PRIVATE ${JVB}/Contrib/cpuinfo/deps/clog/src/clog.c ${CPUI}/init.c ${CPUI}/api.c
  ${CPUI}/linux/gpu.c ${CPUI}/linux/smallfile.c ${CPUI}/linux/processors.c ${CPUI}/linux/current.c
  ${CPUI}/linux/multiline.c ${CPUI}/linux/api.h ${CPUI}/linux/cpulist.c)
  
if (JEVOIS_PLATFORM)
  if (JEVOIS_PRO)
    target_sources(${JEVOIS}base PRIVATE ${CPUI}/arm/uarch.c ${CPUI}/arm/cache.c ${CPUI}/arm/linux/init.c
      ${CPUI}/arm/linux/cpuinfo.c ${CPUI}/arm/linux/clusters.c ${CPUI}/arm/linux/chipset.c ${CPUI}/arm/linux/midr.c
      ${CPUI}/arm/linux/hwcap.c ${CPUI}/arm/linux/aarch64-isa.c)
  else ()
    target_sources(${JEVOIS}base PRIVATE ${CPUI}/arm/uarch.c ${CPUI}/arm/cache.c ${CPUI}/arm/linux/init.c
      ${CPUI}/arm/linux/cpuinfo.c ${CPUI}/arm/linux/clusters.c ${CPUI}/arm/linux/chipset.c ${CPUI}/arm/linux/midr.c
      ${CPUI}/arm/linux/hwcap.c ${CPUI}/arm/linux/aarch32-isa.c)
  endif ()
else (JEVOIS_PLATFORM)
  target_sources(${JEVOIS}base PRIVATE ${CPUI}/x86/init.c ${CPUI}/x86/info.c ${CPUI}/x86/vendor.c ${CPUI}/x86/uarch.c
    ${CPUI}/x86/name.c ${CPUI}/x86/topology.c ${CPUI}/x86/isa.c ${CPUI}/x86/cache/init.c ${CPUI}/x86/cache/descriptor.c
    ${CPUI}/x86/cache/deterministic.c ${CPUI}/x86/linux/init.c ${CPUI}/x86/linux/cpuinfo.c)
endif(JEVOIS_PLATFORM)

# NNPACK: Core
set(NNPS "${JVB}/Contrib/NNPACK/src")
target_sources(${JEVOIS}base PRIVATE ${NNPS}/init.c)
target_sources(${JEVOIS}base PRIVATE ${NNPS}/convolution-output.c)
target_sources(${JEVOIS}base PRIVATE ${NNPS}/convolution-input-gradient.c)
target_sources(${JEVOIS}base PRIVATE ${NNPS}/convolution-kernel-gradient.c)
target_sources(${JEVOIS}base PRIVATE ${NNPS}/convolution-inference.c)
target_sources(${JEVOIS}base PRIVATE ${NNPS}/fully-connected-output.c)
target_sources(${JEVOIS}base PRIVATE ${NNPS}/fully-connected-inference.c)
target_sources(${JEVOIS}base PRIVATE ${NNPS}/pooling-output.c)
target_sources(${JEVOIS}base PRIVATE ${NNPS}/softmax-output.c)
target_sources(${JEVOIS}base PRIVATE ${NNPS}/relu-output.c)
target_sources(${JEVOIS}base PRIVATE ${NNPS}/relu-input-gradient.c)

if (JEVOIS_PLATFORM)
  # Transformations
  target_sources(${JEVOIS}base PRIVATE ${NNPS}/psimd/2d-fourier-8x8.c)
  target_sources(${JEVOIS}base PRIVATE ${NNPS}/psimd/2d-fourier-16x16.c)
  target_sources(${JEVOIS}base PRIVATE ${NNPS}/neon/2d-winograd-8x8-3x3.c)
  target_sources(${JEVOIS}base PRIVATE ${NNPS}/neon/2d-winograd-8x8-3x3-fp16.c)
  # ReLU and Softmax
  target_sources(${JEVOIS}base PRIVATE ${NNPS}/neon/relu.c)
  target_sources(${JEVOIS}base PRIVATE ${NNPS}/psimd/softmax.c)
  # FFT block accumulation
  #target_sources(${JEVOIS}base PRIVATE ${NNPS}/psimd/fft-block-mac.c)
  # Tuple GEMM
  target_sources(${JEVOIS}base PRIVATE ${NNPS}/neon/blas/h4gemm.c)
  target_sources(${JEVOIS}base PRIVATE ${NNPS}/neon/blas/s4gemm.c)
  target_sources(${JEVOIS}base PRIVATE ${NNPS}/neon/blas/c4gemm.c)
  target_sources(${JEVOIS}base PRIVATE ${NNPS}/neon/blas/s4c2gemm.c)
  target_sources(${JEVOIS}base PRIVATE ${NNPS}/neon/blas/c4gemm-conjb.c)
  target_sources(${JEVOIS}base PRIVATE ${NNPS}/neon/blas/s4c2gemm-conjb.c)
  target_sources(${JEVOIS}base PRIVATE ${NNPS}/neon/blas/c4gemm-conjb-transc.c)
  target_sources(${JEVOIS}base PRIVATE ${NNPS}/neon/blas/s4c2gemm-conjb-transc.c)
  # Direct convolution
  target_sources(${JEVOIS}base PRIVATE ${NNPS}/neon/blas/conv1x1.c)
  # BLAS microkernels
  target_sources(${JEVOIS}base PRIVATE ${NNPS}/neon/blas/sgemm.c)
  target_sources(${JEVOIS}base PRIVATE ${NNPS}/neon/blas/sdotxf.c)
  target_sources(${JEVOIS}base PRIVATE ${NNPS}/psimd/blas/shdotxf.c)
  if (JEVOIS_A33)
    target_sources(${JEVOIS}base PRIVATE ${NNPS}/neon/blas/h4gemm-aarch32.S)
    target_sources(${JEVOIS}base PRIVATE ${NNPS}/neon/blas/s4gemm-aarch32.S)
    target_sources(${JEVOIS}base PRIVATE ${NNPS}/neon/blas/sgemm-aarch32.S)
  endif ()
  
   # FFT and Winograd
  target_sources(${JEVOIS}base PRIVATE ${NNPS}/psimd/fft-aos.c)
  target_sources(${JEVOIS}base PRIVATE ${NNPS}/psimd/fft-soa.c)
  target_sources(${JEVOIS}base PRIVATE ${NNPS}/psimd/fft-real.c)
  target_sources(${JEVOIS}base PRIVATE ${NNPS}/psimd/fft-dualreal.c)
  target_sources(${JEVOIS}base PRIVATE ${NNPS}/neon/winograd-f6k3.c)
  target_sources(${JEVOIS}base PRIVATE ${NNPS}/psimd/exp.c)

  add_definitions(-DARM_NEON)

else (JEVOIS_PLATFORM)

  # NOTE: on host we just use scalar as opposed to x86 SIMD extensions
  # Transformations
  target_sources(${JEVOIS}base PRIVATE ${NNPS}/scalar/2d-fourier-8x8.c)
  target_sources(${JEVOIS}base PRIVATE ${NNPS}/scalar/2d-fourier-16x16.c)
  target_sources(${JEVOIS}base PRIVATE ${NNPS}/scalar/2d-winograd-8x8-3x3.c)
  # ReLU and Softmax
  target_sources(${JEVOIS}base PRIVATE ${NNPS}/scalar/relu.c)
  target_sources(${JEVOIS}base PRIVATE ${NNPS}/scalar/softmax.c)
  # Tuple GEMM
  target_sources(${JEVOIS}base PRIVATE ${NNPS}/scalar/blas/s2gemm.c)
  target_sources(${JEVOIS}base PRIVATE ${NNPS}/scalar/blas/s2gemm-transc.c)
  target_sources(${JEVOIS}base PRIVATE ${NNPS}/scalar/blas/cgemm.c)
  target_sources(${JEVOIS}base PRIVATE ${NNPS}/scalar/blas/cgemm-conjb.c)
  target_sources(${JEVOIS}base PRIVATE ${NNPS}/scalar/blas/cgemm-conjb-transc.c)
  # Direct convolution
  target_sources(${JEVOIS}base PRIVATE ${NNPS}/scalar/blas/conv1x1.c)
  # BLAS microkernels
  target_sources(${JEVOIS}base PRIVATE ${NNPS}/scalar/blas/sgemm.c)
  target_sources(${JEVOIS}base PRIVATE ${NNPS}/scalar/blas/sdotxf.c)
  target_sources(${JEVOIS}base PRIVATE ${NNPS}/scalar/blas/shdotxf.c)
  # FFT and Winograd
  target_sources(${JEVOIS}base PRIVATE ${NNPS}/scalar/fft-aos.c)
  target_sources(${JEVOIS}base PRIVATE ${NNPS}/scalar/fft-soa.c)
  target_sources(${JEVOIS}base PRIVATE ${NNPS}/scalar/fft-real.c)
  target_sources(${JEVOIS}base PRIVATE ${NNPS}/scalar/fft-dualreal.c)
  target_sources(${JEVOIS}base PRIVATE ${NNPS}/scalar/winograd-f6k3.c)

  add_definitions(-DNNP_BACKEND_SCALAR=1)

endif(JEVOIS_PLATFORM)

add_definitions(-DNNPACK)

# Now darknet-nnpack proper:
include_directories(Contrib/darknet-nnpack/include)
include_directories(Contrib/darknet-nnpack/3rdparty/stb/include)
set(DKNS "${JVB}/Contrib/darknet-nnpack/src")
add_definitions(-DDARKNET_NNPACK)

# Or use the baseline darknet instead (much slower):
#include_directories(Contrib/darknet/include)
#set(DKNS "${JVB}/Contrib/darknet/src")

# Get the list of sources from:
# cd Contrib/darknet-nnpack/src
# find . -name "*.c" |grep -v darknet.c|sed -e 's/\.\//\$\{DKNS\}\//g' | xargs

target_sources(${JEVOIS}base PRIVATE ${DKNS}/go.c ${DKNS}/swag.c ${DKNS}/maxpool_layer.c ${DKNS}/avgpool_layer.c
  ${DKNS}/nightmare.c ${DKNS}/region_layer.c ${DKNS}/im2col.c ${DKNS}/yolo.c ${DKNS}/parser.c
  ${DKNS}/dark_cuda.c ${DKNS}/gaussian_yolo_layer.c ${DKNS}/tag.c ${DKNS}/lstm_layer.c ${DKNS}/demo.c
  ${DKNS}/dice.c ${DKNS}/captcha.c ${DKNS}/softmax_layer.c ${DKNS}/network.c ${DKNS}/cpu_gemm.c
  ${DKNS}/rnn_layer.c ${DKNS}/dropout_layer.c ${DKNS}/conv_lstm_layer.c ${DKNS}/upsample_layer.c
  ${DKNS}/option_list.c ${DKNS}/convolutional_layer.c ${DKNS}/gemm.c ${DKNS}/reorg_layer.c
  ${DKNS}/classifier.c ${DKNS}/matrix.c ${DKNS}/cost_layer.c ${DKNS}/voxel.c ${DKNS}/crnn_layer.c
  ${DKNS}/deconvolutional_layer.c ${DKNS}/gettimeofday.c ${DKNS}/batchnorm_layer.c ${DKNS}/activations.c
  ${DKNS}/layer.c ${DKNS}/super.c ${DKNS}/crop_layer.c ${DKNS}/writing.c ${DKNS}/list.c
  ${DKNS}/reorg_old_layer.c ${DKNS}/route_layer.c ${DKNS}/utils.c ${DKNS}/blas.c ${DKNS}/col2im.c
  ${DKNS}/normalization_layer.c ${DKNS}/getopt.c ${DKNS}/local_layer.c ${DKNS}/tree.c ${DKNS}/detector.c
  ${DKNS}/art.c ${DKNS}/rnn.c ${DKNS}/shortcut_layer.c ${DKNS}/box.c ${DKNS}/compare.c ${DKNS}/yolo_layer.c
  ${DKNS}/image.c ${DKNS}/cifar.c ${DKNS}/activation_layer.c ${DKNS}/gru_layer.c ${DKNS}/detection_layer.c
  ${DKNS}/data.c ${DKNS}/coco.c ${DKNS}/scale_channels_layer.c ${DKNS}/connected_layer.c ${DKNS}/sam_layer.c
  ${DKNS}/rnn_vid.c)

########################################################################################################################
# ZZBar barcode / QR-code source files:

include_directories(Contrib/ZBar/include Contrib/ZBar/zbar)
set(ZBS "${JVB}/Contrib/ZBar/zbar")
target_sources(${JEVOIS}base PRIVATE ${ZBS}/processor.c ${ZBS}/scanner.c ${ZBS}/symbol.c ${ZBS}/img_scanner.c
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

# This uses legacy opencv API which is not supported anymore in opencv 4.x.

#target_sources(${JEVOIS}base PRIVATE ${JVB}/Contrib/cvEyeTracker-1.2.5/ransac_ellipse.cpp
#  ${JVB}/Contrib/cvEyeTracker-1.2.5/remove_corneal_reflection.cpp ${JVB}/Contrib/cvEyeTracker-1.2.5/svd.c)

########################################################################################################################
# Neon-accelerated Ne10 support. NOTE: as of JeVois 1.13.0, Ne10 is also installed on platform by default as a library,
# may want to use that in new code:
include_directories(Contrib/Ne10/inc)

target_sources(${JEVOIS}base PRIVATE ${JVB}/Contrib/Ne10/modules/imgproc/NE10_boxfilter.c)
if (JEVOIS_PLATFORM)
  target_sources(${JEVOIS}base PRIVATE ${JVB}/Contrib/Ne10/modules/imgproc/NE10_boxfilter.neon.c)
endif (JEVOIS_PLATFORM)

########################################################################################################################
# VLfeat support:
include_directories(Contrib/vlfeat/vl)

# Add VLfeat sources shared among various algorithms:
target_sources(${JEVOIS}base PRIVATE ${JVB}/Contrib/vlfeat/vl/host.c ${JVB}/Contrib/vlfeat/vl/generic.c
  ${JVB}/Contrib/vlfeat/vl/imopv.c)

# The source code for SSE2 convolution seems to be missing...
add_definitions(-DVL_DISABLE_SSE2)

# Other defs to make VLfeat comile:
add_definitions(-DVL_COMPILER_GNUC -DVL_ARCH_LITTLE_ENDIAN)

# Add VLfeat sources used by DenseSift module:
target_sources(${JEVOIS}base PRIVATE ${JVB}/Contrib/vlfeat/vl/dsift.c ${JVB}/Contrib/vlfeat/vl/sift.c)

########################################################################################################################
# Fast optical flow
include_directories(Contrib/OF_DIS)
target_sources(${JEVOIS}base PRIVATE ${JVB}/Contrib/OF_DIS/oflow.cpp ${JVB}/Contrib/OF_DIS/refine_variational.cpp
  ${JVB}/Contrib/OF_DIS/patchgrid.cpp ${JVB}/Contrib/OF_DIS/patch.cpp ${JVB}/Contrib/OF_DIS/FDF1.0.1/image.c
  ${JVB}/Contrib/OF_DIS/FDF1.0.1/opticalflow_aux.c ${JVB}/Contrib/OF_DIS/FDF1.0.1/solver.c)

# Select mode 1 (optical flow) and 1 channel (grayscale):
add_definitions(-DSELECTMODE=1 -DSELECTCHANNEL=1)

# Fix path assumptions in the optical flow code for Eigen:
if (JEVOIS_PLATFORM)
  if (JEVOIS_PRO)
    include_directories(${JEVOIS_BUILD_BASE}/usr/include/eigen3)
  else (JEVOIS_PRO)
    include_directories(${JEVOIS_BUILD_BASE}/host/usr/arm-buildroot-linux-gnueabihf/sysroot/usr/include/eigen3)
  endif(JEVOIS_PRO)
else (JEVOIS_PLATFORM)
  include_directories(/usr/include/eigen3)
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
set(ARTK "${JVB}/Contrib/ARToolKit5arm/lib/SRC")

# In jevoisbase, we compile these files:
# cd Contrib/ARToolKit5arm/lib/SRC
# find . -name *.c | grep -v VideoLinuxV4L|grep -v 1394|grep -v Eden|grep -v Gl|grep -v examples|grep -v GStrea|grep -v calib|grep -v util|sed -e 's/\.\//\$\{ARTK\}\//g' | xargs
target_sources(${JEVOIS}base PRIVATE ${ARTK}/ARMulti/arMultiGetTransMatStereo.c ${ARTK}/ARMulti/arMultiFreeConfig.c
  ${ARTK}/ARMulti/arMultiGetTransMat.c ${ARTK}/ARMulti/arMultiReadConfigFile.c ${ARTK}/VideoDummy/videoDummy.c
  ${ARTK}/AR2/coord.c ${ARTK}/AR2/marker.c ${ARTK}/AR2/matching.c ${ARTK}/AR2/tracking.c ${ARTK}/AR2/featureSet.c
  ${ARTK}/AR2/jpeg.c ${ARTK}/AR2/tracking2d.c ${ARTK}/AR2/selectTemplate.c ${ARTK}/AR2/featureMap.c ${ARTK}/AR2/handle.c
  ${ARTK}/AR2/imageSet.c ${ARTK}/AR2/matching2.c ${ARTK}/AR2/searchPoint.c ${ARTK}/AR2/template.c ${ARTK}/AR2/surface.c
  ${ARTK}/VideoImage/videoImage.c ${ARTK}/ARWrapper/trackingSub.c ${ARTK}/Video/video2.c ${ARTK}/Video/videoSaveImage.c
  ${ARTK}/Video/videoAspectRatio.c ${ARTK}/Video/video.c ${ARTK}/KPM/kpmFopen.c
  ${ARTK}/KPM/FreakMatcher/unsupported/test/mpreal/dlmalloc.c ${ARTK}/AR/arPattLoad.c ${ARTK}/AR/ar3DUtil.c
  ${ARTK}/AR/vHouse.c ${ARTK}/AR/paramDecomp.c ${ARTK}/AR/arImageProc.c ${ARTK}/AR/mSelfInv.c ${ARTK}/AR/arGetTransMat.c
  ${ARTK}/AR/paramDisp.c ${ARTK}/AR/mDisp.c ${ARTK}/AR/mUnit.c ${ARTK}/AR/paramClear.c ${ARTK}/AR/arDetectMarker2.c
  ${ARTK}/AR/mAlloc.c ${ARTK}/AR/mAllocInv.c ${ARTK}/AR/paramChangeSize.c ${ARTK}/AR/vTridiag.c
  ${ARTK}/AR/paramDistortion.c ${ARTK}/AR/mAllocTrans.c ${ARTK}/AR/mMul.c ${ARTK}/AR/mAllocMul.c
  ${ARTK}/AR/arPattAttach.c ${ARTK}/AR/vInnerP.c ${ARTK}/AR/vAlloc.c ${ARTK}/AR/arUtil.c ${ARTK}/AR/mAllocDup.c
  ${ARTK}/AR/ar3DCreateHandle.c ${ARTK}/AR/vFree.c ${ARTK}/AR/arGetTransMatStereo.c ${ARTK}/AR/mDup.c ${ARTK}/AR/mPCA.c
  ${ARTK}/AR/paramGetPerspective.c ${ARTK}/AR/vDisp.c ${ARTK}/AR/mDet.c ${ARTK}/AR/mAllocUnit.c ${ARTK}/AR/mTrans.c
  ${ARTK}/AR/arLabelingSub/arLabelingSubDBZ.c ${ARTK}/AR/arLabelingSub/arLabelingSubEWRCY.c
  ${ARTK}/AR/arLabelingSub/arLabelingSubEWI3CA.c ${ARTK}/AR/arLabelingSub/arLabelingSubEBI3CA.c
  ${ARTK}/AR/arLabelingSub/arLabelingSubDBIYC.c ${ARTK}/AR/arLabelingSub/arLabelingSubEWR3C565.c
  ${ARTK}/AR/arLabelingSub/arLabelingSubEBI3CA5551.c ${ARTK}/AR/arLabelingSub/arLabelingSubEWIA3C.c
  ${ARTK}/AR/arLabelingSub/arLabelingSubDWIA3C.c ${ARTK}/AR/arLabelingSub/arLabelingSubDBRYC.c
  ${ARTK}/AR/arLabelingSub/arLabelingSubDWI3C.c ${ARTK}/AR/arLabelingSub/arLabelingSubDBR3CA.c
  ${ARTK}/AR/arLabelingSub/arLabelingSubDWICY.c ${ARTK}/AR/arLabelingSub/arLabelingSubEWRA3C.c
  ${ARTK}/AR/arLabelingSub/arLabelingSubEBR3CA4444.c ${ARTK}/AR/arLabelingSub/arLabelingSubEWR3CA5551.c
  ${ARTK}/AR/arLabelingSub/arLabelingSubDWR3C565.c ${ARTK}/AR/arLabelingSub/arLabelingSubDWRCY.c
  ${ARTK}/AR/arLabelingSub/arLabelingSubDBIA3C.c ${ARTK}/AR/arLabelingSub/arLabelingSubEBI3C565.c
  ${ARTK}/AR/arLabelingSub/arLabelingSubEWI3C.c ${ARTK}/AR/arLabelingSub/arLabelingSubEWR3C.c
  ${ARTK}/AR/arLabelingSub/arLabelingSubDBI3C565.c ${ARTK}/AR/arLabelingSub/arLabelingSubDWI3C565.c
  ${ARTK}/AR/arLabelingSub/arLabelingSubEWI3CA5551.c ${ARTK}/AR/arLabelingSub/arLabelingSubDBI3CA.c
  ${ARTK}/AR/arLabelingSub/arLabelingSubDBR3C565.c ${ARTK}/AR/arLabelingSub/arLabelingSubDBRA3C.c
  ${ARTK}/AR/arLabelingSub/arLabelingSubDWRC.c ${ARTK}/AR/arLabelingSub/arLabelingSubEWRYC.c
  ${ARTK}/AR/arLabelingSub/arLabelingSubEBR3C565.c ${ARTK}/AR/arLabelingSub/arLabelingSubEBICY.c
  ${ARTK}/AR/arLabelingSub/arLabelingSubDWI3CA4444.c ${ARTK}/AR/arLabelingSub/arLabelingSubEBI3C.c
  ${ARTK}/AR/arLabelingSub/arLabelingSubDBRC.c ${ARTK}/AR/arLabelingSub/arLabelingSubDBI3CA5551.c
  ${ARTK}/AR/arLabelingSub/arLabelingSubEWIYC.c ${ARTK}/AR/arLabelingSub/arLabelingSubDWR3CA4444.c
  ${ARTK}/AR/arLabelingSub/arLabelingSubEBZ.c ${ARTK}/AR/arLabelingSub/arLabelingSubDWZ.c
  ${ARTK}/AR/arLabelingSub/arLabelingSubDBR3CA5551.c ${ARTK}/AR/arLabelingSub/arLabelingSubDWR3CA5551.c
  ${ARTK}/AR/arLabelingSub/arLabelingSubDWIC.c ${ARTK}/AR/arLabelingSub/arLabelingSubEWR3CA.c
  ${ARTK}/AR/arLabelingSub/arLabelingSubEBRC.c ${ARTK}/AR/arLabelingSub/arLabelingSubDBRCY.c
  ${ARTK}/AR/arLabelingSub/arLabelingSubDWIYC.c ${ARTK}/AR/arLabelingSub/arLabelingSubEWRC.c
  ${ARTK}/AR/arLabelingSub/arLabelingSubEBIA3C.c ${ARTK}/AR/arLabelingSub/arLabelingSubEBR3CA.c
  ${ARTK}/AR/arLabelingSub/arLabelingSubDWRA3C.c ${ARTK}/AR/arLabelingSub/arLabelingSubEWI3C565.c
  ${ARTK}/AR/arLabelingSub/arLabelingSubDBR3CA4444.c ${ARTK}/AR/arLabelingSub/arLabelingSubEBR3C.c
  ${ARTK}/AR/arLabelingSub/arLabelingSubEWZ.c ${ARTK}/AR/arLabelingSub/arLabelingSubDBI3CA4444.c
  ${ARTK}/AR/arLabelingSub/arLabelingSubEWR3CA4444.c ${ARTK}/AR/arLabelingSub/arLabelingSubEWICY.c
  ${ARTK}/AR/arLabelingSub/arLabelingSubEBR3CA5551.c ${ARTK}/AR/arLabelingSub/arLabelingSubEBI3CA4444.c
  ${ARTK}/AR/arLabelingSub/arLabelingSubDBIC.c ${ARTK}/AR/arLabelingSub/arLabelingSubEBIYC.c
  ${ARTK}/AR/arLabelingSub/arLabelingSubEBRYC.c ${ARTK}/AR/arLabelingSub/arLabelingSubDWR3C.c
  ${ARTK}/AR/arLabelingSub/arLabelingSubDWR3CA.c ${ARTK}/AR/arLabelingSub/arLabelingSubDWI3CA5551.c
  ${ARTK}/AR/arLabelingSub/arLabelingSubEWI3CA4444.c ${ARTK}/AR/arLabelingSub/arLabelingSubDWRYC.c
  ${ARTK}/AR/arLabelingSub/arLabelingSubEBRA3C.c ${ARTK}/AR/arLabelingSub/arLabelingSubEWIC.c
  ${ARTK}/AR/arLabelingSub/arLabelingSubDBR3C.c ${ARTK}/AR/arLabelingSub/arLabelingSubDBI3C.c
  ${ARTK}/AR/arLabelingSub/arLabelingSubDBICY.c ${ARTK}/AR/arLabelingSub/arLabelingSubEBIC.c
  ${ARTK}/AR/arLabelingSub/arLabelingSubEBRCY.c ${ARTK}/AR/arLabelingSub/arLabelingSubDWI3CA.c ${ARTK}/AR/mFree.c
  ${ARTK}/AR/arPattSave.c ${ARTK}/AR/arLabeling.c ${ARTK}/AR/paramLT.c ${ARTK}/AR/arPattCreateHandle.c
  ${ARTK}/AR/arFilterTransMat.c ${ARTK}/AR/arGetMarkerInfo.c ${ARTK}/AR/arCreateHandle.c ${ARTK}/AR/arPattGetID.c
  ${ARTK}/AR/arGetLine.c ${ARTK}/AR/mInv.c ${ARTK}/AR/paramFile.c ${ARTK}/AR/arDetectMarker.c ${ARTK}/Util/thread_sub.c
  ${ARTK}/Util/profile.c ${ARTK}/ARICP/icpStereoPointRobust.c ${ARTK}/ARICP/icpStereoPoint.c ${ARTK}/ARICP/icpPoint.c
  ${ARTK}/ARICP/icpStereoHandle.c ${ARTK}/ARICP/icpCalibStereo.c ${ARTK}/ARICP/icpUtil.c ${ARTK}/ARICP/icpHandle.c
  ${ARTK}/ARICP/icpCore.c ${ARTK}/ARICP/icpPointRobust.c)

include_directories(Contrib/ARToolKit5arm/include)

########################################################################################################################
# Tensorflow-lite - it is now in jevois/Contrib. libtensorflowlite.so gets installed to the system, but we need to point
# to the includes, which do not get installed:

include_directories(../jevois/Contrib/tensorflow)
include_directories(../jevois/Contrib/tensorflow/tensorflow/lite/tools/make/downloads/flatbuffers/include)

target_link_libraries(${JEVOIS}base tensorflowlite)
if (JEVOIS_PRO)

else()
  if (JEVOIS_PLATFORM)
    # Need to specify path for libtensorflowlite.so; not sure why it is not found in the target buildroot tree:
    target_link_libraries(${JEVOIS}base -L${JEVOIS_PLATFORM_MODULES_ROOT}/lib)
  endif()
endif()

########################################################################################################################
# ovxlib: locate it on jevois-platform (on others, it is in /usr/lib):
if (JEVOIS_PRO)
else()
  if (JEVOIS_PLATFORM)
    include_directories(${JEVOIS_PLATFORM_INSTALL_PREFIX}/include/ovxlib)
  endif()
endif()

########################################################################################################################
# Link to OpenCV libraries for superpixels, aruco, and others:
# From this: ls /usr/share/jevois-opencv-4.*/lib/|grep ".so$"|sed -e s/.so//|sed -e s/^lib//|xargs
message(STATUS "opencv libs: ${JEVOIS_OPENCV_LIBS}")

target_link_libraries(${JEVOIS}base ${JEVOIS_OPENCV_LIBS} opencv_alphamat opencv_aruco opencv_barcode opencv_bgsegm
opencv_bioinspired opencv_calib3d opencv_ccalib opencv_core opencv_datasets opencv_dnn_objdetect opencv_dnn
opencv_dnn_superres opencv_dpm opencv_face opencv_features2d opencv_flann opencv_fuzzy opencv_gapi opencv_hfs
opencv_imgcodecs opencv_img_hash opencv_imgproc opencv_intensity_transform opencv_line_descriptor opencv_mcc opencv_ml
opencv_objdetect opencv_optflow opencv_phase_unwrapping opencv_photo opencv_plot opencv_quality opencv_rapid opencv_reg
opencv_saliency opencv_shape opencv_stereo opencv_stitching opencv_structured_light opencv_superres
opencv_surface_matching opencv_text opencv_tracking opencv_videoio opencv_video opencv_videostab opencv_wechat_qrcode
opencv_xfeatures2d opencv_ximgproc opencv_xobjdetect opencv_xphoto)

# removed: opencv_freetype (not available on platform), opencv_highgui (not needed), opencv_rgbd (not needed)

########################################################################################################################
# Install shared resources (cascade classifiers, neural network weights, etc):

install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/share-${JEVOIS}/"
  DESTINATION "${JEVOIS_MODULES_ROOT}/share" COMPONENT bin)

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
  set(DOC_PATH "share/doc/${JEVOIS}base-platform")
else (JEVOIS_PLATFORM)
  set(DOC_PATH "share/doc/${JEVOIS}base-host")
endif (JEVOIS_PLATFORM)
install(FILES ${DOC_FILES} DESTINATION ${DOC_PATH} COMPONENT bin)

########################################################################################################################
# Debian packaging:

# Create packages (Debian, RPM): in hbuild/ or pbuild/, just type 'sudo cpack' to create the package.
# To list the files created in a package, run: dpkg --contents <package.deb>
set(CPACK_PACKAGE_DESCRIPTION "JeVois Smart Machine Vision Base Modules (${JEVOIS})")
set(CPACK_PACKAGE_DESCRIPTION_SUMMARY "JeVois Smart Embedded Machine Vision Toolkit, Base Modules (${JEVOIS})")
set(CPACK_PACKAGE_CONTACT "Laurent Itti <jevois.org@gmail.com>")
set(CPACK_DEBIAN_PACKAGE_MAINTAINER ${CPACK_PACKAGE_CONTACT})
set(CPACK_DEBIAN_PACKAGE_SECTION "universe")
set(CPACK_PACKAGE_VENDOR "iLab at the University of Southern California")
set(CPACK_PACKAGE_VERSION_MAJOR "${JEVOIS_VERSION_MAJOR}")  # Note: jevoisbase version tracks jevois version
set(CPACK_PACKAGE_VERSION_MINOR "${JEVOIS_VERSION_MINOR}")
set(CPACK_PACKAGE_VERSION_PATCH "${JEVOIS_VERSION_PATCH}")
set(JEVOIS_PACKAGE_RELEASE "1") # packager revision number

if (JEVOIS_PLATFORM)
  set(JEVOIS_DEPEND "${JEVOIS}-platform (>=${JEVOIS_VERSION_MAJOR}.${JEVOIS_VERSION_MINOR}.${JEVOIS_VERSION_PATCH})")
else (JEVOIS_PLATFORM)
  set(JEVOIS_DEPEND "${JEVOIS}-host (>=${JEVOIS_VERSION_MAJOR}.${JEVOIS_VERSION_MINOR}.${JEVOIS_VERSION_PATCH})")
endif (JEVOIS_PLATFORM)

set(CPACK_DEBIAN_PACKAGE_DEPENDS "${JEVOIS_DEPEND}, libgles2-mesa, libgles2-mesa-dev, curl, libdmtx-dev, python3-scipy, libusb-dev")

# Use helper from JeVois.cmake for all other settings:
jevois_setup_cpack(${JEVOIS}base)


########################################################################################################################
# Wrapping up:
jevois_project_finalize()  # call this last in your CMakeLists.txt, will un-mount the live microSD if applicable
