#!/bin/bash
# usage: reinstall.sh [-y]
# will nuke and re-install all contributed packages

# Bump this release number each time you make significant changes here, this will cause rebuild-host.sh to re-run
# this reinstall script:
release=`cat RELEASE`

# Get the current jevoisbase version from CMakeLists.txt
# CAUTION: This is very brittle and requires this exact format in jevoisbase/CMakeLists.txt:
#   set(JEVOISBASE_SOVERSION "1.16.0")
sdir="$( dirname "${BASH_SOURCE[0]}" )"
jvbver=`grep JEVOISBASE_SOVERSION "${sdir}/../CMakeLists.txt" | head -1 | awk -F '"' '{ print $2 }'`

###################################################################################################
function get_github # owner, repo, revision
{
    echo "### JeVois: downloading ${1} / ${2} ..."
    git clone --recursive "https://github.com/${1}/${2}.git"
    if [ "X${3}" != "X" ]; then
        echo "### JeVois: moving ${1} / ${2} to checkout ${3} ..."
        cd "${2}"
        git checkout -q ${3}
        cd ..
    fi
}

###################################################################################################
function patchit # directory
{
    if [ ! -d ${1} ]; then
	    echo "Ooops cannot patch ${1} because directory is missing";
    else
        echo "### JeVois: patching ${1} ..."
	    cd ${1}
	    patch -p1 < ../${1}.patch
	    cd ..
    fi
}

###################################################################################################
if [ "x$1" = "x-y" ]; then
    REPLY="y";
else			   
    read -p "Do you want to nuke, fetch and patch contributed packages [y/N]? "
fi


if [ "X$REPLY" = "Xy" ]; then
    ###################################################################################################
    # Cleanup:
    /bin/rm -rf generalized-hough-tranform Ne10 NNPACK OF_DIS pixy pthreadpool tiny-dnn vlfeat ZBar NNPACK-darknet \
	    FXdiv FP16 psimd darknet-nnpack cpuinfo darknet

    ###################################################################################################
    # Get the packages:

    # Accelerator for convnets, used by tiny-dnn and darnket:
    get_github Maratyszcza NNPACK af40ea7d12702f8ae55aeb13701c09cad09334c3

    # No new release in a while on tiny-dnn; fetch current state as of Sept 14, 2017:
    get_github tiny-dnn tiny-dnn dd906fed8c8aff8dc837657c42f9d55f8b793b0e
    
    # Barcode/QRcode detection:
    get_github ZBar ZBar 854a5d97059e395807091ac4d80c53f7968abb8f
    cp zbar-config.h ZBar/include/config.h

    # CMUcam5 (pixy) color blob tracker:
    #get_github charmedlabs pixy 6ddb42dc8c15383e0bfc6951c7e1d9e1adf7fb35

    # ARM Neon (SIMD) open source project:
    get_github projectNe10 Ne10 f793b5e067737760798bd4928933ad6b87a09488

    # VLfeat computer vision algorithms:
    get_github vlfeat vlfeat 2320d40db0d9ab5a985832fad367b8a65033594b

    # Generalized Hough Transform in C++:
    #get_github jguillon generalized-hough-tranform dd2958657cc2cb8417f436e44da0b7eb36b63486

    # Fast optical flow:
    get_github tikroeger OF_DIS 2c9f2a674f3128d3a41c10e41cc9f3a35bb1b523
    /bin/rm -rf OF_DIS/.git # we will install the whole tree, so skip git files
    
    # openEyes eye tracking: this code is small and a bit old so it needs some patching, but it's not on github so we
    # just added it to our codebase and we track the changes through our master svn:
    #wget http://thirtysixthspan.com/openEyes/cvEyeTracker-1.2.5.tar.gz
    #tar zxvf cvEyeTracker-1.2.5.tar.gz
    #/bin/rm cvEyeTracker-1.2.5.tar.gz

    # NNPACK-accelerated darknet CNNs:
    get_github digitalbrain79 darknet-nnpack 614071ee5f5d066a5ef9ee29deb653fcf0903134

    # pthread-based thread pool for C/C++:
    get_github Maratyszcza pthreadpool 16bd2290da7673199dec90823bbc5063264e4095

    #  C99/C++ header-only library for division via fixed-point multiplication by inverse:
    get_github Maratyszcza FXdiv 811b482bcd9e8d98ad80c6c78d5302bb830184b0

    # Conversion to/from half-precision floating point formats:
    get_github Maratyszcza FP16 4b37bd31c9cc1380ef9f205f7dd031efe0e847ab

    # Portable 128-bit SIMD intrinsics:
    get_github Maratyszcza psimd 3d8bfe7318423462a6d9e0c6537e75efd4822c49

    # NNPACK depends on cpuinfo from pytorch:
    get_github pytorch cpuinfo 8c621ce3f46e51ac1d1a4d878b9ffc2b5dcac0e3
    
    # Darknet original (used for training only):
    git clone https://github.com/pjreddie/darknet.git

    # libcoral for C++ access to Coral Edge TPU
    #get_github google-coral libcoral release-frogfish

    # aml_NPU_app examples of neural network inference on the JeVois-Pro A311D
    #git clone --recursive https://gitlab.com/khadas/aml_npu_app

    ###################################################################################################
    # Patching:
    for f in *.patch; do
	    patchit ${f/.patch/}
    done

    ###################################################################################################
    # Keep track of the last installed release:
    echo $release > .installed
fi

cd ..
###################################################################################################
if [ "x$1" = "x-y" ]; then
    REPLY="y";
else			   
    read -p "Do you want to fetch and unpack contributed data for JeVois-A33 [y/N]? "
fi


if [ "X$REPLY" = "Xy" ]; then
    wget http://jevois.org/data/contrib-data-jevois-${jvbver}.tbz
    tar jxvf contrib-data-jevois-${jvbver}.tbz
    /bin/rm contrib-data-jevois-${jvbver}.tbz
fi

###################################################################################################
if [ "x$1" = "x-y" ]; then
    REPLY="y";
else			   
    read -p "Do you want to fetch and unpack contributed data for JeVois-Pro [y/N]? "
fi


if [ "X$REPLY" = "Xy" ]; then
    wget http://jevois.org/data/contrib-data-jevoispro-${jvbver}.tbz
    tar jxvf contrib-data-jevoispro-${jvbver}.tbz
    /bin/rm contrib-data-jevoispro-${jvbver}.tbz
fi
