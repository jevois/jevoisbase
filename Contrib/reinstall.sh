#!/bin/bash
# usage: reinstall.sh [-y]
# will nuke and re-install all contributed packages

# Bump this release number each time you make significant changes here, this will cause rebuild-host.sh to re-run
# this reinstall script:
release=`cat RELEASE`

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
	    FXdiv FP16 psimd darknet-nnpack cpuinfo darknet tensorflow

    ###################################################################################################
    # Get the packages:

    # Accelerator for convnets, used by tiny-dnn and darnket:
    get_github Maratyszcza NNPACK af40ea7d12702f8ae55aeb13701c09cad09334c3

    # No new release in a while on tiny-dnn; fetch current state as of Sept 14, 2017:
    get_github tiny-dnn tiny-dnn dd906fed8c8aff8dc837657c42f9d55f8b793b0e

    #git clone https://github.com/tiny-dnn/tiny-dnn.git  # Convolutional neural networks
    # To avoid surprises, checkout a specific version of tiny-dnn (since it has been a while since the last release):
    #cd tiny-dnn; git checkout dd906fed8c8aff8dc837657c42f9d55f8b793b0e; cd ..

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
    
    # openEyes eye tracking: this code is small and a bit old so it needs some patching, but it's not on github so we
    # just added it to our codebase and we track the changes through our master svn:
    #wget http://thirtysixthspan.com/openEyes/cvEyeTracker-1.2.5.tar.gz
    #tar zxvf cvEyeTracker-1.2.5.tar.gz
    #/bin/rm cvEyeTracker-1.2.5.tar.gz

    # NNPACK-accelerated darknet CNNs:
    get_github digitalbrain79 darknet-nnpack 79df27c8006734412c72587df379ab3740938971

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

    # Tensorflow 1.9.9:
    get_github tensorflow tensorflow 25c197e02393bd44f50079945409009dd4d434f8
    cd tensorflow
    # oops, tensorflow does not compile anymore if it pulls the latest flatbuffers. Patch to pulling flatbuffers 1.8.0:
    sed -i sXflatbuffers/archive/master.zipXflatbuffers/archive/v1.8.0.zipX \
	./tensorflow/contrib/lite/download_dependencies.sh
    ./tensorflow/contrib/lite/download_dependencies.sh
    cd ..

    ###################################################################################################
    # Patching:
    for f in *.patch; do
	patchit ${f/.patch/}
    done
    
    ###################################################################################################
    # Keep track of the last installed release:
    echo $release > .installed
fi


###################################################################################################
if [ "x$1" = "x-y" ]; then
    REPLY="y";
else			   
    read -p "Do you want to fetch and unpack contributed data [y/N]? "
fi


if [ "X$REPLY" = "Xy" ]; then
    cd ../share
    wget http://jevois.org/data/contrib-data.tbz
    tar jxvf contrib-data.tbz
    /bin/rm contrib-data.tbz
fi
