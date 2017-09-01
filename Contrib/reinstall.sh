#!/bin/bash
# usage: reinstall.sh [-y]
# will nuke and re-install all contributed packages

# Bump this release number each time you make significant changes here, this will cause rebuild-host.sh to re-run
# this reinstall script:
release=`cat RELEASE`

function get_github # owner, repo, revision
{
    git clone --recursive "https://github.com/${1}/${2}.git"
    if [ "X${3}" != "X" ]; then cd "${2}" ; git checkout -q ${3}; cd .. ; fi
}

if [ "x$1" = "x-y" ]; then
    REPLY="y";
else			   
    read -p "Do you want to nuke, fetch and patch contributed packages [y/N]? "
fi

if [ "X$REPLY" = "Xy" ]; then
    ###################################################################################################
    # Cleanup:
    /bin/rm -rf generalized-hough-tranform Ne10 NNPACK OF_DIS pixy pthreadpool tiny-dnn vlfeat ZBar NNPACK-darknet \
	    FXdiv FP16 psimd darknet-nnpack darknet

    ###################################################################################################
    # Get the packages:

    #git clone --recursive https://github.com/Maratyszcza/NNPACK.git # Accelerator for convnets

    # we get some release version of tiny-dnn as the master branch is under quite active development
    # with frequent API changes:
    wget https://github.com/tiny-dnn/tiny-dnn/archive/v1.0.0a3.tar.gz
    tar zxvf v1.0.0a3.tar.gz 
    /bin/rm v1.0.0a3.tar.gz
    mv tiny-dnn-1.0.0a3 tiny-dnn

    #git clone https://github.com/tiny-dnn/tiny-dnn.git  # Convolutional neural networks
    # To avoid surprises, checkout a specific version of tiny-dnn (since it has been a while since the last release):
    #cd tiny-dnn; git checkout dd906fed8c8aff8dc837657c42f9d55f8b793b0e; cd ..

    get_github ZBar ZBar 854a5d97059e395807091ac4d80c53f7968abb8f # Barcode/QRcode detection
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

    # NNPACK acceleration for darknet CNNs. NOTE: We also update it to a more recent upstream:
    get_github thomaspark-pkj NNPACK-darknet 1ecda1044d314893796b2d1c4c71d6aeda84baa2
    cd NNPACK-darknet && \
	git pull --no-edit https://github.com/Maratyszcza/NNPACK.git master && \
	git checkout -q 3627062907e01ba5f030730f1027dd773323e0e3 && \
	cd ..
    
    # NNPACK-accelerated darknet CNNs:
    get_github thomaspark-pkj darknet-nnpack fa5bddcfca788c69defdf1fcaafb31a2b6d685a7

    # pthread-based thread pool for C/C++:
    get_github Maratyszcza pthreadpool 097a0c8971176257d7d565c4d37b754a12b3566b

    #  C99/C++ header-only library for division via fixed-point multiplication by inverse:
    get_github Maratyszcza FXdiv f7960924cb1f67e06f6529dba09eeda9baa4cacb

    # Conversion to/from half-precision floating point formats:
    get_github Maratyszcza FP16 251d93e3b2776f0a97b572f1d5cae10958adf00e

    # Portable 128-bit SIMD intrinsics:
    get_github Maratyszcza psimd c583161bf2097508b168fceb2f383a9d5ebde449

    # Darknet original (used for training only):
    git clone https://github.com/pjreddie/darknet.git
    
    ###################################################################################################
    # Patching:
    cd OF_DIS && patch -p1 < ../OF_DIS.patch && cd ..

    ###################################################################################################
    # Keep track of the last installed release:
    echo $release > .installed
fi


