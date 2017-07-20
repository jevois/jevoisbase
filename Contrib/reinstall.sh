#!/bin/bash
# usage: reinstall.sh [-y]
# will nuke and re-install all contributed packages

if [ "x$1" = "x-y" ]; then
    REPLY="y";
else			   
    read -p "Do you want to nuke, fetch and patch contributed packages [y/N]? "
fi

if [ "X$REPLY" = "Xy" ]; then
    /bin/rm -rf generalized-hough-tranform Ne10 NNPACK OF_DIS pixy pthreadpool tiny-dnn vlfeat ZBar

    git clone --recursive https://github.com/Maratyszcza/NNPACK.git # Accelerator for convnets
    git clone https://github.com/Maratyszcza/pthreadpool.git

    # we get some release version of tiny-dnn as the master branch is under quite active development
    # with frequent API changes:
    #git clone https://github.com/tiny-dnn/tiny-dnn.git  # Convolutional neural networks
    wget https://github.com/tiny-dnn/tiny-dnn/archive/v1.0.0a3.tar.gz
    tar zxvf v1.0.0a3.tar.gz 
    /bin/rm v1.0.0a3.tar.gz
    mv tiny-dnn-1.0.0a3 tiny-dnn

    git clone https://github.com/ZBar/ZBar.git          # Barcode/QRcode detection
    cp zbar-config.h ZBar/include/config.h
    
    git clone https://github.com/charmedlabs/pixy.git   # CMUcam5 (pixy) color blog tracker
    git clone https://github.com/projectNe10/Ne10.git   # ARM Neon (SIMD) open source project
    git clone https://github.com/vlfeat/vlfeat.git      # VLfeat computer vision algorithms
    git clone https://github.com/jguillon/generalized-hough-tranform.git # Generalized Hough Transform in C++
    git clone https://github.com/tikroeger/OF_DIS.git   # Fast optical flow
    
    # openEyes eye tracking: this code is small and a bit old so it needs some patching, bit it's not on github so we
    # just added it to our codebase and track the changes through our master svn:
    #wget http://thirtysixthspan.com/openEyes/cvEyeTracker-1.2.5.tar.gz
    #tar zxvf cvEyeTracker-1.2.5.tar.gz
    #/bin/rm cvEyeTracker-1.2.5.tar.gz 

    cd OF_DIS && patch -p1 < ../OF_DIS.patch && cd ..
fi
