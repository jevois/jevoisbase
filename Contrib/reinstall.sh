#!/bin/bash

read -p "Do you want to nuke, fetch and patch all contributed packages [y/N]? "
if [ "X$REPLY" = "Xy" ]; then
    /bin/rm -rf `/bin/ls -p1 | /bin/grep /`

    git clone --recursive https://github.com/Maratyszcza/NNPACK.git # Accelerator for convnets
    git clone https://github.com/Maratyszcza/pthreadpool.git
    git clone https://github.com/tiny-dnn/tiny-dnn.git  # Convolutional neural networks

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
    git pull # get it back as we just nuked it above...

    cd OF_DIS && patch -p1 < ../OF_DIS.patch && cd ..
    cd tiny-dnn && patch -p1 < ../tiny-dnn.patch && cd ..

fi
