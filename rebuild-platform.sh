#!/bin/sh

# you can specify -DJEVOIS_MODULES_TO_BUILDROOT=ON to install directly to buildroot, thereby bypassing the need to build
# a .jvpkg package that will then be unpacked by the smart camera

# On ARM hosts like Raspberry Pi3, we will likely run out of memory if attempting more than 1 compilation thread:
ncpu=`cat /proc/cpuinfo |grep processor|wc -l`
if [ `cat /proc/cpuinfo | grep ARM | wc -l` -gt 0 ]; then ncpu=1; fi

/bin/rm -rf pbuild && mkdir pbuild && cd pbuild && cmake "$@" -DJEVOIS_PLATFORM=ON .. && make -j ${ncpu} && make install


# avoid clogging up the SD card with tiny-cnn training data:
/bin/rm -f ${HOME}/jevois-sdk/out/sun8iw5p1/linux/common/buildroot/target/jevois/modules/*/JeVois/*/tiny-dnn/*/*.bin

/bin/rm -f ${HOME}/jevois-sdk/out/sun8iw5p1/linux/common/buildroot/target/jevois/modules/*/JeVois/*/tiny-dnn/*/*ubyte
