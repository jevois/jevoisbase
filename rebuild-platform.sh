#!/bin/bash
# USAGE: rebuild-platform.sh [--staging|--microsd /dev/sdX] [cmake opts]

# if -b is specified as first argument, install modules to buildroot for later packing to microSD by jevois-sdk
# otherwise, install to jvpkg/ in jevoisbase (for later make jvpkg)

extra=""
if [ "X$1" = "X--staging" ]; then extra="-DJEVOIS_MODULES_TO_STAGING=ON"; shift; fi
##TODO if [ "X$1" = "X--microsd" ]; then extra="-DJEVOIS_MODULES_TO_STAGING=ON"; shift; fi


# you can specify -DJEVOIS_MODULES_TO_BUILDROOT=ON to install directly to buildroot, thereby bypassing the need to build
# a .jvpkg package that will then be unpacked by the smart camera

# On ARM hosts like Raspberry Pi3, we will likely run out of memory if attempting more than 1 compilation thread:
ncpu=`cat /proc/cpuinfo |grep processor|wc -l`
if [ `cat /proc/cpuinfo | grep ARM | wc -l` -gt 0 ]; then ncpu=1; fi

/bin/rm -rf pbuild && mkdir pbuild && cd pbuild && cmake "${extra} $@" -DJEVOIS_PLATFORM=ON .. && make -j ${ncpu} && sudo make install

