#!/bin/bash
# USAGE: rebuild-platform.sh [--staging|--microsd] [cmake opts]

# OPTIONS:
# [none] - install compiled modules to jvpkg/ for later packing by make jvpkg
# --staging - install to microSD staging area /var/lib/jevois-microsd/
# --microsd - install directly to live microSD card which should be mounted under /media/username/JEVOIS

extra=""
if [ "X$1" = "X--staging" ]; then extra="-DJEVOIS_MODULES_TO_STAGING=ON"; shift; fi
if [ "X$1" = "X--microsd" ]; then extra="-DJEVOIS_MODULES_TO_MICROSD=ON"; shift; fi

# On ARM hosts like Raspberry Pi3, we will likely run out of memory if attempting more than 1 compilation thread:
ncpu=`cat /proc/cpuinfo |grep processor|wc -l`
if [ `cat /proc/cpuinfo | grep ARM | wc -l` -gt 0 ]; then ncpu=1; fi

sudo /bin/rm -rf pbuild \
    && mkdir pbuild \
    && cd pbuild \
    && cmake "${extra} $@" -DJEVOIS_PLATFORM=ON .. \
    && make -j ${ncpu} \
    && sudo make install

