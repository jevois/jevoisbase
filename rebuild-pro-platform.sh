#!/bin/bash
# USAGE: rebuild-pro-platform.sh [--staging|--microsd|--live] [cmake opts]

# OPTIONS:
# [none]    - install compiled modules to jvpkg/ for later packing by make jvpkg
# --staging - install to microSD staging area /var/lib/jevois-microsd/
# --microsd - install directly to microSD card inserted into host computer
# --live    - install directly to microSD card inside running JeVois connected to host computer

extra=""
if [ "X$1" = "X--staging" ]; then extra="-DJEVOIS_MODULES_TO_STAGING=ON"; shift; fi
if [ "X$1" = "X--microsd" ]; then extra="-DJEVOIS_MODULES_TO_MICROSD=ON"; shift; fi
if [ "X$1" = "X--live" ]; then extra="-DJEVOIS_MODULES_TO_LIVE=ON"; shift; fi

# On JeVoisPro, limit the number of compile threads to not run out of memory:
ncpu=`grep -c processor /proc/cpuinfo`
if [ `grep -c JeVois /proc/cpuinfo` -gt 0 ]; then ncpu=4; fi

# Get the external contributed packages if they are not here or are outdated:
./Contrib/check.sh

# For jevoisbase only: we do include the staging /usr/include very early in our CFLAGS so we can get the jevois config
# and other jevois includes. But this means that preference will be given to the staged jevoisbase includes as well over
# those in the current source tree. So here, nuke the staged jevoisbase includes so we will use the source tree:
sudo rm -rf /var/lib/jevoispro-build/usr/include/jevoisbase
sudo rm -f /var/lib/jevoispro-build/usr/lib/libjevoisprobase*

# Let's build it:
sudo /bin/rm -rf ppbuild \
    && mkdir ppbuild \
    && cd ppbuild \
    && cmake "${extra} $@" -DJEVOIS_HARDWARE=PRO -DJEVOIS_PLATFORM=ON .. \
    && make -j ${ncpu} \
    && sudo make install

