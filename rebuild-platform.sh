#!/bin/bash
# USAGE: rebuild-platform.sh [--staging|--microsd|--live] [cmake opts]

# OPTIONS:
# [none]    - install compiled modules to jvpkg/ for later packing by make jvpkg
# --staging - install to microSD staging area /var/lib/jevois-microsd/
# --microsd - install directly to microSD card inserted into host computer
# --live    - install directly to microSD card inside running JeVois connected to host computer

extra=""
if [ "X$1" = "X--staging" ]; then extra="-DJEVOIS_MODULES_TO_STAGING=ON"; shift; fi
if [ "X$1" = "X--microsd" ]; then extra="-DJEVOIS_MODULES_TO_MICROSD=ON"; shift; fi
if [ "X$1" = "X--live" ]; then extra="-DJEVOIS_MODULES_TO_LIVE=ON"; shift; fi

# Get the external contributed packages if they are not here or are outdated:
./Contrib/check.sh

# For jevoisbase only: we do include the staging /usr/include very early in our CFLAGS so we can get the jevois config
# and other jevois includes. But this means that preference will be given to the staged jevoisbase includes as well over
# those in the current source tree. So here, nuke the staged jevoisbase includes so we will use the source tree:
sudo rm -rf /var/lib/jevois-build/usr/include/jevoisbase
sudo rm -f /var/lib/jevois-build/usr/lib/libjevoisbase*

# Let's build it:
sudo /bin/rm -rf pbuild \
    && mkdir pbuild \
    && cd pbuild \
    && cmake "${extra} $@" -DJEVOIS_PLATFORM=ON .. \
    && make -j \
    && sudo make install

