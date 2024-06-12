#!/bin/bash
# USAGE: rebuild-platform.sh [--jvpkg] [cmake opts]
#
# If --jvpkg is specified, then a jevois .jvpkg package will also be created

set -e

create_jvpkg="no"
if [ "X$1" = "X--jvpkg" ]; then create_jvpkg="yes"; shift; fi

# Get the external contributed packages if they are not here or are outdated:
./Contrib/check.sh

# For jevoisbase only: we do include the staging /usr/include very early in our CFLAGS so we can get the jevois config
# and other jevois includes. But this means that preference will be given to the staged jevoisbase includes as well over
# those in the current source tree. So here, nuke the staged jevoisbase includes so we will use the source tree:
sudo rm -rf /var/lib/jevois-build/usr/include/jevoisbase
sudo rm -f /var/lib/jevois-build/usr/lib/libjevoisbase*

# Let's build it:
sudo /bin/rm -rf pbuild
mkdir pbuild
cd pbuild
cmake "$@" -DJEVOIS_PLATFORM=ON ..
make -j
sudo make install
cd ..

if [ $create_jvpkg = "yes" ]; then jevois-jvpkg `cat pbuild/jvpkg-args`; fi
