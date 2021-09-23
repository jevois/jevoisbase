#!/bin/bash
# USAGE: rebuild-pro-platform-pdeb.sh [cmake opts]

# On JeVoisPro, limit the number of compile threads to not run out of memory:
ncpu=`grep -c processor /proc/cpuinfo`
if [ `grep -c JeVois /proc/cpuinfo` -gt 0 ]; then ncpu=4; fi

# Get the external contributed packages if they are not here or are outdated:
./Contrib/check.sh

# For jevoisbase only: we do include the staging /usr/include very early in our CFLAGS so we can get the jevois config
# and other jevois includes. But this means that preference will be given to the staged jevoisbase includes as well over
# those in the current source tree. So here, nuke the staged jevoisbase includes so we will use the source tree:
sudo rm -rf /var/lib/jevoispro-build-pdeb/usr/include/jevoisbase
sudo rm -f /var/lib/jevoispro-build-pdeb/usr/lib/libjevoisprobase*

# Let's build it:
sudo /bin/rm -rf ppdbuild \
    && mkdir ppdbuild \
    && cd ppdbuild \
    && cmake -DJEVOIS_HARDWARE=PRO -DJEVOIS_PLATFORM=ON -DJEVOIS_MODULES_TO_STAGING=ON -DJEVOISPRO_PLATFORM_DEB=ON $@ .. \
    && make -j ${ncpu} \
    && sudo make install \
    && sudo cpack

