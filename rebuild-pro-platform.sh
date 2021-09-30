#!/bin/bash
# USAGE: rebuild-pro-platform.sh [cmake opts]

# Note: on jevois-pro, we only ever build for staging:
extra="-DJEVOIS_MODULES_TO_STAGING=ON"

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
    && make -j \
    && sudo make install

