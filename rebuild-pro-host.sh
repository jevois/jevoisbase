#!/bin/sh

# On JeVoisPro, limit the number of compile threads to not run out of memory:
ncpu=`grep -c processor /proc/cpuinfo`
if [ `grep -c JeVois /proc/cpuinfo` -gt 0 ]; then ncpu=4; fi

# Get the external contributed packages if they are not here or are outdated:
./Contrib/check.sh

# Build everything:
sudo /bin/rm -rf phbuild \
    && mkdir phbuild \
    && cd phbuild \
    && cmake "$@" -DJEVOIS_HARDWARE=PRO .. \
    && make -j ${ncpu} \
    && sudo make install
