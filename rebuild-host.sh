#!/bin/sh

# Get the external contributed packages if they are not here or are outdated:
./Contrib/check.sh

# Build everything:
sudo /bin/rm -rf hbuild \
    && mkdir hbuild \
    && cd hbuild \
    && cmake "$@" .. \
    && make -j \
    && sudo make install
