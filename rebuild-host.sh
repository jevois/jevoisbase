#!/bin/sh

# On ARM hosts like Raspberry Pi3, we will likely run out of memory if attempting more than 1 compilation thread:
ncpu=`cat /proc/cpuinfo |grep processor|wc -l`
if [ `cat /proc/cpuinfo | grep ARM | wc -l` -gt 0 ]; then ncpu=1; fi

# Get the external contributed packages if they are not here:
if [ ! -d Contrib/ZBar ]; then
    echo "It looks like some Contrib packages are missing. Redirecting to Contrib/reinstall.sh ..."
    cd Contrib
    ./reinstall.sh
    cd ..
fi

/bin/rm -rf hbuild \
    && mkdir hbuild \
    && cd hbuild \
    && cmake "$@" .. \
    && make -j ${ncpu} \
    && sudo make install
