#
#  Makefile
#  ARToolKit5
#
#  This file is part of ARToolKit.
#
#  ARToolKit is free software: you can redistribute it and/or modify
#  it under the terms of the GNU Lesser General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  ARToolKit is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU Lesser General Public License for more details.
#
#  You should have received a copy of the GNU Lesser General Public License
#  along with ARToolKit.  If not, see <http://www.gnu.org/licenses/>.
#
#  As a special exception, the copyright holders of this library give you
#  permission to link this library with independent modules to produce an
#  executable, regardless of the license terms of these independent modules, and to
#  copy and distribute the resulting executable under terms of your choice,
#  provided that you also meet, for each linked independent module, the terms and
#  conditions of the license of that module. An independent module is a module
#  which is neither derived from nor based on this library. If you modify this
#  library, you may extend this exception to your version of the library, but you
#  are not obligated to do so. If you do not wish to do so, delete this exception
#  statement from your version.
#
#  Copyright 2015 Daqri, LLC.
#  Copyright 2002-2015 ARToolworks, Inc.
#
#  Author(s): Hirokazu Kato, Philip Lamb
#

UNAME = $(shell uname)

AR_HOME = ../../..
AR_CPPFLAGS = -I$(AR_HOME)/include

OSG_CPPFLAGS = -I$(OSG_ROOT)/include
ifeq "$(UNAME)" "Darwin"
	OSG_CPPFLAGS = 
endif

CC=gcc
CXX=g++ -std=c++11
CPPFLAGS = $(AR_CPPFLAGS) $(OSG_CPPFLAGS)
CFLAGS = -O3 -march=pentium4 -I/usr/include/x86_64-linux-gnu
CXXFLAGS = -O3 -march=pentium4 -I/usr/include/x86_64-linux-gnu
AR=ar
ARFLAGS=rs
RANLIB = true

TARGET = $(AR_HOME)/lib/libARosg.a

HEADERS = \
    $(AR_HOME)/include/AR/arosg.h

OBJS = \
	arosg.o \

# Implicit rule, to compile C++ files with the .cpp suffix (rule already exists for .cc).
%.o : %.cpp
	$(CXX) -c $(CPPFLAGS) $(CXXFLAGS) $< -o $@

default build all: $(TARGET)

$(OBJS): $(HEADERS)

$(TARGET): $(OBJS)
	$(AR) $(ARFLAGS) $@ $^
	$(RANLIB) $@ 

clean:
	-rm -f $(OBJS)
	-rm -f $(TARGET)

allclean:
	-rm -f $(OBJS)
	-rm -f $(TARGET)
	-rm -f Makefile

distclean:
	rm -f $(OBJS)
	rm -f Makefile

