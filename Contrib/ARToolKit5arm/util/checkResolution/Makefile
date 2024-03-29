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

AR2_HOME= ../..
AR2_INC_DIR= $(AR2_HOME)/include
AR2_LIB_DIR= $(AR2_HOME)/lib

BIN_DIR= ../../bin

CC= gcc
CFLAG= -O3 -march=pentium4 -I/usr/include/x86_64-linux-gnu -I$(AR2_INC_DIR)
LDFLAG= -L/usr/lib/x86_64-linux-gnu -L/usr/lib64 -L$(AR2_LIB_DIR)/linux-x86_64 -L$(AR2_LIB_DIR)
LIBS= -lAR2 -lAR -lARICP -lAR -lglut -lGLU -lGL -lX11 -lm -lpthread -ljpeg


OBJS =
HEADDERS =

all: $(BIN_DIR)/checkResolution

$(BIN_DIR)/checkResolution: checkResolution.o $(OBJS)
	${CC} -o $(BIN_DIR)/checkResolution checkResolution.o $(OBJS) $(LDFLAG) $(LIBS)

checkResolution.o: checkResolution.c $(HEADDERS)
	${CC} -c $(CFLAG) checkResolution.c


clean:
	rm -f *.o
	rm -f $(BIN_DIR)/checkResolution

allclean:
	rm -f *.o
	rm -f $(BIN_DIR)/checkResolution
	rm -f Makefile

distclean:
	rm -f *.o
	rm -f Makefile
