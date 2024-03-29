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

INC_DIR= ../../include
LIB_DIR= ../../lib
BIN_DIR= ../../bin

CXX=g++ -std=c++11
LDFLAG=-L/usr/lib/x86_64-linux-gnu -L/usr/lib64 -L$(LIB_DIR)
LIBS= -lARgsub -lARvideo -lAR -lARICP -lAR -L/usr/local/lib -lopencv_stitching -lopencv_superres -lopencv_videostab -lopencv_aruco -lopencv_bgsegm -lopencv_bioinspired -lopencv_ccalib -lopencv_cvv -lopencv_dpm -lopencv_freetype -lopencv_fuzzy -lopencv_line_descriptor -lopencv_optflow -lopencv_reg -lopencv_saliency -lopencv_stereo -lopencv_structured_light -lopencv_phase_unwrapping -lopencv_rgbd -lopencv_surface_matching -lopencv_tracking -lopencv_datasets -lopencv_text -lopencv_face -lopencv_plot -lopencv_dnn -lopencv_xfeatures2d -lopencv_shape -lopencv_video -lopencv_ximgproc -lopencv_calib3d -lopencv_features2d -lopencv_flann -lopencv_xobjdetect -lopencv_objdetect -lopencv_ml -lopencv_xphoto -lopencv_highgui -lopencv_videoio -lopencv_imgcodecs -lopencv_photo -lopencv_imgproc -lopencv_core -lglut -lGLU -lGL -lX11 -lm -lpthread -ljpeg
CFLAG= -O3 -march=pentium4 -I/usr/include/x86_64-linux-gnu -I/usr/local/include/opencv -I/usr/local/include -I$(INC_DIR)


all: $(BIN_DIR)/calib_stereo

$(BIN_DIR)/calib_stereo: calib_stereo.o
	${CXX} -o $(BIN_DIR)/calib_stereo calib_stereo.o\
	   $(LDFLAG) $(LIBS)

calib_stereo.o: calib_stereo.cpp
	${CXX} -c $(CFLAG) calib_stereo.cpp

clean:
	rm -f calib_stereo.o
	rm -f $(BIN_DIR)/calib_stereo

allclean:
	rm -f calib_stereo.o
	rm -f $(BIN_DIR)/calib_stereo
	rm -f Makefile

distclean:
	rm -f *.o
	rm -f Makefile
