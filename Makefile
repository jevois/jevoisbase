# Copyright (C) 2016 by JeVois Inc. -- All rights Reserved.

# This makefile can build both for the JeVois hardware platform (cross-compiling for the embedded ARM processor, JeVois
# camera and USB driver), and for the host computer (usually an Intel-based desktop with X11 display and any compatible
# webcam). Building on the host computer is useful to develop and test new machine vision algorithms without having to
# copy the compiled code and data to SD card each time and to test it on the JeVois hardware. But beware that some
# differences may exist (due, e.g., to the different processor architectures, speeds, memory, etc).
#
# Switching between platform and host modes is via the JEVOIS_PLATFORM variable:
#   make JEVOIS_PLATFORM=1 install # build for the platform, installs into buildroot, to be copied to SD card
#   make install                   # build for the host, assumes that directory /jevois exists and is writeable
#
# Remember to 'make clean' in one mode each time before you switch to the other mode!
.DEFAULT_GOAL := all

JEVOIS_VENDOR := JeVois

# Location of our own source files:
SOURCES := $(shell find src/Components -name '*.[Cc]') $(shell find src/Image -name '*.[Cc]')
MODBASE := src/Modules
APPSOURCES := 

# Add .. to the include path so we can #include <jevoisbase/X/Y>
# also add jevoisbase/Contrib/XXX so we can include the contribs
EXTRAINCLUDES := -I.. -IContrib

########################################################################################################################
# tiny-cnn support:
EXTRAINCLUDES += -IContrib/tiny-cnn

########################################################################################################################
# ZZBar barcode / QR-code source files:

EXTRAINCLUDES += -IContrib/ZBar/include -IContrib/ZBar/zbar

SOURCES += Contrib/ZBar/zbar/processor.c \
Contrib/ZBar/zbar/scanner.c \
Contrib/ZBar/zbar/symbol.c \
Contrib/ZBar/zbar/img_scanner.c \
Contrib/ZBar/zbar/qrcode/rs.c \
Contrib/ZBar/zbar/qrcode/isaac.c \
Contrib/ZBar/zbar/qrcode/util.c \
Contrib/ZBar/zbar/qrcode/qrdectxt.c \
Contrib/ZBar/zbar/qrcode/bch15_5.c \
Contrib/ZBar/zbar/qrcode/binarize.c \
Contrib/ZBar/zbar/qrcode/qrdec.c \
Contrib/ZBar/zbar/config.c \
Contrib/ZBar/zbar/error.c \
Contrib/ZBar/zbar/processor/posix.c \
Contrib/ZBar/zbar/processor/lock.c \
Contrib/ZBar/zbar/processor/null.c \
Contrib/ZBar/zbar/convert.c \
Contrib/ZBar/zbar/decoder/i25.c \
Contrib/ZBar/zbar/decoder/qr_finder.c \
Contrib/ZBar/zbar/decoder/code128.c \
Contrib/ZBar/zbar/decoder/codabar.c \
Contrib/ZBar/zbar/decoder/code39.c \
Contrib/ZBar/zbar/decoder/databar.c \
Contrib/ZBar/zbar/decoder/ean.c \
Contrib/ZBar/zbar/decoder/code93.c \
Contrib/ZBar/zbar/image.c \
Contrib/ZBar/zbar/refcnt.c \
Contrib/ZBar/zbar/decoder.c \

#Contrib/ZBar/zbar/decoder/pdf417.c \

# FIXME need to debug zbar
EXTRACFLAGS += -Wparentheses -w

########################################################################################################################
# cvEyeTracker eye-tracking

SOURCES += Contrib/cvEyeTracker-1.2.5/ransac_ellipse.cpp \
Contrib/cvEyeTracker-1.2.5/remove_corneal_reflection.cpp \
Contrib/cvEyeTracker-1.2.5/svd.c \

########################################################################################################################
# Neon-accelerated NE10 support:
EXTRAINCLUDES += -IContrib/Ne10/inc

SOURCES += Contrib/Ne10/modules/imgproc/NE10_boxfilter.c
ifdef JEVOIS_PLATFORM
SOURCES += Contrib/Ne10/modules/imgproc/NE10_boxfilter.neon.c
endif

########################################################################################################################
# VLfeat support:
EXTRAINCLUDES += -IContrib/vlfeat/vl

# Add VLfeat sources shared among various algorithms:
SOURCES += Contrib/vlfeat/vl/host.c Contrib/vlfeat/vl/generic.c Contrib/vlfeat/vl/imopv.c

# The source code for SSE2 convolution seems to be missing...
EXTRACFLAGS += -DVL_DISABLE_SSE2

# Other defs to make VLfeat comile:
EXTRACFLAGS += -DVL_COMPILER_GNUC -DVL_ARCH_LITTLE_ENDIAN 

# Add VLfeat sources used by DenseSift module:
SOURCES += Contrib/vlfeat/vl/dsift.c Contrib/vlfeat/vl/sift.c

########################################################################################################################
# OpenCV superpixels, aruco, and others in the ximgproc module:

EXTRALIBS := -lopencv_ximgproc -lopencv_aruco -lopencv_calib3d

########################################################################################################################

# Include standard definitions and rules. This creates a bunch of variables (shown under showconfig) and targets
ifndef JEVOIS_ROOT
$(warning You should set the JEVOIS_ROOT environment variable to the root of jevois. Assuming $(HOME)/jevois)
JEVOIS_ROOT := $(HOME)/jevois
endif

# Configure for host or platform:
include $(JEVOIS_ROOT)/Makefile.inc

# Get all the extra includes and flags into our compiler flags:
CFLAGS += $(EXTRAINCLUDES) $(EXTRACFLAGS)
CXXFLAGS += $(EXTRAINCLUDES) $(EXTRACXXFLAGS)

# Link against libjevoisbase.so when building our module .so files:
MODLINK += -ljevoisbase

all: lib/libjevoisbase.so $(MODOBJECTS) $(APPEXE)

lib/libjevoisbase.so: $(OBJECTS)
	$(CXX) $(CXXFLAGS) $(LDFLAGS) -shared -o $@ -fPIC $^ $(OPENCV_LIBS) $(EXTRALIBS) -ltbb -ldl

$(MODOBJECTS): lib/libjevoisbase.so

# vendorinstall is executed as part of "make install"
# add anything you need to install here beyond all your modules and apps:
vendorinstall:
ifdef JEVOIS_PLATFORM
	/bin/cp lib/libjevoisbase.so $(INSTALL_ROOT)/usr/lib/
endif

# vendorclean is executed as part of "make clean"
# add anything you need to cleanup here beyond all your modules and apps:
vendorclean:
	-/bin/rm lib/libjevoisbase.so 2>/dev/null || true
ifdef JEVOIS_PLATFORM
	-/bin/rm $(INSTALL_ROOT)/usr/lib/libjevoisbase.so 2>/dev/null || true
endif
