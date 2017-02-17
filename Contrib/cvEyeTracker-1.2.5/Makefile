COMPILER=g++

LIBDIRS=-L/usr/local/lib -L/usr/X11R6/lib

LIBS=-lm -ldc1394_control -lraw1394 -lopencv -lcvaux -lhighgui

INCLUDES=-I/usr/local/include/opencv 

DEFINES= -O2

SOURCES=cvEyeTracker.c
OBJS = remove_corneal_reflection.o ransac_ellipse.o svd.o timing.o
HEADDERS = remove_corneal_reflection.h ransac_ellipse.h svd.h timing.h

all : cvEyeTracker

cvEyeTracker: cvEyeTracker.o $(OBJS)
	$(COMPILER) -pg -o cvEyeTracker cvEyeTracker.o $(OBJS) $(DEFINES) $(LIBDIRS) $(LIBS)

cvEyeTracker.o: cvEyeTracker.c $(HEADDERS)
	$(COMPILER) -c $(DEFINES) cvEyeTracker.c $(DEFINES) $(INCLUDES)

remove_corneal_reflection.o: remove_corneal_reflection.c $(HEADDERS)
	$(COMPILER) -c $(DEFINES) remove_corneal_reflection.c $(DEFINES) $(INCLUDES)

ransac_ellipse.o: ransac_ellipse.cpp $(HEADDERS)
	$(COMPILER) -c $(DEFINES) ransac_ellipse.cpp $(INCLUDES)

svd.o: svd.c $(HEADDERS)
	$(COMPILER) -c $(DEFINES) svd.c

timing.o: timing.c $(HEADDERS)
	$(COMPILER) -c $(DEFINES) timing.c

clean:
	rm -f *.o cvEyeTracker


