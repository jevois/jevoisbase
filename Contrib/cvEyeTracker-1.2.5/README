/*
 *
 * cvEyeTracker is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * cvEyeTracker is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with cvEyeTracker; if not, write to the Free Software
 * Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
 *
 *
 * cvEyeTracker - Version 1.2.5
 * Part of the openEyes ToolKit -- http://hcvl.hci.iastate.edu/openEyes
 * Release Date:
 * Authors : Dongheng Li <dhli@iastate.edu>
 *           Derrick Parkhurst <derrick.parkhurst@hcvl.hci.iastate.edu>
 *           Jason Babcock <babcock@nyu.edu>
 *           David Winfield <dwinfiel@iastate.edu>
 * Copyright (c) 2004-2006
 * All Rights Reserved.
 *
 */

--------------------------------------------------------------------------

DESCRIPTION:
--------------------------------------------------------------------------
cvEyeTracker is eyetracking software aimed at doing simple dark-pupil eye tracking
from a firewire camera that monitors the eye. The software also supports input
from a second camera to provide a frame of reference of where the eye is looking.

CONTENTS:
--------------------------------------------------------------------------
README                  This document
COPYING                 GPL license
cvEyeTracker.c          cvEyetracker source code

Building
--------------------------------------------------------------------------
Requires:
        Linux Kernel >= 2.6

        Install OpenCV-0.9.5 http://sourceforge.net/projects/opencvlibrary/

        Install libraw1394-0.10.1 http://www.linux1394.org/
                libdc1394-0.9.4   http://sourceforge.net/project/showfiles.php?group_id=8157&package_id=8268


1) Edit 'Makefile' to reflect your library directories
2) run 'make'


Usage:
--------------------------------------------------------------------------
Make sure that all the necessary modules that support firewire capture are
running.

to load the raw1394, dc1394 and video1394 run the script:
(as root) ./insertmodules

exit root

Make sure you have two cameras plugged in.
./cvEyeTracker

This should popup 5 windows and a slider-bar window.

Note that on monitors with resolutions of 1024 x 768,
some of the slider bars on the control window will not be
visible. This is a slight bug and will br fixed soon.



Debugging:
--------------------------------------------------------------------------
Report bugs to babcock@nyu.edu


Known Issues:
--------------------------------------------------------------------------
1) for screens with 1024x768 you cannot seen the bottom-most slider bars.

2) No error handling to check that there is two cameras

3) It takes a few seconds for the slider bars to load completly so be patient because the sliders take
   a few seconds to respond. I think this is an issue in OpenCV's higui interface.

 

