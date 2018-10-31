// ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//
// JeVois Smart Embedded Machine Vision Toolkit - Copyright (C) 2016 by Laurent Itti, the University of Southern
// California (USC), and iLab at USC. See http://iLab.usc.edu and http://jevois.org for information about this project.
//
// This file is part of the JeVois Smart Embedded Machine Vision Toolkit.  This program is free software; you can
// redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software
// Foundation, version 2.  This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY;
// without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public
// License for more details.  You should have received a copy of the GNU General Public License along with this program;
// if not, write to the Free Software Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301, USA.
//
// Contact information: Laurent Itti - 3641 Watt Way, HNB-07A - Los Angeles, CA 90089-2520 - USA.
// Tel: +1 213 740 3527 - itti@pollux.usc.edu - http://iLab.usc.edu - http://jevois.org
// ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/*! \file */

#include <jevois/Core/Module.H>
#include <jevois/Core/ICM20948.H>

#include <jevois/Image/RawImageOps.H>
#include <linux/videodev2.h>
#include <list>

// icon by icons8

static jevois::ParameterCategory const ParamCateg("DemoIMU Options");

//! Parameter \relates DemoIMU
JEVOIS_DECLARE_PARAMETER(afac, float, "Factor applied to acceleration values for display",
                         100.0F, ParamCateg);

//! Parameter \relates DemoIMU
JEVOIS_DECLARE_PARAMETER(gfac, float, "Factor applied to gyroscope values for display",
                         1.0F, ParamCateg);

//! Plot raw IMU readings on top of video
/*! As an optional hardware upgrade, one can install a global shutter sensor into JeVois (an OnSemi AR0135 1.3MP), which
    also includes an inertial measurement unit (IMU). The IMU is a 9-degrees-of-freedom (9DOF) TDK InvenSense ICM-20948
    (with 3-axis accelerometer, 3-axis gyroscope, and 3-axis magnetometer). It also includes a digital motion processing
    unit, which allows it to compute and filter Euler angles or quaternions directly inside the IMU chip.

    This module only works with optional JeVois sensors that include an IMU!

    In this module, we directly access raw registers of the IMU chip. In future versions, some of this low-level work
    will be abstracted into a higher-level IMU driver for JeVois.

    @author Laurent Itti

    @videomapping YUYV 640 512 40.0 YUYV 640 480 40.0 JeVois DemoIMU
    @email itti\@usc.edu
    @address University of Southern California, HNB-07A, 3641 Watt Way, Los Angeles, CA 90089-2520, USA
    @copyright Copyright (C) 2018 by Laurent Itti, iLab and the University of Southern California
    @mainurl http://jevois.org
    @supporturl http://jevois.org/doc
    @otherurl http://iLab.usc.edu
    @license GPL v3
    @distribution Unrestricted
    @restrictions None
    \ingroup modules */
class DemoIMU : public jevois::Module, public jevois::Parameter<afac, gfac>
{
  public:
    //! Constructor
    DemoIMU(std::string const & instance) : jevois::Module(instance)
    {
      itsIMU = addSubComponent<jevois::ICM20948>("imu");
    }

    //! Virtual destructor for safe inheritance
    virtual ~DemoIMU() { }
    
    //! Processing function
    virtual void process(jevois::InputFrame && inframe, jevois::OutputFrame && outframe) override
    {
      // Wait for next available camera image:
      jevois::RawImage const inimg = inframe.get(true);
      int const w = inimg.width, h = inimg.height;
      
      // Wait for an image from our gadget driver into which we will put our results:
      jevois::RawImage outimg = outframe.get();

      // Enforce that the input and output formats and image sizes match:
      outimg.require("output", w, h + 32, inimg.fmt);
      
      // Just copy the pixel data over:
      memcpy(outimg.pixelsw<void>(), inimg.pixels<void>(), std::min(inimg.buf->length(), outimg.buf->length()));
      jevois::rawimage::drawFilledRect(outimg, 0, h, w, outimg.height-h, jevois::yuyv::Black);

      // Let camera know we are done processing the input image:
      inframe.done();

      // Get one IMU reading:
      jevois::IMUdata d = itsIMU->get();
      jevois::rawimage::writeText(outimg, jevois::sformat("Accel: x=%+02.2fg y=%+02.2fg z=%+02.2fg",
                                                          d.ax(), d.ay(), d.az()), 3, h + 3, jevois::yuyv::White);

      jevois::rawimage::writeText(outimg, jevois::sformat("Gyro:  x=%+04.1fdps y=%+04.1fdps z=%+04.1fdps",
                                                          d.gx(), d.gy(), d.gz()), 3, h + 15, jevois::yuyv::White);

      itsIMUdata.push_front(d);
      while (itsIMUdata.size() > w/2) itsIMUdata.pop_back();

      // Plot the IMU data:
      float const hh = h * 0.5F; int const sz = itsIMUdata.size(); int x = w - 1;
      float const a = afac::get(); float const g = gfac::get(); jevois::IMUdata const * pd = nullptr;

      for (jevois::IMUdata const & dd : itsIMUdata)
      {
        if (pd)
        {
          if (a)
          {
            jevois::rawimage::drawLine(outimg, x + 2, pd->ax()*a + hh, x, dd.ax()*a + hh, 1, jevois::yuyv::LightGreen);
            jevois::rawimage::drawLine(outimg, x + 2, pd->ay()*a + hh, x, dd.ay()*a + hh, 1, jevois::yuyv::LightPink);
            jevois::rawimage::drawLine(outimg, x + 2, pd->az()*a + hh, x, dd.az()*a + hh, 1, jevois::yuyv::White);
          }
          
          if (g)
          {
            jevois::rawimage::drawLine(outimg, x + 2, pd->gx()*g + hh, x, dd.gx()*g + hh, 1, jevois::yuyv::DarkGreen);
            jevois::rawimage::drawLine(outimg, x + 2, pd->gy()*g + hh, x, dd.gy()*g + hh, 1, jevois::yuyv::DarkPink);
            jevois::rawimage::drawLine(outimg, x + 2, pd->gz()*g + hh, x, dd.gz()*g + hh, 1, jevois::yuyv::DarkGrey);
          }
          
        }
        pd = &dd;
        x -= 2;
      }
      
      // Send the output image with our processing results to the host over USB:
      outframe.send();
    }

  private:
    std::shared_ptr<jevois::ICM20948> itsIMU;
    std::list<jevois::IMUdata> itsIMUdata;    
};

// Allow the module to be loaded as a shared object (.so) file:
JEVOIS_REGISTER_MODULE(DemoIMU);
