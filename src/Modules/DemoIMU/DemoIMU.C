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

#include <jevois/Image/RawImageOps.H>
#include <linux/videodev2.h>
#include <list>

// icon by icons8

static jevois::ParameterCategory const ParamCateg("DemoIMU Options");

//! Parameter \relates DemoIMU
JEVOIS_DECLARE_PARAMETER(afac, float, "Factor applied to acceleration values for display",
                         0.01F, ParamCateg);

//! Parameter \relates DemoIMU
JEVOIS_DECLARE_PARAMETER(gfac, float, "Factor applied to gyroscope values for display",
                         0.01F, ParamCateg);

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
    //! Default base class constructor ok
    using jevois::Module::Module;

    //! Virtual destructor for safe inheritance
    virtual ~DemoIMU() { }

    //! Get a 16-bit value from the IMU
    inline short imuget(unsigned char hreg)
    {
      short val = short(readIMUregister(hreg) & 0xff) << 8;
      val |= readIMUregister(hreg + 1) & 0xff;
      return val;
    }
    
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
      IMUdata d; float const a = afac::get(); float const g = gfac::get();
      short rawx, rawy, rawz;
      
      rawx = imuget(45); rawy = imuget(47); rawz = imuget(49);
      jevois::rawimage::writeText(outimg, jevois::sformat("Accel: x=%6d y=%6d z=%6d", rawx, rawy, rawz),
                                  3, h + 3, jevois::yuyv::White);
      d.ax = rawx * a; d.ay = rawy * a; d.az = rawz * a;


      rawx = imuget(51); rawy = imuget(53); rawz = imuget(55);
      jevois::rawimage::writeText(outimg, jevois::sformat("Gyro:  x=%6d y=%6d z=%6d", rawx, rawy, rawz),
                                  3, h + 15, jevois::yuyv::White);
      d.gx = rawx * g; d.gy = rawy * g; d.gz = rawz * g;

      itsIMUdata.push_front(d);
      while (itsIMUdata.size() > w/2) itsIMUdata.pop_back();

      // Plot the IMU data:
      float const hh = h * 0.5F; int const sz = itsIMUdata.size(); int x = w - 1; IMUdata const * pd = nullptr;
      for (IMUdata const & dd : itsIMUdata)
      {
        if (pd)
        {
          if (a)
          {
            jevois::rawimage::drawLine(outimg, x + 2, pd->ax + hh, x, dd.ax + hh, 1, jevois::yuyv::LightGreen);
            jevois::rawimage::drawLine(outimg, x + 2, pd->ay + hh, x, dd.ay + hh, 1, jevois::yuyv::LightPink);
            jevois::rawimage::drawLine(outimg, x + 2, pd->az + hh, x, dd.az + hh, 1, jevois::yuyv::White);
          }
          
          if (g)
          {
            jevois::rawimage::drawLine(outimg, x + 2, pd->gx + hh, x, dd.gx + hh, 1, jevois::yuyv::DarkGreen);
            jevois::rawimage::drawLine(outimg, x + 2, pd->gy + hh, x, dd.gy + hh, 1, jevois::yuyv::DarkPink);
            jevois::rawimage::drawLine(outimg, x + 2, pd->gz + hh, x, dd.gz + hh, 1, jevois::yuyv::DarkGrey);
          }
          
        }
        pd = &dd;
        x -= 2;
      }
      
      // Send the output image with our processing results to the host over USB:
      outframe.send();
    }

  private:
    struct IMUdata
    {
        float ax, ay, az;
        float gx, gy, gz;
        float mx, my, mz;
    };
    
    std::list<IMUdata> itsIMUdata;    
};

// Allow the module to be loaded as a shared object (.so) file:
JEVOIS_REGISTER_MODULE(DemoIMU);
