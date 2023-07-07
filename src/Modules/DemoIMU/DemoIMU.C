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
#include <jevois/Debug/Timer.H>

#include <jevois/Image/RawImageOps.H>
#include <linux/videodev2.h>
#include <list>

// icon by icons8

static jevois::ParameterCategory const ParamCateg("DemoIMU Options");

//! Parameter \relates DemoIMU
JEVOIS_DECLARE_PARAMETER(afac, float, "Factor applied to acceleration values for display, or 0 to not display",
                         100.0F, ParamCateg);

//! Parameter \relates DemoIMU
JEVOIS_DECLARE_PARAMETER(gfac, float, "Factor applied to gyroscope values for display, or 0 to not display",
                         0.5F, ParamCateg);

//! Parameter \relates DemoIMU
JEVOIS_DECLARE_PARAMETER(mfac, float, "Factor applied to magnetometer values for display, or 0 to not display",
                         3.0F, ParamCateg);

//! Plot raw IMU readings on top of video
/*! As an optional hardware upgrade, one can install a global shutter sensor into JeVois (an OnSemi AR0135 1.2MP), which
    also includes on its custom circuit board for JeVois an inertial measurement unit (IMU). The IMU is a
    9-degrees-of-freedom (9DOF) TDK InvenSense ICM-20948 (with 3-axis accelerometer, 3-axis gyroscope, and 3-axis
    magnetometer). This IMU also includes a digital motion processing unit (small programmable processor inside the IMU
    chip), which allows it to compute and filter Euler angles or quaternions directly inside the IMU chip.

    This module only works with optional JeVois sensors that include an IMU! The base JeVois-A33 smart camera does not
    have an onboard IMU.
    
    \youtube{MFGpN_Vp7mg}

    This module demonstrates the RAW and FIFO modes of the IMU.

    The specifications of this chip are quite impressive:
    - 3-axis 16-bit accelerometer with full-range sensitivity selectable to +/-2g, +/-4g, +/-8g, and +/-16g.
    - Accelerometer data rate from 4 Hz to 1125 Hz.
    - 3-axis 16-bit gyroscope with full-range sensitivity selectable to +/-250dps (degrees/s), +/-500dps,
      +/-1000dps, and +/-2000dps.
    - Gyroscope data rate from 4 Hz to 1125 Hz.
    - 3-axis 16-bit magnetometer (compass) with wide range of +/-4900uT (micro Tesla).
    - Magnetometer data rates 10 Hz, 20 Hz, 50 Hz, or 100 Hz.
    - 16-bit temperature sensor with readout rate of up to 8 kHz.
    - RAW data mode (get current sensor values at any time), buffered (FIFO) data mode (sensor values accumulate into
      a FIFO at a fixed rate), and digital motion processing mode (DMP; raw data is processed on-chip).
    - On-chip digital motion processor (DMP) can compute, inside the IMU chip itself:
      + quaternion 6 (uses accel + gyro),
      + quaternion 9 (uses accel + gyro + compass),
      + geomag quaternion (uses accel + compass),
      + flip/pickup detection,
      + step detection and counting,
      + basic activity recognition: drive, walk, run, bike, tilt, still.

    With quaternions computed on-chip, with an algorithm that gets sensor data at a highly accurate, fixed rate, and
    applies various calibrations, drift corrections, and compensations on the fly, one gets highly accurate real-time
    estimate of the sensor's pose in the 3D world and of how it is moving.

    Note that communication with the IMU is over a 400kHz I2C bus, which may limit data readout rate depending on
    which data is requested from the IMU.

    This IMU has 3 basic modes of operation (parameter mode, which can only be set in params.cfg):

    - RAW: One can access the latest raw sensor data at any time using the getRaw() or get() functions. This is the
      simplest mode of operation. One disadvantage is that if you are not calling get() at a perfectly regular
      interval, there will be some time jitter in your readouts. The IMU does not provide any time stamps for its
      data.

    - FIFO: In this mode, data from the sensor is piled up into a 1 kbyte FIFO buffer at a precise, constant rate
      (when all three of accelerometer, gyroscope, and magnetometer are on, the gyro rate determines the FIFO
      buffering rate). Main advantage is that you can then read out the data without having to worry about calling
      getRaw() or get() at a highly precise interval. But you need to be careful that the FIFO can fill up and
      overflow very quickly when using high sensor data rates.

    - DMP: In this mode, data is captured from the sensor at an accurate, fixed rate, and is fed to the on-chip
      digital motion processor (DMP). The DMP then computes quaternions, activity recognition, etc and pushes data
      packets into the FIFO as results from these algorithms become available.

    @author Laurent Itti

    @videomapping YUYV 640 512 30.0 YUYV 640 480 30.0 JeVois DemoIMU
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
class DemoIMU : public jevois::Module, public jevois::Parameter<afac, gfac, mfac>
{
  public:
    //! Constructor
    DemoIMU(std::string const & instance) : jevois::Module(instance)
    { itsIMU = addSubComponent<jevois::ICM20948>("imu"); }

    //! Virtual destructor for safe inheritance
    virtual ~DemoIMU() { }
    
    //! Processing function
    virtual void process(jevois::InputFrame && inframe, jevois::OutputFrame && outframe) override
    {
      // Wait for next available camera image:
      jevois::RawImage const inimg = inframe.get(true);
      unsigned int const w = inimg.width, h = inimg.height;
      
      // Wait for an image from our gadget driver into which we will put our results:
      jevois::RawImage outimg = outframe.get();

      // Enforce that the input and output formats and image sizes match:
      outimg.require("output", w, h + 32, inimg.fmt);
      
      // Just copy the pixel data over:
      memcpy(outimg.pixelsw<void>(), inimg.pixels<void>(), std::min(inimg.buf->length(), outimg.buf->length()));
      jevois::rawimage::drawFilledRect(outimg, 0, h, w, outimg.height-h, jevois::yuyv::Black);

      // Let camera know we are done processing the input image:
      inframe.done();

      // Get one or more IMU readings:
      jevois::IMUdata d = itsIMU->get();
      jevois::rawimage::writeText(outimg,
         jevois::sformat("Accel: x=%+06.2fg    y=%+06.2fg    z=%+06.2fg         Magn: %+09.2fuT %+09.2fuT %+09.2fuT",
                         d.ax(), d.ay(), d.az(), d.mx(), d.my(), d.mz()),
                         3, h + 3, jevois::yuyv::White);

      jevois::rawimage::writeText(outimg,
         jevois::sformat("Gyro:  x=%+07.1fdps y=%+07.1fdps z=%+07.1fdps      Temp: %05.1fC  %s",
                         d.gx(), d.gy(), d.gz(), d.temp(), d.magovf ? "Magn overflow" : " "),
                         3, h + 15, jevois::yuyv::White);

      itsIMUdata.push_front(d);

      // In FIFO mode at high data rates, we may have more samples in the FIFO; read a bunch:
      while (itsIMU->dataReady() > 32) itsIMUdata.push_front(itsIMU->get());

      // Only keep as much data as we can display:
      while (itsIMUdata.size() > w/2) itsIMUdata.pop_back();

      // Plot the IMU data:
      float const hh = h * 0.5F; int x = w - 1;
      // Plot so that positive values go up (so, negate all values):
      float const a = -afac::get(); float const g = -gfac::get(); float const m = -mfac::get();
      jevois::IMUdata const * pd = nullptr;

      for (jevois::IMUdata const & dd : itsIMUdata)
      {
        if (pd)
        {
          if (a)
          {
            jevois::rawimage::drawLine(outimg, x + 2, pd->ax()*a + hh, x, dd.ax()*a + hh, 1, jevois::yuyv::DarkGreen);
            jevois::rawimage::drawLine(outimg, x + 2, pd->ay()*a + hh, x, dd.ay()*a + hh, 1, jevois::yuyv::MedGreen);
            jevois::rawimage::drawLine(outimg, x + 2, pd->az()*a + hh, x, dd.az()*a + hh, 1, jevois::yuyv::LightGreen);
          }
          
          if (g)
          {
            jevois::rawimage::drawLine(outimg, x + 2, pd->gx()*g + hh, x, dd.gx()*g + hh, 1, jevois::yuyv::DarkPink);
            jevois::rawimage::drawLine(outimg, x + 2, pd->gy()*g + hh, x, dd.gy()*g + hh, 1, jevois::yuyv::MedPink);
            jevois::rawimage::drawLine(outimg, x + 2, pd->gz()*g + hh, x, dd.gz()*g + hh, 1, jevois::yuyv::LightPink);
          }

          if (m)
          {
            jevois::rawimage::drawLine(outimg, x + 2, pd->mx()*m + hh, x, dd.mx()*m + hh, 1, jevois::yuyv::DarkTeal);
            jevois::rawimage::drawLine(outimg, x + 2, pd->my()*m + hh, x, dd.my()*m + hh, 1, jevois::yuyv::MedTeal);
            jevois::rawimage::drawLine(outimg, x + 2, pd->mz()*m + hh, x, dd.mz()*m + hh, 1, jevois::yuyv::LightTeal);
          }
         
        }
        pd = &dd;
        x -= 2;
      }
      
      // Send the output image with our processing results to the host over USB:
      outframe.send();
    }

#ifdef JEVOIS_PRO
    //! Processing function with GUI output
    virtual void process(jevois::InputFrame && inframe, jevois::GUIhelper & helper) override
    {
      static jevois::Timer timer("processing", 100, LOG_DEBUG);

      // Start the GUI frame:
      unsigned short winw, winh;
      helper.startFrame(winw, winh);

      // Draw the camera frame:
      int x = 0, y = 0; unsigned short iw = 0, ih = 0;
      helper.drawInputFrame("camera", inframe, x, y, iw, ih);
      helper.itext("JeVois-Pro Inertial Measurement Unit (IMU)");
      
      // Let camera know we are done processing the input image:
      inframe.done();

      timer.start();

      // Get one or more IMU readings:
      jevois::IMUdata d = itsIMU->get();
      helper.itext(jevois::sformat("Accel: x=%+06.2fg    y=%+06.2fg    z=%+06.2fg         "
                                   "Magn: %+09.2fuT %+09.2fuT %+09.2fuT",
                                   d.ax(), d.ay(), d.az(), d.mx(), d.my(), d.mz()));

      helper.itext(jevois::sformat("Gyro:  x=%+07.1fdps y=%+07.1fdps z=%+07.1fdps      Temp: %05.1fC  %s",
                                   d.gx(), d.gy(), d.gz(), d.temp(), d.magovf ? "Magn overflow" : " "));

      itsIMUdata.push_front(d);

      // In FIFO mode at high data rates, we may have more samples in the FIFO; read a bunch:
      while (itsIMU->dataReady() > 32) itsIMUdata.push_front(itsIMU->get());

      // Only keep as much data as we can display:
      int const count = 300;
      while (itsIMUdata.size() > count) itsIMUdata.pop_back();
      
      float ax[count] = {}, ay[count] = {}, az[count] = {};
      float gx[count] = {}, gy[count] = {}, gz[count] = {};
      float mx[count] = {}, my[count] = {}, mz[count] = {};
      
      // Plot so that positive values go up (so, negate all values):
      float const a = -afac::get(); float const g = -gfac::get(); float const m = -mfac::get();

      for (int i = count - 1; jevois::IMUdata const & dd : itsIMUdata)
      {
        if (a) { ax[i] = dd.ax() * a; ay[i] = dd.ay() * a; az[i] = dd.az() * a; }
        if (g) { gx[i] = dd.gx() * g; gy[i] = dd.gy() * g; gz[i] = dd.gz() * g; }
        if (m) { mx[i] = dd.mx() * m; my[i] = dd.my() * m; mz[i] = dd.mz() * m; }
        --i;
      }

      // Draw the data:
      if (ImGui::Begin("IMU data"))
      {
        ImGui::PushItemWidth(ImGui::GetContentRegionAvail().x - 10.0F);
        ImGui::Text("Acceleration X, Y, Z:");
        ImGui::PlotLines("##AccelX", ax, count, 0, "Accel X", -100, 100);
        ImGui::PlotLines("##AccelY", ay, count, 0, "Accel Y", -100, 100);
        ImGui::PlotLines("##AccelZ", az, count, 0, "Accel Z", -100, 100);
        ImGui::Separator();
        ImGui::Text("Gyroscope X, Y, Z:");
        ImGui::PlotLines("##GyroX", gx, count, 0, "Gyro X", -100, 100);
        ImGui::PlotLines("##GyroY", gy, count, 0, "Gyro Y", -100, 100);
        ImGui::PlotLines("##GyroZ", gz, count, 0, "Gyro Z", -100, 100);
        ImGui::Separator();
        ImGui::Text("Magnetometer X, Y, Z:");
        ImGui::PlotLines("##MagnX", mx, count, 0, "Magn X", -100, 100);
        ImGui::PlotLines("##MagnY", my, count, 0, "Magn Y", -100, 100);
        ImGui::PlotLines("##MagnZ", mz, count, 0, "Magn Z", -100, 100);
        ImGui::PopItemWidth();
      }
      ImGui::End();
        
      // Show processing fps:
      std::string const & fpscpu = timer.stop();
      helper.iinfo(inframe, fpscpu, winw, winh);

      // Render the image and GUI:
      helper.endFrame();
    }
#endif
    
  private:
    std::shared_ptr<jevois::ICM20948> itsIMU;
    std::list<jevois::IMUdata> itsIMUdata;    
};

// Allow the module to be loaded as a shared object (.so) file:
JEVOIS_REGISTER_MODULE(DemoIMU);
