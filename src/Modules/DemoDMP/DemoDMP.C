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

#include <Eigen/Geometry>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/core/eigen.hpp>

// icon by icons8

static jevois::ParameterCategory const ParamCateg("DemoDMP Options");

//! Plot results of processing IMU data with the on-chip Digital Motion Processor (DMP)
/*! As an optional hardware upgrade, one can install a global shutter sensor into JeVois (an OnSemi AR0135 1.2MP), which
    also includes on its custom circuit board for JeVois an inertial measurement unit (IMU). The IMU is a
    9-degrees-of-freedom (9DOF) TDK InvenSense ICM-20948 (with 3-axis accelerometer, 3-axis gyroscope, and 3-axis
    magnetometer). This IMU also includes a digital motion processing unit (small programmable processor inside the IMU
    chip), which allows it to compute and filter quaternions directly inside the IMU chip.

    This module only works with optional JeVois sensors that include an IMU! The base JeVois-A33 smart camera does not
    have an onboard IMU.

    \youtube{MFGpN_Vp7mg}

    This module demonstrates the digital motion processing (DMP) mode of the IMU.

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

    @videomapping YUYV 640 520 25.0 YUYV 640 480 25.0 JeVois DemoDMP
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
class DemoDMP : public jevois::Module
{
  public:
    //! Constructor
    DemoDMP(std::string const & instance) : jevois::Module(instance)
    {
      itsIMU = addSubComponent<jevois::ICM20948>("imu");
    }

    //! Virtual destructor for safe inheritance
    virtual ~DemoDMP() { }
    
    //! Processing function
    virtual void process(jevois::InputFrame && inframe, jevois::OutputFrame && outframe) override
    {
      // Wait for next available camera image:
      jevois::RawImage const inimg = inframe.get(true);
      int const w = inimg.width, h = inimg.height;
      
      // Wait for an image from our gadget driver into which we will put our results:
      jevois::RawImage outimg = outframe.get();

      // Load camera calibration params if needed (for 3D cube displays):
      static cv::Mat itsCamMatrix = cv::Mat();
      static cv::Mat itsDistCoeffs = cv::Mat();
      if (itsCamMatrix.empty())
      {
        std::string const cpf = std::string(JEVOIS_SHARE_PATH) + "/camera/calibration" +
          std::to_string(w) + 'x' + std::to_string(h) + ".yaml";
        
        cv::FileStorage fs(cpf, cv::FileStorage::READ);
        if (fs.isOpened())
        {
          fs["camera_matrix"] >> itsCamMatrix;
          fs["distortion_coefficients"] >> itsDistCoeffs;
          LINFO("Loaded camera calibration from " << cpf);
        }
        else LERROR("Failed to read camera parameters from file [" << cpf << "] -- IGNORED");
      }
      
      // Enforce that the input and output formats and image sizes match:
      outimg.require("output", w, h + 40, inimg.fmt);
      
      // Just copy the pixel data over:
      memcpy(outimg.pixelsw<void>(), inimg.pixels<void>(), std::min(inimg.buf->length(), outimg.buf->length()));
      jevois::rawimage::drawFilledRect(outimg, 0, h, w, outimg.height-h, jevois::yuyv::Black);

      // Let camera know we are done processing the input image:
      inframe.done();

      // Note: we get different packets for steps, activity, and for quaternions. So let's cache the data:
      static long step_ts = 0;
      static long steps = 0;
      static std::vector<std::string> bac;
      static cv::Vec3d rvec(0.0, 0.0, 0.0); // Euler angles computed from quaternion
      static cv::Vec3d tvec(0.0, 0.0, 0.5); // Place cube 0.5 meters in front of the camera
      float const hsz = 0.075F; // half size of cube, in meters
      static int pickup = 0; // Number of frames remaining to show pickup message
      long quat[3] = { 0, 0, 0 }; // Our latest quaternion
      std::string qtype;
      jevois::IMUrawData rd;
      bool has_accel = false, has_gyro = false, has_cpass = false;
      float fsync = -1.0F;
      
      // Get as many IMU readings as we can so we keep the FIFO empty:
      jevois::DMPdata d; int npkt= 0;
      while (npkt == 0 || itsIMU->dataReady())
      {
        d = itsIMU->getDMP(); ++npkt;
        
        // ---------- Quaternions using 9 DOF:
        if (d.header1 & JEVOIS_DMP_QUAT9)
        {
          quat[0] = d.quat9[0]; quat[1] = d.quat9[1]; quat[2] = d.quat9[2];
          qtype = "Quat-9";
        }

        // ---------- Quaternions using 6 DOF:
        if (d.header1 & JEVOIS_DMP_QUAT6)
        {
          quat[0] = d.quat6[0]; quat[1] = d.quat6[1]; quat[2] = d.quat6[2];
          qtype = "Quat-6";
        }

        // ---------- Quaternions using GeoMag:
        if (d.header1 & JEVOIS_DMP_GEOMAG)
        {
          quat[0] = d.geomag[0]; quat[1] = d.geomag[1]; quat[2] = d.geomag[2];
          qtype = "GeoMag";
        }

        // ---------- Step detection:
        if (d.header1 & JEVOIS_DMP_PED_STEPDET) { step_ts = d.stepts; steps += d.steps; }

        // ---------- Activity recognition:
        if (d.header2 & JEVOIS_DMP_ACT_RECOG) bac = d.activity2();
      
        // ---------- Pickup/flip detection:
        if (d.header2 & JEVOIS_DMP_FLIP_PICKUP) pickup = 20;

        // ---------- FSYNC detection:
        if (d.header2 & JEVOIS_DMP_FSYNC) fsync = d.fsync_us();

        // ---------- Raw accel, gyro, mag:
        if (d.header1 & JEVOIS_DMP_ACCEL)
        { has_accel = true; rd.ax() = d.accel[0]; rd.ay() = d.accel[1]; rd.az() = d.accel[2]; }

        if (d.header1 & JEVOIS_DMP_GYRO)
        { has_gyro = true; rd.gx() = d.gyro[0]; rd.gy() = d.gyro[1]; rd.gz() = d.gyro[2]; }

        if (d.header1 & JEVOIS_DMP_CPASS)
        { has_cpass = true; rd.mx() = d.cpass[0]; rd.my() = d.cpass[1]; rd.mz() = d.cpass[2]; }
      }
      
      // ---------- Display a cube rotated by our latest quaternion:
      if (qtype.empty() == false)
      {
        Eigen::Quaterniond q(0.0, d.fix2float(quat[0]), d.fix2float(quat[1]), d.fix2float(quat[2]));

        // Compute the scalar part, which is not returned by the IMU, to get a normalized quaternion:
        q.w() = std::sqrt(1.0 - q.squaredNorm());
        
        // Display it:
        jevois::rawimage::writeText(outimg, qtype +
                                    jevois::sformat(": x=%+09.6f y=%+09.6f z=%+09.6f", q.x(), q.y(), q.z()),
                                    3, h + 3, jevois::yuyv::White);
        
        // Compute OpenCV rvec for use with projectPoints() below:
        // You could also compute Euler angles: Eigen::Vector3f eig_rvec = q.toRotationMatrix().eulerAngles(0, 1, 2);
        Eigen::Matrix3d eig_rmat = q.toRotationMatrix();
        cv::Mat rmat(3, 3, CV_64FC1); cv::eigen2cv(eig_rmat, rmat);
        cv::Rodrigues(rmat, rvec);
        
        // Project axis points:
        std::vector<cv::Point3f> axisPoints;
        axisPoints.push_back(cv::Point3f(0.0F, 0.0F, 0.0F));
        axisPoints.push_back(cv::Point3f(hsz * 1.5F, 0.0F, 0.0F));
        axisPoints.push_back(cv::Point3f(0.0F, hsz * 1.5F, 0.0F));
        axisPoints.push_back(cv::Point3f(0.0F, 0.0F, hsz * 1.5F));
        axisPoints.push_back(cv::Point3f(hsz * 1.55F, 0.0F, 0.0F));
        axisPoints.push_back(cv::Point3f(0.0F, hsz * 1.55F, 0.0F));
        axisPoints.push_back(cv::Point3f(0.0F, 0.0F, hsz * 1.55F));
        
        std::vector<cv::Point2f> imagePoints;
        cv::projectPoints(axisPoints, rvec, tvec, itsCamMatrix, itsDistCoeffs, imagePoints);
        
        // Draw axis lines
        jevois::rawimage::drawLine(outimg, int(imagePoints[0].x + 0.5F), int(imagePoints[0].y + 0.5F),
                                   int(imagePoints[1].x + 0.5F), int(imagePoints[1].y + 0.5F),
                                   2, jevois::yuyv::MedPurple);
        jevois::rawimage::writeText(outimg, "X", int(imagePoints[4].x - 4.0F), int(imagePoints[4].y - 9.5F),
                                    jevois::yuyv::White, jevois::rawimage::Font10x20);
        
        jevois::rawimage::drawLine(outimg, int(imagePoints[0].x + 0.5F), int(imagePoints[0].y + 0.5F),
                                   int(imagePoints[2].x + 0.5F), int(imagePoints[2].y + 0.5F),
                                   2, jevois::yuyv::MedGreen);
        jevois::rawimage::writeText(outimg, "Y", int(imagePoints[5].x - 4.0F), int(imagePoints[5].y - 9.5F),
                                    jevois::yuyv::White, jevois::rawimage::Font10x20);
        
        jevois::rawimage::drawLine(outimg, int(imagePoints[0].x + 0.5F), int(imagePoints[0].y + 0.5F),
                                   int(imagePoints[3].x + 0.5F), int(imagePoints[3].y + 0.5F),
                                   2, jevois::yuyv::MedGrey);
        jevois::rawimage::writeText(outimg, "Z", int(imagePoints[6].x - 4.0F), int(imagePoints[6].y - 9.5F),
                                    jevois::yuyv::White, jevois::rawimage::Font10x20);
        
        // Also draw a cube:
        std::vector<cv::Point3f> cubePoints;
        cubePoints.push_back(cv::Point3f(-hsz, -hsz, -hsz));
        cubePoints.push_back(cv::Point3f(hsz, -hsz, -hsz));
        cubePoints.push_back(cv::Point3f(hsz, hsz, -hsz));
        cubePoints.push_back(cv::Point3f(-hsz, hsz, -hsz));
        cubePoints.push_back(cv::Point3f(-hsz, -hsz, hsz));
        cubePoints.push_back(cv::Point3f(hsz, -hsz, hsz));
        cubePoints.push_back(cv::Point3f(hsz, hsz, hsz));
        cubePoints.push_back(cv::Point3f(-hsz, hsz, hsz));
        
        std::vector<cv::Point2f> cuf;
        cv::projectPoints(cubePoints, rvec, tvec, itsCamMatrix, itsDistCoeffs, cuf);
        
        // Round all the coordinates:
        std::vector<cv::Point> cu;
        for (auto const & p : cuf) cu.push_back(cv::Point(int(p.x + 0.5F), int(p.y + 0.5F)));
        
        // Draw cube lines:
        jevois::rawimage::drawLine(outimg, cu[0].x, cu[0].y, cu[1].x, cu[1].y, 2, jevois::yuyv::LightGreen);
        jevois::rawimage::drawLine(outimg, cu[1].x, cu[1].y, cu[2].x, cu[2].y, 2, jevois::yuyv::LightGreen);
        jevois::rawimage::drawLine(outimg, cu[2].x, cu[2].y, cu[3].x, cu[3].y, 2, jevois::yuyv::LightGreen);
        jevois::rawimage::drawLine(outimg, cu[3].x, cu[3].y, cu[0].x, cu[0].y, 2, jevois::yuyv::LightGreen);
        jevois::rawimage::drawLine(outimg, cu[4].x, cu[4].y, cu[5].x, cu[5].y, 2, jevois::yuyv::LightGreen);
        jevois::rawimage::drawLine(outimg, cu[5].x, cu[5].y, cu[6].x, cu[6].y, 2, jevois::yuyv::LightGreen);
        jevois::rawimage::drawLine(outimg, cu[6].x, cu[6].y, cu[7].x, cu[7].y, 2, jevois::yuyv::LightGreen);
        jevois::rawimage::drawLine(outimg, cu[7].x, cu[7].y, cu[4].x, cu[4].y, 2, jevois::yuyv::LightGreen);
        jevois::rawimage::drawLine(outimg, cu[0].x, cu[0].y, cu[4].x, cu[4].y, 2, jevois::yuyv::LightGreen);
        jevois::rawimage::drawLine(outimg, cu[1].x, cu[1].y, cu[5].x, cu[5].y, 2, jevois::yuyv::LightGreen);
        jevois::rawimage::drawLine(outimg, cu[2].x, cu[2].y, cu[6].x, cu[6].y, 2, jevois::yuyv::LightGreen);
        jevois::rawimage::drawLine(outimg, cu[3].x, cu[3].y, cu[7].x, cu[7].y, 2, jevois::yuyv::LightGreen);
      }
      
      // ---------- Step detection:
      jevois::rawimage::writeText(outimg, jevois::sformat("Steps: %6u (last at %010u)", steps, step_ts),
                                  3, h + 15, jevois::yuyv::White);

      // ---------- FSYNC detection:
      if (fsync >= 0.0F) jevois::rawimage::writeText(outimg, jevois::sformat("FSYNC: %4d", int(fsync + 0.499F)),
                                                     w - 130, h + 3, jevois::yuyv::White);

      // ---------- Activity recognition:
      if (bac.size()) jevois::rawimage::writeText(outimg, jevois::join(bac, ", "), 290, h + 15, jevois::yuyv::White);
      
      // ---------- Pickup/flip detection:
      if (pickup) jevois::rawimage::writeText(outimg, "pickup/flip", 214, h + 15, jevois::yuyv::White);

      // ---------- Show raw accel/gyro/compass:
      if (has_accel || has_gyro || has_cpass)
      {
        // Convert from raw data to scaled data:
        jevois::IMUdata dd(rd, itsIMU->arange::get(), itsIMU->grange::get());

        // Display the ones that are on:
        if (has_accel)
          jevois::rawimage::writeText(outimg,
                                      jevois::sformat("Acc: x=%+06.2fg y=%+06.2fg z=%+06.2fg",
                                                      dd.ax(), dd.ay(), dd.az()),
                                      333, h + 3, jevois::yuyv::White);
        if (has_gyro)
          jevois::rawimage::writeText(outimg,
                                      jevois::sformat("Gyr: x=%+07.1fdps y=%+07.1fdps z=%+07.1fdps",
                                                      dd.gx(), dd.gy(), dd.gz()),
                                      3, h + 27, jevois::yuyv::White);
        if (has_cpass)
          jevois::rawimage::writeText(outimg,
                                      jevois::sformat("Mag: %+09.2fuT %+09.2fuT %+09.2fuT",
                                                      dd.mx(), dd.my(), dd.mz()),
                                      333, h + 27, jevois::yuyv::White);
      }

      // ---------- Show number of IMU packets processed:
      jevois::rawimage::writeText(outimg, jevois::sformat("%03d pkts", npkt), 586, h + 3, jevois::yuyv::White);
      
      // Decay counters for some ephemerous messages:
      if (pickup > 0) --pickup;

      // Send the output image with our processing results to the host over USB:
      outframe.send();
    }

  private:
    std::shared_ptr<jevois::ICM20948> itsIMU;
};

// Allow the module to be loaded as a shared object (.so) file:
JEVOIS_REGISTER_MODULE(DemoDMP);
