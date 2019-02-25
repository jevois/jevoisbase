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
    chip), which allows it to compute and filter Euler angles or quaternions directly inside the IMU chip.

    This module only works with optional JeVois sensors that include an IMU! The base JeVois-A33 smart camera does not
    have an onboard IMU.

    @author Laurent Itti

    @videomapping YUYV 640 520 40.0 YUYV 640 480 40.0 JeVois DemoDMP
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

      // Get one IMU reading:
      jevois::DMPdata d = itsIMU->getDMP();

      // Note: we get different packets for steps and for quaternions. So let's cache the steps:
      static long step_ts = 0;
      static long steps = 0;
      static std::vector<std::string> bac;
      static int bactim = 0; // number of frames for which we display the activity
      static cv::Vec3d rvec(0.0, 0.0, 0.0); // Euler angles computed from quaternion
      static cv::Vec3d tvec(0.0, 0.0, 0.5); // Place cube 0.5 meters in front of the camera
      float const hsz = 0.075F; // half size of cube, in meters
      static int pickup = 0; // Number of frames remaining to show pickup message
        
      // ---------- Quaternions using 9 DOF:
      if (d.header1 & JEVOIS_DMP_QUAT9)
      {
        Eigen::Quaterniond q(0.0, d.fix2float(d.quat9[0]), d.fix2float(d.quat9[1]), d.fix2float(d.quat9[2]));

        // Compute the scalar part, which is not returned by the IMU, to get a normalized quaternion:
        q.w() = std::sqrt(1.0 - q.squaredNorm());
        
        // Display it:
        jevois::rawimage::writeText(outimg, jevois::sformat("Quat: x=%+09.6f y=%+09.6f z=%+09.6f", q.x(), q.y(), q.z()),
                                    3, h + 3, jevois::yuyv::White);

        // Compute OpenCV rvec for use with projectPoints() below:
        // You could also compute Euler angles: Eigen::Vector3f eig_rvec = q.toRotationMatrix().eulerAngles(0, 1, 2);
        Eigen::Matrix3d eig_rmat = q.toRotationMatrix();
        cv::Mat rmat(3, 3, CV_64FC1); cv::eigen2cv(eig_rmat, rmat);
        cv::Rodrigues(rmat, rvec);
      }

      // ---------- Quaternions using 6 DOF:
      if (d.header1 & JEVOIS_DMP_QUAT6)
      {
        Eigen::Quaterniond q(0.0, d.fix2float(d.quat6[0]), d.fix2float(d.quat6[1]), d.fix2float(d.quat6[2]));

        // Compute the scalar part, which is not returned by the IMU, to get a normalized quaternion:
        q.w() = std::sqrt(1.0 - q.squaredNorm());
        
        // Display it:
        jevois::rawimage::writeText(outimg, jevois::sformat("Quat: x=%+09.6f y=%+09.6f z=%+09.6f", q.x(), q.y(), q.z()),
                                    3, h + 3, jevois::yuyv::White);

        // Compute OpenCV rvec for use with projectPoints() below:
        Eigen::Matrix3d eig_rmat = q.toRotationMatrix();
        cv::Mat rmat(3, 3, CV_64FC1); cv::eigen2cv(eig_rmat, rmat);
        cv::Rodrigues(rmat, rvec);
      }

      // ---------- Display a cube rotated by our latest quaternion:
      
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
      
      // ---------- Step detection:
      if (d.header1 & JEVOIS_DMP_PED_STEPDET) { step_ts = d.stepts; steps += d.steps; }
      jevois::rawimage::writeText(outimg, jevois::sformat("Steps: %6u (last at %010u)", steps, step_ts),
                                  3, h + 15, jevois::yuyv::White);

      // ---------- Activity recognition:
      if (d.header2 & JEVOIS_DMP_ACT_RECOG) { LINFO("bac = 0x" << std::hex << d.bacstate);bac = d.activity(); bactim = 20; }
      if (bactim) jevois::rawimage::writeText(outimg, jevois::join(bac, ", "), 260, h + 15, jevois::yuyv::White);
      
      // ---------- Pickup/flip detection:
      if (d.header2 & JEVOIS_DMP_FLIP_PICKUP) { LINFO("pickup = 0x" << std::hex << d.pickup); pickup = 20; }
      if (pickup) jevois::rawimage::writeText(outimg, "pickup/flip", 260, h + 3, jevois::yuyv::White);

      // Decay counters for some ephemerous messages:
      if (pickup > 0) --pickup;
      if (bactim > 0) --bactim;

      // Send the output image with our processing results to the host over USB:
      outframe.send();
    }

  private:
    std::shared_ptr<jevois::ICM20948> itsIMU;
};

// Allow the module to be loaded as a shared object (.so) file:
JEVOIS_REGISTER_MODULE(DemoDMP);
