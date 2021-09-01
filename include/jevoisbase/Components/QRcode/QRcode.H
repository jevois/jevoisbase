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

#pragma once

#include <jevois/Component/Component.H>
#include <jevois/GPU/GUIhelper.H>
#include <zbar.h>

namespace jevois { class StdModule; }

namespace qrcode
{
  static jevois::ParameterCategory const ParamCateg("QRcode/Barcode Options");

  //! Parameter \relates QRcode
  JEVOIS_DECLARE_PARAMETER_WITH_CALLBACK(symbol, std::string, "Type(s) of QR code / barcode symbols searched for, "
                                         "multiple entries are possible and should be separated by / characters. "
                                         "Selecting many symbol types or ALL will slow down processing. Supported "
                                         "symbols are: QRCODE, EAN2, EAN5, EAN8, EAN13, UPCE, UPCA, ISBN10, ISBN13, "
                                         "COMPOSITE, I25, DATABAR, DATABAREXP, CODABAR, CODE39, PDF417, CODE93, "
                                         "and CODE128",
                                         "ALL", ParamCateg);
  
  //! Parameter \relates QRcode
  JEVOIS_DECLARE_PARAMETER(xdensity, int, "Scanner vertical scan density (pixel stride), or 0 to disable",
                           1, ParamCateg);
  
  //! Parameter \relates QRcode
  JEVOIS_DECLARE_PARAMETER(ydensity, int, "Scanner horizontal scan density (pixel stride), or 0 to disable",
                           1, ParamCateg);
}

//! QRcode and Barcode detection using ZBar
/*! This is drectly modeled after the scan_image.cpp ZBar example code. \ingroup components */
class QRcode : public jevois::Component,
               public jevois::Parameter<qrcode::symbol, qrcode::xdensity, qrcode::ydensity>
{
  public:
    //! Constructor
    QRcode(std::string const & instance);

    //! Destructor
    ~QRcode();
    
    //! Process a greayscale image
    /*! Results will be stored in the image */
    void process(zbar::Image & image);

    //! Process a greayscale image
    /*! Results will be stored in the provided vector */
    void process(zbar::Image & image, std::vector<std::string> & results);

    //! Draw any markers previously detected by process()
    /*! If txtx,txty are positive, also print a text string there */
    void drawDetections(jevois::RawImage & outimg, int txtx, int txty, zbar::Image & zgray, int w, int h, size_t nshow);

#ifdef JEVOIS_PRO
    //! Draw any markers previously detected by process()
    void drawDetections(jevois::GUIhelper & helper, zbar::Image & zgray, int w, int h);
#endif
    
    //! Send serial messages about our detections
    void sendSerial(jevois::StdModule * mod, zbar::Image & img, unsigned int camw, unsigned int camh);

  protected:
    //! Parse symbol config strings when they change
    void onParamChange(qrcode::symbol const & param, std::string const & newval) override;
    
    std::shared_ptr<zbar::ImageScanner> itsScanner;
};
