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

#include <jevoisbase/Components/QRcode/QRcode.H>
#include <jevois/Debug/Log.H>
#include <jevois/Util/Utils.H>

// ####################################################################################################
QRcode::QRcode(std::string const & instance) :
    jevois::Component(instance), itsScanner(new zbar::ImageScanner())
{
  itsScanner->set_config(zbar::ZBAR_NONE, zbar::ZBAR_CFG_ENABLE, 1);
}

// ####################################################################################################
QRcode::~QRcode()
{ }

// ####################################################################################################
void QRcode::onParamChange(qrcode::symbol const & JEVOIS_UNUSED_PARAM(param), std::string const & newval)
{
  // Disable scanning for no symbols:
  itsScanner->set_config(zbar::ZBAR_NONE, zbar::ZBAR_CFG_ENABLE, 0);

  // Enable the requested ones:
  std::vector<std::string> vec = jevois::split(newval, "/");
  for (std::string const & v : vec)
    if (v == "ALL")
    {
      itsScanner->set_config(zbar::ZBAR_QRCODE, zbar::ZBAR_CFG_ENABLE, 1);
      itsScanner->set_config(zbar::ZBAR_EAN2, zbar::ZBAR_CFG_ENABLE, 1);
      itsScanner->set_config(zbar::ZBAR_EAN5, zbar::ZBAR_CFG_ENABLE, 1);
      itsScanner->set_config(zbar::ZBAR_EAN8, zbar::ZBAR_CFG_ENABLE, 1);
      itsScanner->set_config(zbar::ZBAR_EAN13, zbar::ZBAR_CFG_ENABLE, 1);
      itsScanner->set_config(zbar::ZBAR_UPCE, zbar::ZBAR_CFG_ENABLE, 1);
      itsScanner->set_config(zbar::ZBAR_UPCA, zbar::ZBAR_CFG_ENABLE, 1);
      itsScanner->set_config(zbar::ZBAR_ISBN10, zbar::ZBAR_CFG_ENABLE, 1);
      itsScanner->set_config(zbar::ZBAR_ISBN13, zbar::ZBAR_CFG_ENABLE, 1);
      itsScanner->set_config(zbar::ZBAR_COMPOSITE, zbar::ZBAR_CFG_ENABLE, 1);
      itsScanner->set_config(zbar::ZBAR_I25, zbar::ZBAR_CFG_ENABLE, 1);
      itsScanner->set_config(zbar::ZBAR_DATABAR, zbar::ZBAR_CFG_ENABLE, 1);
      itsScanner->set_config(zbar::ZBAR_DATABAR_EXP, zbar::ZBAR_CFG_ENABLE, 1);
      itsScanner->set_config(zbar::ZBAR_CODABAR, zbar::ZBAR_CFG_ENABLE, 1);
      itsScanner->set_config(zbar::ZBAR_CODE39, zbar::ZBAR_CFG_ENABLE, 1);
      itsScanner->set_config(zbar::ZBAR_PDF417, zbar::ZBAR_CFG_ENABLE, 1);
      itsScanner->set_config(zbar::ZBAR_CODE93, zbar::ZBAR_CFG_ENABLE, 1);
      itsScanner->set_config(zbar::ZBAR_CODE128, zbar::ZBAR_CFG_ENABLE, 1);
    }
    else if (v == "QRCODE") itsScanner->set_config(zbar::ZBAR_QRCODE, zbar::ZBAR_CFG_ENABLE, 1);
    else if (v == "EAN2") itsScanner->set_config(zbar::ZBAR_EAN2, zbar::ZBAR_CFG_ENABLE, 1);
    else if (v == "EAN5") itsScanner->set_config(zbar::ZBAR_EAN5, zbar::ZBAR_CFG_ENABLE, 1);
    else if (v == "EAN8") itsScanner->set_config(zbar::ZBAR_EAN8, zbar::ZBAR_CFG_ENABLE, 1);
    else if (v == "EAN13") itsScanner->set_config(zbar::ZBAR_EAN13, zbar::ZBAR_CFG_ENABLE, 1);
    else if (v == "UPCE") itsScanner->set_config(zbar::ZBAR_UPCE, zbar::ZBAR_CFG_ENABLE, 1);
    else if (v == "UPCA") itsScanner->set_config(zbar::ZBAR_UPCA, zbar::ZBAR_CFG_ENABLE, 1);
    else if (v == "ISBN10")
    {
      itsScanner->set_config(zbar::ZBAR_EAN13, zbar::ZBAR_CFG_ENABLE, 1); // required?
      itsScanner->set_config(zbar::ZBAR_ISBN10, zbar::ZBAR_CFG_ENABLE, 1);
    }
    else if (v == "ISBN13")
    {
      itsScanner->set_config(zbar::ZBAR_EAN13, zbar::ZBAR_CFG_ENABLE, 1); // required?
      itsScanner->set_config(zbar::ZBAR_ISBN13, zbar::ZBAR_CFG_ENABLE, 1);
    }
    else if (v == "COMPOSITE") itsScanner->set_config(zbar::ZBAR_COMPOSITE, zbar::ZBAR_CFG_ENABLE, 1);
    else if (v == "I25") itsScanner->set_config(zbar::ZBAR_I25, zbar::ZBAR_CFG_ENABLE, 1);
    else if (v == "DATABAR") itsScanner->set_config(zbar::ZBAR_DATABAR, zbar::ZBAR_CFG_ENABLE, 1);
    else if (v == "DATABAREXP") itsScanner->set_config(zbar::ZBAR_DATABAR_EXP, zbar::ZBAR_CFG_ENABLE, 1);
    else if (v == "CODABAR") itsScanner->set_config(zbar::ZBAR_CODABAR, zbar::ZBAR_CFG_ENABLE, 1);
    else if (v == "CODE39") itsScanner->set_config(zbar::ZBAR_CODE39, zbar::ZBAR_CFG_ENABLE, 1);
    else if (v == "PDF417") itsScanner->set_config(zbar::ZBAR_PDF417, zbar::ZBAR_CFG_ENABLE, 1);
    else if (v == "CODE93") itsScanner->set_config(zbar::ZBAR_CODE93, zbar::ZBAR_CFG_ENABLE, 1);
    else if (v == "CODE128") itsScanner->set_config(zbar::ZBAR_CODE128, zbar::ZBAR_CFG_ENABLE, 1);
    else LFATAL("Invalid symbol type [" << v <<']');
}

// ####################################################################################################
void QRcode::process(zbar::Image & image)
{
  // Update config using Component parameters:
  itsScanner->set_config(zbar::ZBAR_NONE, zbar::ZBAR_CFG_X_DENSITY, qrcode::xdensity::get());
  itsScanner->set_config(zbar::ZBAR_NONE, zbar::ZBAR_CFG_Y_DENSITY, qrcode::ydensity::get());

  // Scan for barcodes, they will be stored in the image:
  int n = itsScanner->scan(image);
  if (n == -1) LERROR("ZBar error");
}

// ####################################################################################################
void QRcode::process(zbar::Image & image, std::vector<std::string> & results)
{
  // Update config using Component parameters:
  itsScanner->set_config(zbar::ZBAR_NONE, zbar::ZBAR_CFG_X_DENSITY, qrcode::xdensity::get());
  itsScanner->set_config(zbar::ZBAR_NONE, zbar::ZBAR_CFG_Y_DENSITY, qrcode::ydensity::get());

  // Clear any input junk:
  results.clear();
  
  // Scan for barcodes, they will be stored in the image:
  int n = itsScanner->scan(image);

  if (n == -1)
    LERROR("ZBar error");
  else if (n > 0)
    for (zbar::Image::SymbolIterator symbol = image.symbol_begin(); symbol != image.symbol_end(); ++symbol)
      results.push_back(std::string(symbol->get_type_name()) + ' ' + symbol->get_data());
}

