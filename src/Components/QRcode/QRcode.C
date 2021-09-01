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
#include <jevois/Core/Module.H>
#include <jevois/Image/RawImageOps.H>
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

// ####################################################################################################
void QRcode::sendSerial(jevois::StdModule * mod, zbar::Image & img, unsigned int camw, unsigned int camh)
{
  for (zbar::Image::SymbolIterator symbol = img.symbol_begin(); symbol != img.symbol_end(); ++symbol)
  {
    std::string id = symbol->get_type_name();
    std::string extra = symbol->get_data();

    std::vector<cv::Point> corners;
        
    // Note: For QR codes, we get 4 points at the corners, but for others we get a bunch of points all over the
    // barcode, hopefully users will not select the Fine serial style...
    for (zbar::Symbol::PointIterator pitr = symbol->point_begin(); pitr != symbol->point_end(); ++pitr)
    {
      zbar::Symbol::Point p(*pitr);
      corners.push_back(cv::Point(p.x, p.y));
    }
    
    mod->sendSerialContour2D(camw, camh, corners, id, extra);
  }
}

// ####################################################################################################
void QRcode::drawDetections(jevois::RawImage & outimg, int txtx, int txty, zbar::Image & zgray,
                            int w, int h, size_t nshow)
{
  static unsigned short const txtcol = jevois::yuyv::White;
  
  // Show all the results:
  std::string txt; std::vector<std::string> qdata;
  for (zbar::Image::SymbolIterator symbol = zgray.symbol_begin(); symbol != zgray.symbol_end(); ++symbol)
  {
    // Build up some strings to be displayed as video overlay:
    txt += ' ' + symbol->get_type_name();
    qdata.push_back(std::string(symbol->get_type_name()) + ": " + symbol->get_data());
    
    // Draw a polygon around the detected symbol: for QR codes, we get 4 points at the corners, but for others we
    // get a bunch of points all over the barcode:
    if (symbol->get_type() == zbar::ZBAR_QRCODE)
    {
      zbar::Symbol::Point pp(-1000000, -1000000);
      for (zbar::Symbol::PointIterator pitr = symbol->point_begin(); pitr != symbol->point_end(); ++pitr)
      {
        zbar::Symbol::Point p(*pitr);
        if (pp.x != -1000000) jevois::rawimage::drawLine(outimg, pp.x, pp.y, p.x, p.y, 1, jevois::yuyv::DarkPink);
        pp = p;
      }
      if (pp.x != -1000000)
      {
        zbar::Symbol::Point p = *(symbol->point_begin());
        jevois::rawimage::drawLine(outimg, pp.x, pp.y, p.x, p.y, 1, jevois::yuyv::DarkPink);
      }
    }
    else
    {
      int tlx = w, tly = h, brx = -1, bry = -1;
      for (zbar::Symbol::PointIterator pitr = symbol->point_begin(); pitr != symbol->point_end(); ++pitr)
      {
        zbar::Symbol::Point p(*pitr);
        if (p.x < tlx) tlx = p.x;
        if (p.x > brx) brx = p.x;
        if (p.y < tly) tly = p.y;
        if (p.y > bry) bry = p.y;
      }
      tlx = std::min(int(w)-1, std::max(0, tlx));
      brx = std::min(int(w)-1, std::max(0, brx));
      tly = std::min(int(h)-1, std::max(0, tly));
      bry = std::min(int(h)-1, std::max(0, bry));
      jevois::rawimage::drawRect(outimg, tlx, tly, brx - tlx, bry - tly, 1, jevois::yuyv::DarkPink);
    }
  }
  
  // Write some strings in the output video with what we found and decoded:
  if (qdata.empty())
    jevois::rawimage::writeText(outimg, "Found no symbols.", txtx, txty, txtcol);
  else
  {
    txt = "Found " + std::to_string(qdata.size()) + " symbols:" + txt;
    jevois::rawimage::writeText(outimg, txt.c_str(), txtx, txty, txtcol);
    for (size_t i = 0; i < std::min(qdata.size(), nshow - 1); ++i)
      jevois::rawimage::writeText(outimg, qdata[i].c_str(), txtx, txty + (i+1) * 10, txtcol);
  }
}

#ifdef JEVOIS_PRO
// ####################################################################################################
void QRcode::drawDetections(jevois::GUIhelper & helper, zbar::Image & zgray, int w, int h)
{
  ImU32 const col = ImColor(255, 192, 64, 255); // for lines

  // Show all the results:
  std::vector<std::string> qdata;
  for (zbar::Image::SymbolIterator symbol = zgray.symbol_begin(); symbol != zgray.symbol_end(); ++symbol)
  {
    // Build up some strings to be displayed as video overlay:
    qdata.push_back("  - " + std::string(symbol->get_type_name()) + ": " + symbol->get_data());
    
    // Draw a polygon around the detected symbol: for QR codes, we get 4 points at the corners, but for others we
    // get a bunch of points all over the barcode:
    if (symbol->get_type() == zbar::ZBAR_QRCODE)
    {
      std::vector<cv::Point2f> points;
      for (zbar::Symbol::PointIterator pitr = symbol->point_begin(); pitr != symbol->point_end(); ++pitr)
      {
        zbar::Symbol::Point p(*pitr);
        points.emplace_back(cv::Point2f(p.x, p.y));
      }
      helper.drawPoly(points, col, true);
    }
    else
    {
      int tlx = w, tly = h, brx = -1, bry = -1;
      for (zbar::Symbol::PointIterator pitr = symbol->point_begin(); pitr != symbol->point_end(); ++pitr)
      {
        zbar::Symbol::Point p(*pitr);
        if (p.x < tlx) tlx = p.x;
        if (p.x > brx) brx = p.x;
        if (p.y < tly) tly = p.y;
        if (p.y > bry) bry = p.y;
      }
      tlx = std::min(int(w)-1, std::max(0, tlx));
      brx = std::min(int(w)-1, std::max(0, brx));
      tly = std::min(int(h)-1, std::max(0, tly));
      bry = std::min(int(h)-1, std::max(0, bry));
      helper.drawRect(tlx, tly, brx, bry, col, true);
    }
  }
  
  // Write some strings in the output video with what we found and decoded:
  helper.itext("Detected " + std::to_string(qdata.size()) + " QRcode/Barcode symbols.");
  for (size_t i = 0; i < qdata.size(); ++i) helper.itext(qdata[i]);
}

#endif
