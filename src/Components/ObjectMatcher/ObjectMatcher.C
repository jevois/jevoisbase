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

#include <jevoisbase/Components/ObjectMatcher/ObjectMatcher.H>
#include <jevois/Debug/Log.H>
#include <jevois/Debug/Profiler.H>
#include <sys/types.h>
#include <fcntl.h>
#include <dirent.h>
#include <sys/stat.h>
#include <unistd.h>

#include <opencv2/core/utility.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>

// ####################################################################################################
ObjectMatcher::~ObjectMatcher()
{ }

// ####################################################################################################
void ObjectMatcher::postInit()
{
  // Initialize our feature computer and matcher:
  itsFeatureDetector = cv::xfeatures2d::SURF::create(hessian::get());
  size_t const ncores = std::min(4U, std::thread::hardware_concurrency());
  for (size_t i = 0; i < ncores; ++i)
    itsMatcher.push_back(cv::Ptr<cv::DescriptorMatcher>(new cv::BFMatcher(cv::NORM_L2)));

  //itsFeatureDetector = cv::ORB::create();
  //itsMatcher.reset(new cv::BFMatcher(cv::NORM_HAMMING));

  // Load training images and compute keypoints and descriptors:
  itsTrainData.clear();
  std::string const dirname = absolutePath(traindir::get());

  LINFO("Training from " << dirname);
  
  DIR * dir = opendir(dirname.c_str());
  if (dir == nullptr) PLFATAL("Cound not scan directory " << dirname);

  struct dirent * ent;
  while ((ent = readdir(dir)) != nullptr)
  {
    std::string const fname = ent->d_name;
    std::string const fullfname = dirname + "/" + fname;
    if (fname[0] == '.') continue; // skip files that start with a period
    struct stat st;
    if (stat(fullfname.c_str(), &st) == -1) continue; // ignore any stat error
    if ((st.st_mode & S_IFDIR) != 0) continue; // skip sub-directories

    // Create a training data entry:
    itsTrainData.push_back(TrainData());
    TrainData & td = itsTrainData.back();
    td.name = fname;
    td.image = cv::imread(fullfname);

    // Compute keypoints and descriptors:
    itsFeatureDetector->detectAndCompute(td.image, cv::Mat(), td.keypoints, td.descriptors);

    LDEBUG(fname << ": " << td.keypoints.size() << " keypoints.");
  }
  
  closedir(dir);

  LINFO("Training complete for " << itsTrainData.size() << " images.");
}

// ####################################################################################################
double ObjectMatcher::process(cv::Mat const & img, size_t & trainidx, std::vector<cv::Point2f> & corners)
{
  std::vector<cv::KeyPoint> keypoints;
  this->detect(img, keypoints);

  cv::Mat descriptors;
  this->compute(img, keypoints, descriptors);

  return this->match(keypoints, descriptors, trainidx, corners);
}

// ####################################################################################################
double ObjectMatcher::process(cv::Mat const & img, size_t & trainidx)
{
  std::vector<cv::KeyPoint> keypoints;
  this->detect(img, keypoints);

  cv::Mat descriptors;
  this->compute(img, keypoints, descriptors);

  return this->match(keypoints, descriptors, trainidx);
}

// ####################################################################################################
void ObjectMatcher::detect(cv::Mat const & img, std::vector<cv::KeyPoint> & keypoints)
{
  itsFeatureDetector->detect(img, keypoints);
}

// ####################################################################################################
void ObjectMatcher::compute(cv::Mat const & img, std::vector<cv::KeyPoint> & keypoints, cv::Mat & descriptors)
{
  itsFeatureDetector->compute(img, keypoints, descriptors);
}

// ####################################################################################################
double ObjectMatcher::match(std::vector<cv::KeyPoint> const & keypoints, cv::Mat const & descriptors,
                            size_t & trainidx, std::vector<cv::Point2f> & corners)
{
  if (itsTrainData.empty()) LFATAL("No training data loaded");

  // Parallelize the matching over our cores:
  size_t const ncores = std::min(4U, std::thread::hardware_concurrency());
  size_t const ntrain = itsTrainData.size();
  size_t const r = ntrain % ncores;
  size_t const q = (ntrain - r) / ncores;
  size_t percore = q + 1; // up to r cores will process q+1 elements, and up to ncores-r will process q
  std::vector<std::future<ObjectMatcher::MatchData> > fut;

  size_t startidx = 0;
  for (size_t i = 0; i < ncores; ++i)
  {
    if (i == r) percore = q;
    if (percore == 0) break;
    
    size_t const endidx = std::min(startidx + percore, ntrain);
    fut.push_back(std::async(std::launch::async, [&](size_t cn, size_t mi, size_t ma)
                             { return this->matchcore(cn, keypoints, descriptors, mi, ma, true); },
                             i, startidx, endidx));
    startidx += percore;
  }

  // Wait for all jobs to complete, ignore any exception (which may seldom occur due to ill conditioned homography or
  // such), and pick the best result:
  double bestdist = 1.0e30;
  for (auto & f : fut)
    try
    {
      MatchData md = f.get();
      if (md.avgdist < bestdist) { bestdist = md.avgdist; trainidx = md.trainidx; corners = md.corners; }
    }
    catch (...) { }

  return bestdist;
}

// ####################################################################################################
double ObjectMatcher::match(std::vector<cv::KeyPoint> const & keypoints, cv::Mat const & descriptors,
                            size_t & trainidx)
{
  if (itsTrainData.empty()) LFATAL("No training data loaded");

  // Parallelize the matching over our cores:
  size_t const ncores = std::min(4U, std::thread::hardware_concurrency());
  size_t const ntrain = itsTrainData.size();
  size_t const r = ntrain % ncores;
  size_t const q = (ntrain - r) / ncores;
  size_t percore = q + 1; // up to r cores will process q+1 elements, and up to ncores-r will process q
  std::vector<std::future<ObjectMatcher::MatchData> > fut;

  size_t startidx = 0;
  for (size_t i = 0; i < ncores; ++i)
  {
    if (i == r) percore = q;
    if (percore == 0) break;
    
    size_t const endidx = std::min(startidx + percore, ntrain);
    fut.push_back(std::async(std::launch::async, [&](size_t cn, size_t mi, size_t ma)
                             { return this->matchcore(cn, keypoints, descriptors, mi, ma, false); },
                             i, startidx, endidx));
    startidx += percore;
  }

  // Wait for all jobs to complete, ignore any exception (which may seldom occur due to ill conditioned homography or
  // such), and pick the best result:
  double bestdist = 1.0e30;
  for (auto & f : fut)
    try
    {
      MatchData md = f.get();
      if (md.avgdist < bestdist) { bestdist = md.avgdist; trainidx = md.trainidx; }
    }
    catch (...) { }

  return bestdist;
}

// ####################################################################################################
ObjectMatcher::MatchData ObjectMatcher::matchcore(size_t corenum, std::vector<cv::KeyPoint> const & keypoints,
                                                  cv::Mat const & descriptors, size_t minidx, size_t maxidx,
                                                  bool do_corners)
{
  // Compute matches between query and training images:
  double bestsofar = 1.0e30;
  MatchData mdata { bestsofar, 0, { } };

  for (size_t idx = minidx; idx < maxidx; ++idx)
  {
    TrainData const & td = itsTrainData[idx];

    std::vector<cv::DMatch> matches;
    itsMatcher[corenum]->match(descriptors, td.descriptors, matches);
    if (matches.empty()) continue;
    
    // Sort the matches by distance:
    std::sort(matches.begin(), matches.end());
    std::vector<cv::DMatch> good_matches;
    double const minDist = matches.front().distance, maxDist = matches.back().distance;

    // Keep only the good matches:
    size_t const ptsPairs = std::min(goodpts::get().max(), matches.size());
    double const dthresh = distthresh::get();
    for (size_t i = 0; i < ptsPairs; ++i)
      if (matches[i].distance <= dthresh) good_matches.push_back(matches[i]); else break;

    LDEBUG(td.name << ": Match distances: " << minDist << " .. " << maxDist << ", " <<
           matches.size() << " matches, " << good_matches.size() << " good ones.");

    // Abort here if we did not get enough good matches:
    if (good_matches.size() < goodpts::get().min()) continue;

    // Compute the average match distance:
    double avgdist = 0.0;
    for (cv::DMatch const & gm : good_matches) avgdist += gm.distance;
    avgdist /= good_matches.size();
    if (avgdist >= bestsofar) continue;
    
    LDEBUG("Object match: found " << td.name << " distance " << avgdist);

    // Localize the object:
    std::vector<cv::Point2f> obj, scene;
    for (size_t i = 0; i < good_matches.size(); ++i)
    {
      obj.push_back(td.keypoints[good_matches[i].trainIdx].pt);
      scene.push_back(keypoints[good_matches[i].queryIdx].pt);
    }

    if (do_corners)
    {
      // Get the corners from the training image:
      std::vector<cv::Point2f> obj_corners(4);
      obj_corners[0] = cv::Point(0, 0);
      obj_corners[1] = cv::Point(td.image.cols, 0);
      obj_corners[2] = cv::Point(td.image.cols, td.image.rows);
      obj_corners[3] = cv::Point(0, td.image.rows);

      // Project the corners into the scene image. This may throw if the homography is too messed up:
      try
      {
        // Compute the homography:
        cv::Mat H = cv::findHomography(obj, scene, cv::RANSAC, 5.0);

        // Check that the homography is not garbage:
        std::vector<double> sv; cv::SVD::compute(H, sv, cv::SVD::NO_UV);
        LDEBUG("Homography sv " << sv.front() << " ... " << sv.back());
      
        if (sv.empty() == false && sv.back() >= 0.001 && sv.front() / sv.back() >= 100.0)
        {
          // The homography looks good, use it to map the object corners to the scene image:
          cv::perspectiveTransform(obj_corners, mdata.corners, H);

          // If all went well, this is our current best object match:
          mdata.trainidx = idx;
          mdata.avgdist = avgdist;
          bestsofar = avgdist;
        }
      } catch (...) { }
    }
    else
    {
      mdata.trainidx = idx;
      mdata.avgdist = avgdist;
      bestsofar = avgdist;
    }
  }
  
  return mdata;
}

// ####################################################################################################
ObjectMatcher::TrainData const & ObjectMatcher::traindata(size_t idx) const
{
  if (idx >= itsTrainData.size()) LFATAL("Index too large");
  return itsTrainData[idx];
}

// ####################################################################################################
size_t ObjectMatcher::numtrain() const
{ return itsTrainData.size(); }

