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

#include <jevoisbase/Components/Saliency/Saliency.H>

namespace surprise
{
  static jevois::ParameterCategory const ParamCateg("Surprise Options");

  //! Parameter \relates Surprise
  JEVOIS_DECLARE_PARAMETER(updatefac, float, "Surprise update factor on every video frame", 0.95F,
                           jevois::Range<float>(0.001F, 0.999F), ParamCateg);

  //! Parameter \relates Surprise
  JEVOIS_DECLARE_PARAMETER(channels, std::string, "Channels to use for surprise computation: any combination of "
                           "S (saliency), G (gist), C (color), I (intensity), O (orientation), F (flicker), and "
                           "M (motion). Duplicate letters will be ignored.",
                           "SCIOFMG", boost::regex("^[SCIOFMG]+$"), ParamCateg);
}


//! Compute Itti & Baldi surprise over video frames
/*! This component detects surprising events in video frames using Itti & Baldi's Bayesian theory of surprise.

    They defined surprise in a formal, quantitative manner (for the first time!), as follows: An observation is
    surprising if it significantly affects the internal (subjective) beliefs of an observer. For example, if I believe
    that there is a 10% chance of rain today (my prior belief), and then I look outside and I see only a few small
    scattered clouds, then I may still believe in that same 10% chance of rain (posterior belief after the
    observation). My observation was not surprising, and Itti & Baldi say that this is because it did not affect my
    beliefs. Formally, when my posterior beliefs after an observation are very similar to what my prior beliefs were
    before the observation, the observation carries no surprise. In contrast, if I see a sky covered with menacing dark
    clouds all over, I may revise my belief to a 80% chance of rain today. Because my posterior beliefs are now much
    different than my prior beliefs, the observation of clouds is said to carry a high surprise. Itti & Baldi further
    specify how to compute surprise by using Bayes' theorem to compute posterior beliefs in a pricipled way, and by
    using the Kullback-Leibler divergence to measure the difference between posterior and prior distributions of
    beliefs. This gives rise to a new quantitative measure of surprise, with a new unit, the 'wow' (one wow of surprise
    is experienced when your belief in something doubles).

    For more information, check out [L. Itti, P. F. Baldi, Bayesian Surprise Attracts Human Attention, Vision Research,
    Vol. 49, No. 10, pp. 1295-1306, May 2009.](http://ilab.usc.edu/publications/doc/Itti_Baldi09vr.pdf)

    In this component, we compute feature maps and a saliency map. These will provide some degree of invariance and
    robustness to noise, which will yield more stable overall results than if we were to compute surprise directly over
    RGB pixel values (see next point);

    We then compute surprise in each pixel of each feature map. This is similar to what Itti & Baldi did but simplified
    to run in real time on the JeVois smart camera. Each pixel in each feature map will over time gather beliefs about
    what it usually 'sees' at that location in the video. When things change significantly and in a surprising way, that
    pixel will emit a local surprise signal. Because surprise is more complex than just computing an instantaneous
    difference, or measuring whether the current observation simply is an outlier to a learned distribution, it will be
    able to handle periodic motions (foliage in the wind, ripples on a body of water), periodic flickers (a constantly
    blinking light in the field of view), and noise.

    This approach is related to [R. C. Voorhies, L. Elazary, L. Itti, Neuromorphic Bayesian Surprise for Far-Range Event
    Detection, In: Proc. 9th IEEE International Conference on Advanced Video and Signal-Based Surveillance (AVSS),
    Beijing, China, Sep 2012.](http://ilab.usc.edu/publications/doc/Voorhies_etal12avss.pdf).

    \ingroup components */
class Surprise : public jevois::Component,
                 public jevois::Parameter<surprise::updatefac, surprise::channels>
{
  public:
    //! Constructor
    Surprise(std::string const & instance);

    //! Virtual destructor for safe inheritance
    ~Surprise();
    
    //! Compute surprise from a YUYV video frame and return the surprise value in wows
    double process(jevois::RawImage const & input);

  protected:
    std::shared_ptr<Saliency> itsSaliency;
    std::vector<float> itsAlpha, itsBeta;
};


