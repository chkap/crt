# CRT: Convolutional Regression for Visual Tracking

## Introduction

This is a python-implemented visual object tracking algorithm, created by Kai Chen.

The two main ideas in this tracking algorithm are:
* We learn a linear regression model by optimizing a single convolution layer via gradient descent.
* We propose an improved objective function to speed up the training and improve the accuracy.

## Citation

If the code has been used in your publication, please cite:
```
@article{DBLP:journals/corr/ChenT16a,
  author    = {Kai Chen and
               Wenbing Tao},
  title     = {Convolutional Regression for Visual Tracking},
  journal   = {CoRR},
  volume    = {abs/1611.04215},
  year      = {2016},
  url       = {http://arxiv.org/abs/1611.04215},
  timestamp = {Wed, 07 Jun 2017 14:41:23 +0200},
  biburl    = {http://dblp.uni-trier.de/rec/bib/journals/corr/ChenT16a},
  bibsource = {dblp computer science bibliography, http://dblp.org}
}
```

## Installation

Please see _install.md_.


## Integrate into VOT-2017

The interface for integrating the tracker into the vot evaluation tool kit is implemented in the module `vot_run_CRT.py`.

A sample `tracer_CRT.m` file can be found in this root folder. You only need to change the directories in this file.

However, the trax tool provided in VOT-toolkit only supports python2.
You will need make some changes to the native trax tool for python.
To make the trax tool for python2 (at `vot-toolkit-root/native/trax/python`) available in python3, we need:

1. replace `xrange` found in all the python source files with `range`.
2. replace line 70 at `trax/region.py` with `tokens = list(map(float, string.split(',')))`.

If you still fail to integrate this tracker into the VOT-2017 toolkit, you may find solutions in my article
[integrate python-based tracker into VOT-2017 toolkit on Ubuntu](http://chkap.com/blog/read?id=18). 
