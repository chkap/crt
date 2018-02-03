
If you failed to install and run this tracker, please email me (chkhappy@hotmail.com or chkap@hust.edu.cn).

## Prerequisites

* python3
* tensorflow >= 1.0
* numpy
* matplotlib
* opencv

## VGG model for feature extraction
In our tracker, we can integrate either hand-crafted features (like HOG, ColorName features) or deeply learned convolutional features.
To achieve the best performance, we need to integrate the deep convolutional features extracted through VGG model.
The VGG model can be downloaded from my shared google drive:[VGG_16_layers_py3.npz](https://drive.google.com/file/d/0B1sg8Yyw1JCDOUNsYkpQTGdLYVU/view?usp=sharing).
Then, you should copy the VGG model file `VGG_16_layers_py3.npz` to the subfolder './vgg_model', so that the tracker can find and load the vgg_model to extract CNN features.

## Detailed steps to install the prerequisites

It is highly recommended to install anaconda (a powerful python distribution) first.

* install anaconda first.
* create a virtual environment in anaconda: \
  `conda create -n vot_track python=3.5`. \
  The name for this environment `vot_track` can be replaced by anything you like.
* activate the environment you created above: \
  `source activate vot_track`.
* install tensorlfow-gpu following the instructions in the official website of tensorflow. Make sure gpu is enabled for this tracker.
* install the other prerequisites:\
  `conda install numpy matplotlib opencv`.
* test whether the tracker is ready to run by running: \
  `cd {the root directory}` \
  `python test_tracker.py` \
If the tracker is ready, you will see the tracking results.


