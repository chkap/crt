import vot
import sys
import time

import numpy
import collections
import cv2

import simgeo
import tracker


class VOT_CRT_Wrapper(object):

    def __init__(self, image, region):
        self._tracker = tracker.ConvRegTracker()

        _init_rect = simgeo.Rect(region.x, region.y, region.width, region.height)

        self._tracker.init(image, _init_rect)

    def track(self, image):

        res_rect = self._tracker.track(image)

        return vot.Rectangle(res_rect.x, res_rect.y, res_rect.w, res_rect.h)

handle = vot.VOT("rectangle")
selection = handle.region()

imagefile = handle.frame()
if not imagefile:
    sys.exit(0)

image = cv2.imread(imagefile)
trk = VOT_CRT_Wrapper(image, selection)
while True:
    imagefile = handle.frame()
    if not imagefile:
        break
    image = cv2.imread(imagefile)
    region = trk.track(image)
    handle.report(region)

