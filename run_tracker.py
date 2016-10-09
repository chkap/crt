
import cv2

import conv_reg_config
from tracker import ConvRegTracker
from simgeo import Rect


def run_tracker(s_frames, init_rect):

    trker = ConvRegTracker()

    image = cv2.imread(s_frames[0])
    rect = Rect(*init_rect)
    trker.init(image, rect)

    res = []
    res.append(list(init_rect))
    for i in range(1, len(s_frames)):
        image = cv2.imread(s_frames[i])
        rect = trker.track(image)
        res.append([rect.x, rect.y, rect.w, rect.h])

    return res


if __name__ == '__main__':
    pass

