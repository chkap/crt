import math

import numpy as np

import feature_extractor
from config import TrainDataCfg
from simgeo import Rect


def clip_image(image,rect):
    iw = image.shape[1]
    ih = image.shape[0]
    im_rect = Rect(0,0,iw,ih)
    if rect.is_in_rect(im_rect):
        return image[rect.y:rect.y+rect.h,rect.x:rect.x+rect.w,:].copy()

    xa = np.arange(rect.w)+rect.x
    xa[xa<0] = 0
    xa[xa>=iw] = iw-1
    xa = np.tile(xa[None,:],(rect.h,1))

    ya = np.arange(rect.h)+rect.y
    ya[ya<0] = 0
    ya[ya>=ih] = ih-1
    ya = np.tile(ya[:,None],(1,rect.w))

    return image[ya,xa]


class TrainData(object):

    def __init__(self, patch, patch_rect, gt_rect, feature, response):
        self.patch = patch
        self.patch_rect = patch_rect
        self.gt_rect = gt_rect
        self.feature = feature
        self.response = response


class TrainDataProvider(object):

    def __init__(self, extractor, init_rect):
        self.extractor = extractor
        self.search_patch_ratio = TrainDataCfg.SEARCH_PATCH_RATIO
        init_search_rect = init_rect.get_copy().scale_from_center(self.search_patch_ratio)
        _tmp = self.search_patch_ratio*self.extractor.get_resolution()
        self.patch_scale_w = int(int(init_search_rect.w / float(_tmp) + 0.5) * _tmp)
        self.patch_scale_h = int(int(init_search_rect.h / float(_tmp) + 0.5) * _tmp)

        self.response_sigma_x = float(self.patch_scale_w) / self.extractor.get_resolution() * \
            TrainDataCfg.RESPONSE_GAUSSIAN_SIGMA_RATIO
        self.response_sigma_y = float(self.patch_scale_h) / self.extractor.get_resolution() * \
            TrainDataCfg.RESPONSE_GAUSSIAN_SIGMA_RATIO

    def generate_input_feature(self, image, patch_rect):
        patch = image.clip(patch_rect)
        if patch.shape[1] == self.patch_scale_h and patch.shape[0] == self.patch_scale_w:
            feature = self.extractor.extract_feature(patch)
        else:
            patch_scaled = cv2.resize(patch, (self.patch_scale_w, self.patch_scale_h))
            feature = self.extractor.extract_feature(patch_scaled)
        assert feature.shape[2] == self.extractor.get_channel_num()

        return feature

    def generate_label_response(self, response_size, patch_rect, gt_rect):
        dx, dy = gt_rect.get_center() - patch_rect.get_center()
        _x_resolution = patch_rect.w / float(response_size[0])
        _y_resolution = patch_rect.h / float(response_size[1])
        dxi = math.floor(float(dx) / _x_resolution + 0.5)
        dyi = math.floor(float(dy) / _y_resolution + 0.5)
        xi = int(dxi + response_size[0] / 2.0)
        yi = int(dyi + response_size[1] / 2.0)
        assert 0 <= xi < response_size[0] and 0 <= yi < response_size[1]

        _x_index = np.arange(0, response_size[0])
        _y_index = np.arange(0, response_size[1])
        yv, xv = np.meshgrid(_y_index, _x_index, indexing='ij')
        yv -= yi
        xv -= xi
        _y1 = yv * yv / 2 / self.response_sigma_y / self.response_sigma_y
        _x1 = xv * xv / 2 / self.response_sigma_x / self.response_sigma_x
        response = np.exp(-(_y1 + _x1))
        return response

    def generate_train_data(self, image, gt_rect):
        patch_rect = gt_rect.get_copy().scale_from_center(self.search_patch_ratio)
        patch = image.clip(patch_rect)
        if patch.shape[1] == self.patch_scale_h and patch.shape[0] == self.patch_scale_w:
            feature = self.extractor.extract_feature(patch)
        else:
            patch_scaled = cv2.resize(patch, (self.patch_scale_w, self.patch_scale_h))
            feature = self.extractor.extract_feature(patch_scaled)
        assert feature.shape[2] == self.extractor.get_channel_num()

        dx, dy = gt_rect.get_center() - patch_rect.get_center()
        _x_resolution = patch.shape[1] / float(feature.shape[1])
        _y_resolution = patch.shape[0] / float(feature.shape[0])
        dxi = math.floor(float(dx)/_x_resolution + 0.5)
        dyi = math.floor(float(dy)/_y_resolution + 0.5)
        xi = int(dxi + feature.shape[1] / 2.0)
        yi = int(dyi + feature.shape[0] / 2.0)
        assert 0 <= xi < feature.shape[0] and 0 <= yi < feature.shape[1]

        _x_index = np.arange(0, feature.shape[1])
        _y_index = np.arange(0, feature.shape[0])
        yv, xv = np.meshgrid(_y_index, _x_index, indexing='ij')
        yv -= yi
        xv -= xi
        _y1 = yv*yv/2/self.response_sigma_y/self.response_sigma_y
        _x1 = xv*xv/2/self.response_sigma_x/self.response_sigma_x
        response = np.exp(-(_y1+_x1))

        return TrainData(patch, patch_rect, gt_rect.get_copy(), feature, response)

    def get_final_prediction(self, patch_rect, response_size, predict_index):
        res_width, res_height = response_size
        _yi, _xi = predict_index

        _x_resolution = patch_rect.w / float(res_width)
        _y_resolution = patch_rect.h / float(res_height)

        _dyi = _yi - int(res_height / 2.0)
        _dxi = _xi - int(res_width / 2.0)

        patch_cx, patch_cy = patch_rect.get_center()
        pd_cx, pd_cy = patch_cx + _dxi*_x_resolution, patch_cy + _dyi*_y_resolution
        pd_w, pd_h = int(patch_rect.w/self.search_patch_ratio), int(patch_rect.h/self.search_patch_ratio)

        pd_tlx = int(pd_cx - pd_w/2.0 + 0.5)
        pd_tly = int(pd_cy - pd_h/2.0 + 0.5)

        final_rect = Rect(pd_tlx, pd_tly, pd_w, pd_h)
        return final_rect
