import math

import numpy as np
import cv2

import feature_extractor
from conv_reg_config import TrainDataCfg
from simgeo import Rect
import display


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

    def __init__(self, extractor, object_rect):
        # search_size: h, w        object_size: h, w
        object_size_h, object_size_w = object_rect.h, object_rect.w
        self.extractor_class = extractor
        self.extractor = self.extractor_class()

        _extractor_resolution = self.extractor.get_resolution()
        _object_aspect = object_size_h / float(object_size_w)

        self.convolution_w = round(math.sqrt(TrainDataCfg.CONVOLUTION_SIZE_TH**2 / float(_object_aspect)))
        self.convolution_h = round(_object_aspect*self.convolution_w)

        # self.feature_size_w = round(search_size_w / float(object_size_w) * self.convolution_w)
        # self.feature_size_h = round(search_size_h / float(object_size_h) * self.convolution_h)
        self.feature_size_w = self.convolution_w * TrainDataCfg.SEARCH_RATIO_WIDTH
        self.feature_size_h = self.convolution_h * TrainDataCfg.SEARCH_RATIO_HEIGHT

        self.input_object_w = self.convolution_w * _extractor_resolution
        self.input_object_h = self.convolution_h * _extractor_resolution

        self.input_search_w = self.feature_size_w * _extractor_resolution
        self.input_search_h = self.feature_size_h * _extractor_resolution

        self.response_size_w = self.feature_size_w - self.convolution_w + 1
        self.response_size_h = self.feature_size_h - self.convolution_h + 1
        assert self.response_size_h %2 == 1 and self.response_size_w %2 == 1

        self.response_sigma_x = self.convolution_w * TrainDataCfg.RESPONSE_GAUSSIAN_SIGMA_RATIO
        self.response_sigma_y = self.convolution_h * TrainDataCfg.RESPONSE_GAUSSIAN_SIGMA_RATIO

        self.motion_sigma = TrainDataCfg.CONVOLUTION_SIZE_TH * TrainDataCfg.MOTION_GAUSSIAN_SIGMA_RATIO

        self.scale_test_num = TrainDataCfg.SCALE_TEST_NUM
        assert self.scale_test_num >= 0
        self.scale_ratio = TrainDataCfg.SCALE_RATIO

        self._show_label_response_fid = TrainDataCfg.SHOW_LABEL_RESPONSE_FID
        self._show_motion_map_fid = TrainDataCfg.SHOW_MOTION_MAP_FID
        self._show_search_bgr_fid = TrainDataCfg.SHOW_SEARCH_BGR_FID

        # self.search_patch_ratio = TrainDataCfg.SEARCH_PATCH_RATIO
        # _size = math.sqrt(init_rect.w * init_rect.h)
        # # if _size > TrainDataCfg.OBJECT_RESIZE_TH:
        # #     _scale = TrainDataCfg.OBJECT_RESIZE_TH / float(_size)
        # #     print('\tobject will be resized with scale ratio: {:f}'.format(_scale))
        # # else:
        # #     _scale = 1.0
        # _scale = TrainDataCfg.OBJECT_RESIZE_TH / float(_size)
        # scale_w = init_rect.w * _scale * self.search_patch_ratio
        # scale_h = init_rect.h * _scale * self.search_patch_ratio
        # _tmp = self.search_patch_ratio*self.extractor.get_resolution()
        # self.patch_scale_w = int(int(scale_w / float(_tmp) + 0.5) * _tmp)
        # self.patch_scale_h = int(int(scale_h / float(_tmp) + 0.5) * _tmp)
        # print('\tObject rescaled: width: {:d} height: {:d}, scale: {:f}'.format(self.patch_scale_w,
        #                                                                         self.patch_scale_h,
        #                                                                         _scale))
        # self.feature_size_w = round(self.patch_scale_w / self.extractor.get_resolution())
        # self.feature_size_h = round(self.patch_scale_h / self.extractor.get_resolution())
        #
        # self.response_sigma_x = self.feature_size_w * TrainDataCfg.RESPONSE_GAUSSIAN_SIGMA_RATIO
        # self.response_sigma_y = self.feature_size_h * TrainDataCfg.RESPONSE_GAUSSIAN_SIGMA_RATIO
        # self.motion_sigma_x = self.feature_size_w * TrainDataCfg.MOTION_GAUSSIAN_SIGMA_RATIO
        # self.motion_sigma_y = self.feature_size_h * TrainDataCfg.MOTION_GAUSSIAN_SIGMA_RATIO
        #
        # self._show_label_response_fid = TrainDataCfg.SHOW_LABEL_RESPONSE_FID
        # self._show_motion_map_fid = TrainDataCfg.SHOW_MOTION_MAP_FID

    def get_search_feature(self, image, object_rect):
        _search_ratio_w = self.feature_size_w / float(self.convolution_w)
        _search_ratio_h = self.feature_size_h / float(self.convolution_h)
        _search_rect = object_rect.get_copy().scale_from_center(_search_ratio_w,
                                                                _search_ratio_h)
        _search_bgr = clip_image(image, _search_rect)
        _search_input = cv2.resize(_search_bgr, (self.input_search_w, self.input_search_h))

        if self._show_search_bgr_fid:
            display.show_image(_search_bgr, self._show_search_bgr_fid)
        _search_feature = self.extractor.extract_feature(_search_input)
        return _search_rect, _search_bgr, _search_feature

    def get_scaled_search_feature(self, image, object_rect):
        _scale_step_w = max(1, round(object_rect.w * self.scale_ratio))
        _scale_step_h = max(1, round(object_rect.h * self.scale_ratio))
        scaled_object_rects = []
        for i in range(2 * self.scale_test_num + 1):
            w = object_rect.w + _scale_step_w * (i - self.scale_test_num)
            h = object_rect.h + _scale_step_h * (i - self.scale_test_num)
            if w < 3 or h < 3:
                print('Warning: w < 3 or h < 3')
                continue
            cx, cy = object_rect.get_center()
            tl_x = round(cx - (w - 1)/2.0)
            tl_y = round(cy - (h - 1)/2.0)
            _rect = Rect(tl_x, tl_y, w, h)
            scaled_object_rects.append(_rect)

        _search_ratio_w = self.feature_size_w / float(self.convolution_w)
        _search_ratio_h = self.feature_size_h / float(self.convolution_h)

        _search_rect_list = []
        _search_bgr_list = []
        _search_input_list = []
        for _scaled_rect in scaled_object_rects:
            _search_rect = _scaled_rect.get_copy().scale_from_center(_search_ratio_w,
                                                                     _search_ratio_h)
            _search_bgr = clip_image(image, _search_rect)
            _search_input = cv2.resize(_search_bgr, (self.input_search_w, self.input_search_h))

            _search_rect_list.append(_search_rect)
            _search_bgr_list.append(_search_bgr)
            _search_input_list.append(_search_input)
        if self._show_search_bgr_fid:
            display.show_image(_search_bgr_list[0], self._show_search_bgr_fid)

        _search_features = self.extractor.extract_multiple_features(_search_input_list)
        return _search_rect_list, _search_bgr_list, _search_features, scaled_object_rects

    def get_object_index_by_rect(self, search_rect, object_rect):
        dx = object_rect.get_center()[0] - search_rect.get_center()[0]
        dy = object_rect.get_center()[1] - search_rect.get_center()[1]
        _x_resolution = search_rect.w / float(self.feature_size_w)
        _y_resolution = search_rect.h / float(self.feature_size_h)
        dxi = round(float(dx) / _x_resolution)
        dyi = round(float(dy) / _y_resolution)
        xi = dxi + int((self.response_size_w - 1) / 2.0)
        yi = dyi + int((self.response_size_h - 1) / 2.0)
        assert 0 <= xi < self.response_size_w and 0 <= yi < self.response_size_h
        return yi, xi

    def get_label_response(self, obj_index_y, obj_index_x):
        assert 0 <= obj_index_x < self.response_size_w and 0 <= obj_index_y < self.response_size_h
        _x_index = np.arange(0, self.response_size_w)
        _y_index = np.arange(0, self.response_size_h)
        yv, xv = np.meshgrid(_y_index, _x_index, indexing='ij')
        yv -= obj_index_y
        xv -= obj_index_x
        _y1 = yv * yv / 2 / self.response_sigma_y / self.response_sigma_y
        _x1 = xv * xv / 2 / self.response_sigma_x / self.response_sigma_x
        response = np.exp(-(_y1 + _x1))
        # response[response < 1e-5] = 0.0
        if self._show_label_response_fid:
            display.show_map(response, self._show_label_response_fid)
        return response

    def get_motion_response(self, obj_index_y, obj_index_x):
        assert 0 <= obj_index_x < self.response_size_w and 0 <= obj_index_y < self.response_size_h
        _x_index = np.arange(0, self.response_size_w)
        _y_index = np.arange(0, self.response_size_h)
        yv, xv = np.meshgrid(_y_index, _x_index, indexing='ij')
        yv -= obj_index_y
        xv -= obj_index_x
        _y1 = yv * yv / 2 / self.motion_sigma / self.motion_sigma
        _x1 = xv * xv / 2 / self.motion_sigma / self.motion_sigma
        response = np.exp(-(_y1 + _x1))
        # response[response < 1e-5] = 0.0
        if self._show_motion_map_fid:
            display.show_map(response, self._show_motion_map_fid)
        return response

    def get_object_rect_by_index(self, search_rect, obj_index_y, obj_index_x):
        _yi, _xi = obj_index_y, obj_index_x

        _x_resolution = search_rect.w / float(self.feature_size_w)
        _y_resolution = search_rect.h / float(self.feature_size_h)

        _dyi = _yi - int((self.response_size_h-1) / 2.0)
        _dxi = _xi - int((self.response_size_w-1) / 2.0)

        patch_cx, patch_cy = search_rect.get_center()
        pd_cx, pd_cy = patch_cx + _dxi*_x_resolution, patch_cy + _dyi*_y_resolution
        _search_ratio_w = self.feature_size_w / float(self.convolution_w)
        _search_ratio_h = self.feature_size_h / float(self.convolution_h)
        pd_w, pd_h = round(search_rect.w/_search_ratio_w), round(search_rect.h/_search_ratio_h)

        pd_tlx = round(pd_cx - (pd_w-1)/2.0)
        pd_tly = round(pd_cy - (pd_h-1)/2.0)

        final_rect = Rect(pd_tlx, pd_tly, pd_w, pd_h)
        return final_rect

    # def generate_input_feature(self, image, patch_rect):
    #     patch = clip_image(image, patch_rect)
    #     if patch.shape[0] == self.patch_scale_h and patch.shape[1] == self.patch_scale_w:
    #         feature = self.extractor.extract_feature(patch)
    #     else:
    #         patch_scaled = cv2.resize(patch, (self.patch_scale_w, self.patch_scale_h))
    #         feature = self.extractor.extract_feature(patch_scaled)
    #     assert feature.shape[2] == self.extractor.get_channel_num()
    #
    #     return feature
    #
    # def generate_label_response(self, response_size, patch_rect, gt_rect):
    #     # response_size -> (h, w)
    #     dx = gt_rect.get_center()[0] - patch_rect.get_center()[0]
    #     dy = gt_rect.get_center()[1] - patch_rect.get_center()[1]
    #     _x_resolution = patch_rect.w / float(self.feature_size_w)
    #     _y_resolution = patch_rect.h / float(self.feature_size_h)
    #     dxi = round(float(dx) / _x_resolution)
    #     dyi = round(float(dy) / _y_resolution)
    #     xi = dxi + int((response_size[1]-1) / 2.0)
    #     yi = dyi + int((response_size[0]-1) / 2.0)
    #     assert 0 <= xi < response_size[1] and 0 <= yi < response_size[0]
    #
    #     _x_index = np.arange(0, response_size[1])
    #     _y_index = np.arange(0, response_size[0])
    #     yv, xv = np.meshgrid(_y_index, _x_index, indexing='ij')
    #     yv -= yi
    #     xv -= xi
    #     _y1 = yv * yv / 2 / self.response_sigma_y / self.response_sigma_y
    #     _x1 = xv * xv / 2 / self.response_sigma_x / self.response_sigma_x
    #     response = np.exp(-(_y1 + _x1))
    #     # response[response < 1e-5] = 0.0
    #     if self._show_label_response_fid:
    #         display.show_map(response, self._show_label_response_fid)
    #     return response
    #
    # def generate_motion_map(self, response_size, patch_rect, last_obj_rect):
    #     dx = last_obj_rect.get_center()[0] - patch_rect.get_center()[0]
    #     dy = last_obj_rect.get_center()[1] - patch_rect.get_center()[1]
    #     _x_resolution = patch_rect.w / self.feature_size_w
    #     _y_resolution = patch_rect.h / self.feature_size_h
    #     dxi = round(float(dx) / _x_resolution)
    #     dyi = round(float(dy) / _y_resolution)
    #     xi = dxi + int((response_size[1]-1) / 2.0)
    #     yi = dyi + int((response_size[0]-1) / 2.0)
    #     assert 0 <= xi < response_size[1] and 0 <= yi < response_size[0]
    #
    #     _x_index = np.arange(0, response_size[1])
    #     _y_index = np.arange(0, response_size[0])
    #     yv, xv = np.meshgrid(_y_index, _x_index, indexing='ij')
    #     yv -= yi
    #     xv -= xi
    #     _y1 = yv * yv / 2 / self.motion_sigma_y / self.motion_sigma_y
    #     _x1 = xv * xv / 2 / self.motion_sigma_x / self.motion_sigma_x
    #     response = np.exp(-(_y1 + _x1))
    #     # response[response < 1e-5] = 0.0
    #     if self._show_motion_map_fid:
    #         display.show_map(response, self._show_motion_map_fid)
    #     return response

    # def generate_train_data(self, image, gt_rect):
    #     patch_rect = gt_rect.get_copy().scale_from_center(self.search_patch_ratio)
    #     patch = image.clip(patch_rect)
    #     if patch.shape[1] == self.patch_scale_h and patch.shape[0] == self.patch_scale_w:
    #         feature = self.extractor.extract_feature(patch)
    #     else:
    #         patch_scaled = cv2.resize(patch, (self.patch_scale_w, self.patch_scale_h))
    #         feature = self.extractor.extract_feature(patch_scaled)
    #     assert feature.shape[2] == self.extractor.get_channel_num()
    #
    #     dx, dy = gt_rect.get_center() - patch_rect.get_center()
    #     _x_resolution = patch.shape[1] / float(feature.shape[1])
    #     _y_resolution = patch.shape[0] / float(feature.shape[0])
    #     dxi = math.floor(float(dx)/_x_resolution + 0.5)
    #     dyi = math.floor(float(dy)/_y_resolution + 0.5)
    #     xi = int(dxi + feature.shape[1] / 2.0)
    #     yi = int(dyi + feature.shape[0] / 2.0)
    #     assert 0 <= xi < feature.shape[0] and 0 <= yi < feature.shape[1]
    #
    #     _x_index = np.arange(0, feature.shape[1])
    #     _y_index = np.arange(0, feature.shape[0])
    #     yv, xv = np.meshgrid(_y_index, _x_index, indexing='ij')
    #     yv -= yi
    #     xv -= xi
    #     _y1 = yv*yv/2/self.response_sigma_y/self.response_sigma_y
    #     _x1 = xv*xv/2/self.response_sigma_x/self.response_sigma_x
    #     response = np.exp(-(_y1+_x1))
    #
    #     return TrainData(patch, patch_rect, gt_rect.get_copy(), feature, response)

    # def get_final_prediction(self, patch_rect, response_size, predict_index):
    #     # response_size: (h, w)  predict_index ( yi, xi)
    #     res_height, res_width = response_size
    #     _yi, _xi = predict_index
    #
    #     _x_resolution = patch_rect.w / float(self.feature_size_w)
    #     _y_resolution = patch_rect.h / float(self.feature_size_h)
    #
    #     _dyi = _yi - int((res_height-1) / 2.0)
    #     _dxi = _xi - int((res_width-1) / 2.0)
    #
    #     patch_cx, patch_cy = patch_rect.get_center()
    #     pd_cx, pd_cy = patch_cx + _dxi*_x_resolution, patch_cy + _dyi*_y_resolution
    #     pd_w, pd_h = int(patch_rect.w/self.search_patch_ratio), int(patch_rect.h/self.search_patch_ratio)
    #
    #     pd_tlx = round(pd_cx - (pd_w-1)/2.0)
    #     pd_tly = round(pd_cy - (pd_h-1)/2.0)
    #
    #     final_rect = Rect(pd_tlx, pd_tly, pd_w, pd_h)
    #     return final_rect


def _test_data_provider():
    patch_rect = Rect(0,0, 500, 500)
    gt_rect = Rect(152, 134, 42, 120)
    response_size = (32, 32)




if __name__ == '__main__':
    _test_data_provider()
