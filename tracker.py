from __future__ import division

import numpy as np


from train_data_provider import TrainData, TrainDataProvider
from conv_reg_config import ConvRegTrackerCfg
from conv_reg import ConvRegression
import display
import feature_extractor


class TrackInfo(object):

    def __init__(self, patch_rect=None, patch_feature=None, obj_rect=None):
        self.patch_rect = patch_rect
        self.patch_feature = patch_feature
        self.obj_rect = obj_rect


class ConvRegTracker(object):

    def __init__(self):
        self.data_provider = None
        self.conv_regression = None
        self.feature_extractor = feature_extractor.FhogCnExtractor
        self._train_init_max_step_num = ConvRegTrackerCfg.TRAIN_INIT_MAX_STEP_NUM
        self._train_update_max_step_num = ConvRegTrackerCfg.TRAIN_UPDATE_MAX_STEP_NUM
        self._train_loss_th = ConvRegTrackerCfg.TRAIN_LOSS_TH
        self._show_final_response_fid = ConvRegTrackerCfg.SHOW_OVERALL_RESPONSE_FID
        self._update_confidence_th = ConvRegTrackerCfg.UPDATE_CONFIDENCE_TH
        self._last_rect = None

        self._frame_no = None
        self._track_info_list = list()

    def init(self, image, init_rect):
        if self.conv_regression is not None:
            self.conv_regression.close()
            self.conv_regression = None

        self._frame_no = 0
        self._track_info_list = list()

        self.data_provider = TrainDataProvider(self.feature_extractor, init_rect)
        patch_rect = init_rect.get_copy().scale_from_center(self.data_provider.search_patch_ratio,
                                                            self.data_provider.search_patch_ratio)
        feature = self.data_provider.generate_input_feature(image, patch_rect)
        assert feature.shape[0] == self.data_provider.feature_size_h and \
            feature.shape[1] == self.data_provider.feature_size_w
        feature_height = feature.shape[0]
        feature_width = feature.shape[1]
        conv_height = int(feature_height / self.data_provider.search_patch_ratio + 0.5)
        conv_width = int(feature_width / self.data_provider.search_patch_ratio + 0.5)
        assert abs(conv_height * self.data_provider.search_patch_ratio - feature_height) < 1 and \
               abs(conv_width * self.data_provider.search_patch_ratio - feature_width) < 1
        conv_size = [conv_height, conv_width]

        response_size = [feature.shape[0]-conv_height+1, feature.shape[1]-conv_width+1]
        response = self.data_provider.generate_label_response(response_size, patch_rect, init_rect)

        self.conv_regression = ConvRegression(feature[np.newaxis,:,:,:], conv_size)

        self.conv_regression.train(feature[np.newaxis,:,:,:],
                                   response[np.newaxis,:,:,np.newaxis],
                                   self._train_init_max_step_num,
                                   self._train_loss_th)

        self._last_rect = init_rect

        track_info = TrackInfo(patch_rect, feature, init_rect)
        self._track_info_list.append(track_info)

    def track(self, image):
        self._frame_no += 1
        last_rect = self._track_info_list[-1].obj_rect

        patch_rect = last_rect.get_copy().scale_from_center(self.data_provider.search_patch_ratio,
                                                            self.data_provider.search_patch_ratio)
        feature = self.data_provider.generate_input_feature(image, patch_rect)
        pred_response = self.conv_regression.inference(feature[np.newaxis,:,:,:])[0,:,:,0]
        pred_response_size = [pred_response.shape[0], pred_response.shape[1]]
        motion_map = self.data_provider.generate_motion_map(pred_response_size,
                                                            patch_rect,
                                                            last_rect)
        # display.show_map(motion_map, 'motion map')
        overall_response = motion_map*pred_response
        # overall_response = pred_response
        if self._show_final_response_fid:
            display.show_map(overall_response, self._show_final_response_fid)

        tmp = np.unravel_index([np.argmax(overall_response),], overall_response.shape)
        pred_index_y, pred_index_x = tmp[0][0], tmp[1][0]
        confidence = min(1.0, overall_response[pred_index_y, pred_index_x])

        pred_rect = self.data_provider.get_final_prediction(patch_rect,
                                                            pred_response_size,
                                                            [pred_index_y, pred_index_x])

        label_response = self.data_provider.generate_label_response(pred_response_size, patch_rect, pred_rect)

        if confidence > self._update_confidence_th:
            _update_step = confidence * self._train_update_max_step_num
            self.conv_regression.train(feature[np.newaxis,:,:,:],
                                       label_response[np.newaxis,:,:,np.newaxis],
                                       _update_step,
                                       self._train_loss_th)

        track_info = TrackInfo(patch_rect, feature, pred_rect)
        self._track_info_list.append(track_info)

        return pred_rect





