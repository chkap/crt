
import cv2
import fhog_feature
import cn_feature
import numpy as np
import conv_reg_config


class FeatureExtractor(object):

    def __init__(self):
        self._resolution = 1.0
        self._channel_num = 1

    def extract_multiple_features(self, input_images):
        pass

    def get_resolution(self):
        return self._resolution

    def get_channel_num(self):
        return self._channel_num


class GrayExtractor(FeatureExtractor):

    def extract_feature(self, input_image):
        gray = cv2.cvtColor(input_image, cv2.COLOR_RGB2GRAY)
        return gray


class RgbExtractor(FeatureExtractor):

    def __init__(self):
        super(RgbExtractor, self).__init__()
        self._channel_num = 3

    def extract_feature(self, input_image):
        return input_image


class FhogExtractor(FeatureExtractor):

    def __init__(self):
        super(FhogExtractor, self).__init__()
        self.cell_size = conv_reg_config.FhogCfg.CELL_SIZE
        self.bin_num = conv_reg_config.FhogCfg.BIN_NUM
        self._resolution = self.cell_size
        self._channel_num = 3*self.bin_num + 4

    def extract_feature(self, input_image):
        assert input_image.shape[2] == 3
        im = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)
        im = np.asarray(im, dtype=np.float32)
        return fhog_feature.extract(im, bin_size=self.cell_size, n_orients=self.bin_num)


class FhogCnExtractor(FeatureExtractor):

    def __init__(self):
        super(FhogCnExtractor, self).__init__()
        self.cell_size = conv_reg_config.FhogCnCfg.CELL_SIZE
        self.bin_num = conv_reg_config.FhogCnCfg.BIN_NUM
        self._resolution = self.cell_size
        self._channel_num = 3*self.bin_num + 4 + 10

    def extract_feature(self, input_image):
        assert input_image.shape[2] == 3
        im = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)
        im = np.asarray(im, dtype=np.float32)
        fhog = fhog_feature.extract(im, bin_size=self.cell_size, n_orients=self.bin_num)

        rescaled_im = cv2.resize(input_image, (fhog.shape[1], fhog.shape[0]))
        cn = cn_feature.extract(rescaled_im)

        res = np.concatenate((fhog, cn), axis=2)
        return res


