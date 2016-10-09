
import cv2
import fhog
import conv_reg_config


class FeatureExtractor(object):

    def __init__(self):
        self._resolution = 1.0
        self._channel_num = 1

    def extract_feature(self, input_image):
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
        return fhog.get_fhog(input_image, self.cell_size, self.bin_num)


