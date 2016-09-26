
import feature_extractor

class TrainDataCfg(object):
    SEARCH_PATCH_RATIO = 3.0
    RESPONSE_GAUSSIAN_SIGMA_RATIO = 0.03


class FhogCfg(object):
    CELL_SIZE = 4
    BIN_NUM = 7

class ConvRegressionCfg(object):
    REGULARIZATION_COEF = 0.0001
    SGD_LEARNING_RATE = 0.01
    SGD_MOMENTUM = 0.8
    VERBOSE = True


class ConvRegTrackerCfg(object):
    FEATURE_EXTRACTOR = feature_extractor.RgbExtractor

