
import feature_extractor


class TrainDataCfg(object):
    SEARCH_PATCH_RATIO = 3.0
    RESPONSE_GAUSSIAN_SIGMA_RATIO = 0.03
    OBJECT_RESIZE_TH = 50

class FhogCfg(object):
    CELL_SIZE = 4
    BIN_NUM = 7


class ConvRegressionCfg(object):
    REGULARIZATION_COEF = 0
    SGD_LEARNING_RATE = 1e-12
    SGD_MOMENTUM = 0.0
    VERBOSE = True


class ConvRegTrackerCfg(object):
    FEATURE_EXTRACTOR = feature_extractor.RgbExtractor
    TRAIN_LOSS_TH = 1e-1
    TRAIN_MAX_STEP_NUM = 5000


class TestCfg(object):
    SEQUENCE_DIR = '/home/chenkai/workspace/tracker_benchmark_python/data/'

