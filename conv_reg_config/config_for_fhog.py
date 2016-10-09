
import feature_extractor


class TrainDataCfg(object):
    SEARCH_PATCH_RATIO = 3.0
    RESPONSE_GAUSSIAN_SIGMA_RATIO = 0.03
    MOTION_GAUSSIAN_SIGMA_RATIO = 0.06
    OBJECT_RESIZE_TH = 80
    SHOW_LABEL_RESPONSE_FID = ''  # 'label_response'


class FhogCfg(object):
    CELL_SIZE = 4
    BIN_NUM = 7


class ConvRegressionCfg(object):
    REGULARIZATION_COEF = 1e-4
    SGD_LEARNING_RATE = 1e-5
    SGD_MOMENTUM = 0.0
    LOSS_WEIGHT_A = 0.1
    LOSS_WEIGHT_B = 3
    LOSS_THRESHOLD = 0.03
    VERBOSE = True
    SHOW_RESPONSE_FID = ''  # 'output_response'
    SHOW_STEP = 20


class ConvRegTrackerCfg(object):
    FEATURE_EXTRACTOR = feature_extractor.FhogExtractor
    TRAIN_LOSS_TH = 0.1
    TRAIN_INIT_MAX_STEP_NUM = 2000
    TRAIN_UPDATE_MAX_STEP_NUM = 30
    SHOW_OVERALL_RESPONSE_FID = ''  # 'final response'
    UPDATE_CONFIDENCE_TH = 0.5


class TestCfg(object):
    SEQUENCE_DIR = '/home/chenkai/workspace/tracker_benchmark_python/data/'
    SHOW_TRACK_RESULT_FID = 'track results'

