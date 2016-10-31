
class TrainDataCfg(object):
    CONVOLUTION_SIZE_TH = 10
    SEARCH_RATIO_WIDTH = 9
    SEARCH_RATIO_HEIGHT = 5
    # SEARCH_PATCH_RATIO = 4

    SCALE_TEST_NUM = 1
    SCALE_RATIO = 0.02

    RESPONSE_GAUSSIAN_SIGMA_RATIO = 0.10
    MOTION_GAUSSIAN_SIGMA_RATIO = 0.6
    OBJECT_RESIZE_TH = 20

    SHOW_LABEL_RESPONSE_FID = ''  # 'label_response'
    SHOW_MOTION_MAP_FID = ''  # 'motion_map'
    SHOW_SEARCH_BGR_FID = ''  # 'search_bgr'


class FhogCfg(object):
    CELL_SIZE = 4
    BIN_NUM = 9


class FhogCnCfg(object):
    CELL_SIZE = 4
    BIN_NUM = 9


class ConvRegressionCfg(object):
    REGULARIZATION_COEF = 1e3
    SGD_LEARNING_RATE = 2e-8
    SGD_UPDATE_LEARNING_RATE = 5e-9
    SGD_MOMENTUM = 0.0
    LOSS_WEIGHT_A = 0.1
    LOSS_WEIGHT_B = 1.0
    LOSS_THRESHOLD = 0.1
    VERBOSE = False
    SHOW_RESPONSE_FID = 'output_response'
    SHOW_STEP = 5


class ConvRegTrackerCfg(object):
    TRAIN_LOSS_TH = 0.01
    TRAIN_INIT_MAX_STEP_NUM = 4000
    TRAIN_UPDATE_MAX_STEP_NUM = 15
    TRAIN_UPDATE_STEP_NUM = 2
    UPDATE_CONFIDENCE_TH = 0.2

    TRAIN_DATA_HISTORY_LENGTH = 5
    TRAIN_DATA_GAP = 2
    SHOW_OVERALL_RESPONSE_FID = ''  #'final response'


class TestCfg(object):
    SEQUENCE_DIR = '/home/chenkai/workspace/tracker_benchmark_python/data/'
    SHOW_TRACK_RESULT_FID = 'track results'

