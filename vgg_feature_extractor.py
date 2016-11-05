import numpy as np
import cv2
import tensorflow as tf

from feature_extractor import FeatureExtractor
import display
from conv_reg_config import TrainDataCfg

VGG_MODEL_PATH = '/home/chenkai/workspace/caffe_model/vgg16_D/VGG_16_layers_py3.npz'
VGG_MEAN = [103.939, 116.779, 123.68]


class VggExtractor(FeatureExtractor):
    def __init__(self):
        super(VggExtractor, self).__init__()
        self._feature_width = 0
        self._feature_height = 0
        self._channel_num = 64
        self._resolution = 4
        self._feature_mean = TrainDataCfg.VGG_FEATURE_MEAN
        self._feature_std = TrainDataCfg.VGG_FEATURE_STD
        self._feature_offset = None
        self._feature_scale = None

        self._graph = None
        self._session = None
        self._input_holder = None
        self._output_feature = None

        self._use_pca = True
        self.pca = None
        self._load_data()

    def _build_network(self, input_height, input_width):
        pass

    def _load_data(self):
        pass

    def extract_multiple_features(self, input_images):
        # assert len(input_images) > 0
        # input_width = input_images[0].shape[1]
        # input_height = input_images[0].shape[0]
        #
        # if input_height != self._feature_height or input_width != self._feature_width:
        #     self._build_network(input_height, input_width)
        #
        # _merge_list = []
        # for image in input_images:
        #     _merge_list.append(image[np.newaxis, :, :, :])
        # merged = np.concatenate(_merge_list, axis=0)
        # feed_dict = {self._input_holder: merged}
        # output_feature = self._session.run(self._output_feature, feed_dict=feed_dict)
        # return output_feature
        assert len(input_images) > 0
        input_width = input_images[0].shape[1]
        input_height = input_images[0].shape[0]

        if input_height != self._feature_height or input_width != self._feature_width:
            self._build_network(input_height, input_width)
            self.pca = None
            self._feature_offset = None
            self._feature_scale = None

        _merge_list = []
        for image in input_images:
            _merge_list.append(image[np.newaxis, :, :, :])
        merged = np.concatenate(_merge_list, axis=0)
        feed_dict = {self._input_holder: merged}
        output_features = self._session.run(self._output_feature, feed_dict=feed_dict)

        if self._use_pca:
            assert self._channel_num < output_features.shape[3]
            if not self.pca:
                self.pca = FeatureReduction(output_features[0], self._channel_num)

            re_features = self.pca.project(output_features)
        else:
            assert self._channel_num == output_features.shape[3]
            re_features = output_features

        if not self._feature_offset or not self._feature_scale:
            _mean = np.mean(re_features)
            _std = np.std(re_features)
            self._feature_offset = self._feature_mean - _mean
            self._feature_scale = self._feature_std / _std

        normalized_feature = self._feature_scale * (re_features-self._feature_offset)

        # _mean = np.mean(normalized_feature)
        # _std = np.std(normalized_feature)
        # print('\tfeatures mean:{:.6e}, std:{:.6e}'.format(_mean, _std))
        # hist, bin_edges = np.histogram(normalized_feature, bins=100)
        # display.show_histogram(hist, bin_edges)
        return normalized_feature


class VggL1Extractor(VggExtractor):
    def __init__(self):
        super(VggL1Extractor, self).__init__()
        self._use_pca = False

    def _build_network(self, input_height, input_width):

        assert not input_height % self._resolution and not input_width % self._resolution
        if self._session:
            self._session.close()
        self._graph = tf.Graph()
        self._feature_height = input_height
        self._feature_width = input_width
        print('Starting building the network for h={:d} w={:d}'.format(input_height, input_width))
        with self._graph.as_default():
            _input_shape = (None, input_height, input_width, 3)
            self._input_holder = tf.placeholder(tf.float32, shape=_input_shape)
            _mean = tf.Variable(VGG_MEAN, trainable=False)
            _sub_mean = self._input_holder - _mean

            _conv_11_w = tf.Variable(self._conv_data_11_weights)
            _conv_11_b = tf.Variable(self._conv_data_11_bias)
            _conv_11_output = tf.nn.conv2d(_sub_mean, _conv_11_w, [1, 1, 1, 1], padding='SAME') + _conv_11_b
            _conv_11_act = tf.nn.relu(_conv_11_output)
            _max_pool_11_output = tf.nn.max_pool(_conv_11_act, (1, 2, 2, 1), (1, 2, 2, 1), padding='SAME')

            _conv_12_w = tf.Variable(self._conv_data_12_weights)
            _conv_12_b = tf.Variable(self._conv_data_12_bias)
            _conv_12_output = tf.nn.conv2d(_max_pool_11_output, _conv_12_w, [1, 1, 1, 1], padding='SAME') + _conv_12_b
            _conv_12_act = tf.nn.relu(_conv_12_output)
            _max_pool_12_output = tf.nn.max_pool(_conv_12_act, (1, 2, 2, 1), (1, 2, 2, 1), padding='SAME')

            self._output_feature = _max_pool_12_output
            self._session = tf.Session(graph=self._graph)
            self._session.run(tf.initialize_all_variables())

    def _load_data(self):
        with np.load(VGG_MODEL_PATH) as npz_file:
            self._conv_data_11_weights = npz_file['conv1_1/weights']
            self._conv_data_11_bias = npz_file['conv1_1/biases']
            self._conv_data_12_weights = npz_file['conv1_2/weights']
            self._conv_data_12_bias = npz_file['conv1_2/biases']
        print('VggL1 parameters loaded successfully!')

    # def extract_feature(self, input_image):
    #     input_width = input_image.shape[1]
    #     input_height = input_image.shape[0]
    #     assert input_image.shape[2] == 3
    #     if input_height != self._feature_height or input_width != self._feature_width:
    #         self._build_network(input_height, input_width)
    #     assert self._session
    #     feed_dict = {self._input_holder: input_image[np.newaxis,:,:,:]}
    #     output_feature = self._session.run(self._output_feature, feed_dict=feed_dict)
    #     return output_feature[0,:,:,:]


class FeatureReduction(object):
    def __init__(self, image_feature, max_components):
        assert image_feature.ndim == 3
        feature = np.reshape(image_feature, (-1, image_feature.shape[2]))
        _mean = np.mean(feature, axis=0, keepdims=True)
        self.mean, self.eigen_vecs = cv2.PCACompute(feature, _mean, maxComponents=max_components)
        print('\tPCA computed!')

    def project(self, images_features):
        assert images_features.ndim == 4
        data = np.reshape(images_features, (-1, images_features.shape[3]))
        coeffs = cv2.PCAProject(data, self.mean, self.eigen_vecs)
        re_shape = list(images_features.shape)
        re_shape[3] = len(self.eigen_vecs)
        re_features = np.reshape(coeffs, re_shape)
        return re_features


class VggL2Extractor(VggExtractor):
    def __init__(self):
        super(VggL2Extractor, self).__init__()

    def _build_network(self, input_height, input_width):

        assert not input_height % self._resolution and not input_width % self._resolution
        if self._session:
            self._session.close()
        self._graph = tf.Graph()
        self._feature_height = input_height
        self._feature_width = input_width
        print('Starting building the network for h={:d} w={:d}'.format(input_height, input_width))
        with self._graph.as_default():
            _input_shape = (None, input_height, input_width, 3)
            self._input_holder = tf.placeholder(tf.float32, shape=_input_shape)
            _mean = tf.Variable(VGG_MEAN, trainable=False)
            _sub_mean = self._input_holder - _mean

            _conv_11_w = tf.Variable(self._conv_data_11_weights)
            _conv_11_b = tf.Variable(self._conv_data_11_bias)
            _conv_11_output = tf.nn.conv2d(_sub_mean, _conv_11_w, [1, 1, 1, 1], padding='SAME') + _conv_11_b
            _conv_11_act = tf.nn.relu(_conv_11_output)

            _conv_12_w = tf.Variable(self._conv_data_12_weights)
            _conv_12_b = tf.Variable(self._conv_data_12_bias)
            _conv_12_output = tf.nn.conv2d(_conv_11_act, _conv_12_w, [1, 1, 1, 1], padding='SAME') + _conv_12_b
            _conv_12_act = tf.nn.relu(_conv_12_output)
            _max_pool_12_output = tf.nn.max_pool(_conv_12_act, (1, 2, 2, 1), (1, 2, 2, 1), padding='SAME')

            _conv_21_w = tf.Variable(self._conv_data_21_weights)
            _conv_21_b = tf.Variable(self._conv_data_21_bias)
            _conv_21_output = tf.nn.conv2d(_max_pool_12_output, _conv_21_w, (1, 1, 1, 1), padding='SAME') + _conv_21_b
            _conv_21_act = tf.nn.relu(_conv_21_output)

            _conv_22_w = tf.Variable(self._conv_data_22_weights)
            _conv_22_b = tf.Variable(self._conv_data_22_bias)
            _conv_22_output = tf.nn.conv2d(_conv_21_act, _conv_22_w, (1, 1, 1, 1), padding='SAME') + _conv_22_b
            _conv_22_act = tf.nn.relu(_conv_22_output)
            _max_pool_22_output = tf.nn.max_pool(_conv_22_act, (1, 2, 2, 1), (1, 2, 2, 1,), padding='SAME')

            self._output_feature = _max_pool_22_output
            self._session = tf.Session(graph=self._graph)
            self._session.run(tf.initialize_all_variables())

    def _load_data(self):
        with np.load(VGG_MODEL_PATH) as npz_file:
            self._conv_data_11_weights = npz_file['conv1_1/weights']
            self._conv_data_11_bias = npz_file['conv1_1/biases']
            self._conv_data_12_weights = npz_file['conv1_2/weights']
            self._conv_data_12_bias = npz_file['conv1_2/biases']
            self._conv_data_21_weights = npz_file['conv2_1/weights']
            self._conv_data_21_bias = npz_file['conv2_1/biases']
            self._conv_data_22_weights = npz_file['conv2_2/weights']
            self._conv_data_22_bias = npz_file['conv2_2/biases']
        print('VggL2 parameters loaded successfully!')


class VggL3Extractor(VggExtractor):
    def __init__(self):
        super(VggL3Extractor, self).__init__()

    def _build_network(self, input_height, input_width):
        assert not input_height % self._resolution and not input_width % self._resolution
        if self._session:
            self._session.close()
        self._graph = tf.Graph()
        self._feature_height = input_height
        self._feature_width = input_width
        print('Starting building the network for h={:d} w={:d}'.format(input_height, input_width))
        with self._graph.as_default():
            _input_shape = (None, input_height, input_width, 3)
            self._input_holder = tf.placeholder(tf.float32, shape=_input_shape)
            _mean = tf.Variable(VGG_MEAN, trainable=False)
            _sub_mean = self._input_holder - _mean

            _conv_11_w = tf.Variable(self._conv_data_11_weights)
            _conv_11_b = tf.Variable(self._conv_data_11_bias)
            _conv_11_output = tf.nn.conv2d(_sub_mean, _conv_11_w, [1, 1, 1, 1], padding='SAME') + _conv_11_b
            _conv_11_act = tf.nn.relu(_conv_11_output)

            _conv_12_w = tf.Variable(self._conv_data_12_weights)
            _conv_12_b = tf.Variable(self._conv_data_12_bias)
            _conv_12_output = tf.nn.conv2d(_conv_11_act, _conv_12_w, [1, 1, 1, 1], padding='SAME') + _conv_12_b
            _conv_12_act = tf.nn.relu(_conv_12_output)
            _max_pool_12_output = tf.nn.max_pool(_conv_12_act, (1, 2, 2, 1), (1, 2, 2, 1), padding='SAME')

            _conv_21_w = tf.Variable(self._conv_data_21_weights)
            _conv_21_b = tf.Variable(self._conv_data_21_bias)
            _conv_21_output = tf.nn.conv2d(_max_pool_12_output, _conv_21_w, (1, 1, 1, 1), padding='SAME') + _conv_21_b
            _conv_21_act = tf.nn.relu(_conv_21_output)

            _conv_22_w = tf.Variable(self._conv_data_22_weights)
            _conv_22_b = tf.Variable(self._conv_data_22_bias)
            _conv_22_output = tf.nn.conv2d(_conv_21_act, _conv_22_w, (1, 1, 1, 1), padding='SAME') + _conv_22_b
            _conv_22_act = tf.nn.relu(_conv_22_output)
            _max_pool_22_output = tf.nn.max_pool(_conv_22_act, (1, 2, 2, 1), (1, 2, 2, 1,), padding='SAME')

            _conv_31_w = tf.Variable(self._conv_data_31_weights)
            _conv_31_b = tf.Variable(self._conv_data_31_bias)
            _conv_31_output = tf.nn.conv2d(_max_pool_22_output, _conv_31_w, (1, 1, 1, 1), padding='SAME') + \
                              _conv_31_b
            _conv_31_act = tf.nn.relu(_conv_31_output)

            _conv_32_w = tf.Variable(self._conv_data_32_weights)
            _conv_32_b = tf.Variable(self._conv_data_31_bias)
            _conv_32_output = tf.nn.conv2d(_conv_31_act, _conv_32_w, (1, 1, 1, 1), padding='SAME') + _conv_32_b
            _conv_32_act = tf.nn.relu(_conv_32_output)

            _conv_33_w = tf.Variable(self._conv_data_33_weights)
            _conv_33_b = tf.Variable(self._conv_data_33_bias)
            _conv_33_output = tf.nn.conv2d(_conv_32_act, _conv_33_w, (1, 1, 1, 1), padding='SAME') + _conv_33_b
            _conv_33_act = tf.nn.relu(_conv_33_output)

            self._output_feature = _conv_33_act
            self._session = tf.Session(graph=self._graph)
            self._session.run(tf.initialize_all_variables())

    def _load_data(self):
        with np.load(VGG_MODEL_PATH) as npz_file:
            self._conv_data_11_weights = npz_file['conv1_1/weights']
            self._conv_data_11_bias = npz_file['conv1_1/biases']
            self._conv_data_12_weights = npz_file['conv1_2/weights']
            self._conv_data_12_bias = npz_file['conv1_2/biases']
            self._conv_data_21_weights = npz_file['conv2_1/weights']
            self._conv_data_21_bias = npz_file['conv2_1/biases']
            self._conv_data_22_weights = npz_file['conv2_2/weights']
            self._conv_data_22_bias = npz_file['conv2_2/biases']
            self._conv_data_31_weights = npz_file['conv3_1/weights']
            self._conv_data_31_bias = npz_file['conv3_1/biases']
            self._conv_data_32_weights = npz_file['conv3_2/weights']
            self._conv_data_32_bias = npz_file['conv3_2/biases']
            self._conv_data_33_weights = npz_file['conv3_3/weights']
            self._conv_data_33_bias = npz_file['conv3_3/biases']
        print('VggL3 parameters loaded successfully!')


class VggL4Extractor(VggExtractor):

    def __init__(self):
        super(VggL4Extractor, self).__init__()

    def _build_network(self, input_height, input_width):
        assert not input_height % self._resolution and not input_width % self._resolution
        if self._session:
            self._session.close()
        self._graph = tf.Graph()
        self._feature_height = input_height
        self._feature_width = input_width
        print('Starting building the network for h={:d} w={:d}'.format(input_height, input_width))
        with self._graph.as_default():
            _input_shape = (None, input_height, input_width, 3)
            self._input_holder = tf.placeholder(tf.float32, shape=_input_shape)
            _mean = tf.Variable(VGG_MEAN, trainable=False)
            _sub_mean = self._input_holder - _mean

            _conv_11_w = tf.Variable(self._conv_data_11_weights)
            _conv_11_b = tf.Variable(self._conv_data_11_bias)
            _conv_11_output = tf.nn.conv2d(_sub_mean, _conv_11_w, [1, 1, 1, 1], padding='SAME') + _conv_11_b
            _conv_11_act = tf.nn.relu(_conv_11_output)

            _conv_12_w = tf.Variable(self._conv_data_12_weights)
            _conv_12_b = tf.Variable(self._conv_data_12_bias)
            _conv_12_output = tf.nn.conv2d(_conv_11_act, _conv_12_w, [1, 1, 1, 1], padding='SAME') + _conv_12_b
            _conv_12_act = tf.nn.relu(_conv_12_output)
            _max_pool_12_output = tf.nn.max_pool(_conv_12_act, (1, 2, 2, 1), (1, 2, 2, 1), padding='SAME')

            _conv_21_w = tf.Variable(self._conv_data_21_weights)
            _conv_21_b = tf.Variable(self._conv_data_21_bias)
            _conv_21_output = tf.nn.conv2d(_max_pool_12_output, _conv_21_w, (1, 1, 1, 1), padding='SAME') + _conv_21_b
            _conv_21_act = tf.nn.relu(_conv_21_output)

            _conv_22_w = tf.Variable(self._conv_data_22_weights)
            _conv_22_b = tf.Variable(self._conv_data_22_bias)
            _conv_22_output = tf.nn.conv2d(_conv_21_act, _conv_22_w, (1, 1, 1, 1), padding='SAME') + _conv_22_b
            _conv_22_act = tf.nn.relu(_conv_22_output)
            _max_pool_22_output = tf.nn.max_pool(_conv_22_act, (1, 2, 2, 1), (1, 2, 2, 1,), padding='SAME')

            _conv_31_w = tf.Variable(self._conv_data_31_weights)
            _conv_31_b = tf.Variable(self._conv_data_31_bias)
            _conv_31_output = tf.nn.conv2d(_max_pool_22_output, _conv_31_w, (1, 1, 1, 1), padding='SAME') + \
                              _conv_31_b
            _conv_31_act = tf.nn.relu(_conv_31_output)

            _conv_32_w = tf.Variable(self._conv_data_32_weights)
            _conv_32_b = tf.Variable(self._conv_data_31_bias)
            _conv_32_output = tf.nn.conv2d(_conv_31_act, _conv_32_w, (1, 1, 1, 1), padding='SAME') + _conv_32_b
            _conv_32_act = tf.nn.relu(_conv_32_output)

            _conv_33_w = tf.Variable(self._conv_data_33_weights)
            _conv_33_b = tf.Variable(self._conv_data_33_bias)
            _conv_33_output = tf.nn.conv2d(_conv_32_act, _conv_33_w, (1, 1, 1, 1), padding='SAME') + _conv_33_b
            _conv_33_act = tf.nn.relu(_conv_33_output)

            _conv_41_w = tf.Variable(self._conv_data_41_weights)
            _conv_41_b = tf.Variable(self._conv_data_41_bias)
            _conv_41_output = tf.nn.conv2d(_conv_33_act, _conv_41_w, (1, 1, 1, 1), padding='SAME') + \
                              _conv_41_b
            _conv_41_act = tf.nn.relu(_conv_41_output)

            _conv_42_w = tf.Variable(self._conv_data_42_weights)
            _conv_42_b = tf.Variable(self._conv_data_42_bias)
            _conv_42_output = tf.nn.conv2d(_conv_41_act, _conv_42_w, (1,1,1,1), padding='SAME') + _conv_42_b
            _conv_42_act = tf.nn.relu(_conv_42_output)

            _conv_43_w = tf.Variable(self._conv_data_43_weights)
            _conv_43_b = tf.Variable(self._conv_data_43_bias)
            _conv_43_output = tf.nn.conv2d(_conv_42_act, _conv_43_w, (1,1,1,1), padding='SAME') + _conv_43_b
            _conv_43_act = tf.nn.relu(_conv_43_output)

            self._output_feature = _conv_43_act
            self._session = tf.Session(graph=self._graph)
            self._session.run(tf.initialize_all_variables())

    def _load_data(self):
        with np.load(VGG_MODEL_PATH) as npz_file:
            self._conv_data_11_weights = npz_file['conv1_1/weights']
            self._conv_data_11_bias = npz_file['conv1_1/biases']
            self._conv_data_12_weights = npz_file['conv1_2/weights']
            self._conv_data_12_bias = npz_file['conv1_2/biases']
            self._conv_data_21_weights = npz_file['conv2_1/weights']
            self._conv_data_21_bias = npz_file['conv2_1/biases']
            self._conv_data_22_weights = npz_file['conv2_2/weights']
            self._conv_data_22_bias = npz_file['conv2_2/biases']
            self._conv_data_31_weights = npz_file['conv3_1/weights']
            self._conv_data_31_bias = npz_file['conv3_1/biases']
            self._conv_data_32_weights = npz_file['conv3_2/weights']
            self._conv_data_32_bias = npz_file['conv3_2/biases']
            self._conv_data_33_weights = npz_file['conv3_3/weights']
            self._conv_data_33_bias = npz_file['conv3_3/biases']
            self._conv_data_41_weights = npz_file['conv4_1/weights']
            self._conv_data_41_bias = npz_file['conv4_1/biases']
            self._conv_data_42_weights = npz_file['conv4_2/weights']
            self._conv_data_42_bias = npz_file['conv4_2/biases']
            self._conv_data_43_weights = npz_file['conv4_3/weights']
            self._conv_data_43_bias = npz_file['conv4_3/biases']
        print('VggL4 parameters loaded successfully!')


class VggL5Extractor(VggExtractor):

    def __init__(self):
        super(VggL5Extractor, self).__init__()

    def _build_network(self, input_height, input_width):
        assert not input_height % self._resolution and not input_width % self._resolution
        if self._session:
            self._session.close()
        self._graph = tf.Graph()
        self._feature_height = input_height
        self._feature_width = input_width
        print('Starting building the network for h={:d} w={:d}'.format(input_height, input_width))
        with self._graph.as_default():
            _input_shape = (None, input_height, input_width, 3)
            self._input_holder = tf.placeholder(tf.float32, shape=_input_shape)
            _mean = tf.Variable(VGG_MEAN, trainable=False)
            _sub_mean = self._input_holder - _mean

            _conv_11_w = tf.Variable(self._conv_data_11_weights)
            _conv_11_b = tf.Variable(self._conv_data_11_bias)
            _conv_11_output = tf.nn.conv2d(_sub_mean, _conv_11_w, [1, 1, 1, 1], padding='SAME') + _conv_11_b
            _conv_11_act = tf.nn.relu(_conv_11_output)

            _conv_12_w = tf.Variable(self._conv_data_12_weights)
            _conv_12_b = tf.Variable(self._conv_data_12_bias)
            _conv_12_output = tf.nn.conv2d(_conv_11_act, _conv_12_w, [1, 1, 1, 1], padding='SAME') + _conv_12_b
            _conv_12_act = tf.nn.relu(_conv_12_output)
            _max_pool_12_output = tf.nn.max_pool(_conv_12_act, (1, 2, 2, 1), (1, 2, 2, 1), padding='SAME')

            _conv_21_w = tf.Variable(self._conv_data_21_weights)
            _conv_21_b = tf.Variable(self._conv_data_21_bias)
            _conv_21_output = tf.nn.conv2d(_max_pool_12_output, _conv_21_w, (1, 1, 1, 1), padding='SAME') + _conv_21_b
            _conv_21_act = tf.nn.relu(_conv_21_output)

            _conv_22_w = tf.Variable(self._conv_data_22_weights)
            _conv_22_b = tf.Variable(self._conv_data_22_bias)
            _conv_22_output = tf.nn.conv2d(_conv_21_act, _conv_22_w, (1, 1, 1, 1), padding='SAME') + _conv_22_b
            _conv_22_act = tf.nn.relu(_conv_22_output)
            _max_pool_22_output = tf.nn.max_pool(_conv_22_act, (1, 2, 2, 1), (1, 2, 2, 1,), padding='SAME')

            _conv_31_w = tf.Variable(self._conv_data_31_weights)
            _conv_31_b = tf.Variable(self._conv_data_31_bias)
            _conv_31_output = tf.nn.conv2d(_max_pool_22_output, _conv_31_w, (1, 1, 1, 1), padding='SAME') + \
                              _conv_31_b
            _conv_31_act = tf.nn.relu(_conv_31_output)

            _conv_32_w = tf.Variable(self._conv_data_32_weights)
            _conv_32_b = tf.Variable(self._conv_data_31_bias)
            _conv_32_output = tf.nn.conv2d(_conv_31_act, _conv_32_w, (1, 1, 1, 1), padding='SAME') + _conv_32_b
            _conv_32_act = tf.nn.relu(_conv_32_output)

            _conv_33_w = tf.Variable(self._conv_data_33_weights)
            _conv_33_b = tf.Variable(self._conv_data_33_bias)
            _conv_33_output = tf.nn.conv2d(_conv_32_act, _conv_33_w, (1, 1, 1, 1), padding='SAME') + _conv_33_b
            _conv_33_act = tf.nn.relu(_conv_33_output)

            _conv_41_w = tf.Variable(self._conv_data_41_weights)
            _conv_41_b = tf.Variable(self._conv_data_41_bias)
            _conv_41_output = tf.nn.conv2d(_conv_33_act, _conv_41_w, (1, 1, 1, 1), padding='SAME') + \
                              _conv_41_b
            _conv_41_act = tf.nn.relu(_conv_41_output)

            _conv_42_w = tf.Variable(self._conv_data_42_weights)
            _conv_42_b = tf.Variable(self._conv_data_42_bias)
            _conv_42_output = tf.nn.conv2d(_conv_41_act, _conv_42_w, (1, 1, 1, 1), padding='SAME') + _conv_42_b
            _conv_42_act = tf.nn.relu(_conv_42_output)

            _conv_43_w = tf.Variable(self._conv_data_43_weights)
            _conv_43_b = tf.Variable(self._conv_data_43_bias)
            _conv_43_output = tf.nn.conv2d(_conv_42_act, _conv_43_w, (1, 1, 1, 1), padding='SAME') + _conv_43_b
            _conv_43_act = tf.nn.relu(_conv_43_output)

            _conv_51_w = tf.Variable(self._conv_data_51_weights)
            _conv_51_b = tf.Variable(self._conv_data_51_bias)
            _conv_51_output = tf.nn.conv2d(_conv_43_act, _conv_51_w, (1,1,1,1), padding='SAME') + _conv_51_b
            _conv_51_act = tf.nn.relu(_conv_51_output)

            _conv_52_w = tf.Variable(self._conv_data_52_weights)
            _conv_52_b = tf.Variable(self._conv_data_52_bias)
            _conv_52_output = tf.nn.conv2d(_conv_51_act, _conv_52_w, (1,1,1,1), padding='SAME') + _conv_52_b
            _conv_52_act = tf.nn.relu(_conv_52_output)

            _conv_53_w = tf.Variable(self._conv_data_53_weights)
            _conv_53_b = tf.Variable(self._conv_data_53_bias)
            _conv_53_output = tf.nn.conv2d(_conv_52_act, _conv_53_w, (1,1,1,1), padding='SAME') + _conv_53_b
            _conv_53_act = tf.nn.relu(_conv_53_output)

            self._output_feature = _conv_53_act
            self._session = tf.Session(graph=self._graph)
            self._session.run(tf.initialize_all_variables())

    def _load_data(self):
        with np.load(VGG_MODEL_PATH) as npz_file:
            self._conv_data_11_weights = npz_file['conv1_1/weights']
            self._conv_data_11_bias = npz_file['conv1_1/biases']
            self._conv_data_12_weights = npz_file['conv1_2/weights']
            self._conv_data_12_bias = npz_file['conv1_2/biases']
            self._conv_data_21_weights = npz_file['conv2_1/weights']
            self._conv_data_21_bias = npz_file['conv2_1/biases']
            self._conv_data_22_weights = npz_file['conv2_2/weights']
            self._conv_data_22_bias = npz_file['conv2_2/biases']
            self._conv_data_31_weights = npz_file['conv3_1/weights']
            self._conv_data_31_bias = npz_file['conv3_1/biases']
            self._conv_data_32_weights = npz_file['conv3_2/weights']
            self._conv_data_32_bias = npz_file['conv3_2/biases']
            self._conv_data_33_weights = npz_file['conv3_3/weights']
            self._conv_data_33_bias = npz_file['conv3_3/biases']
            self._conv_data_41_weights = npz_file['conv4_1/weights']
            self._conv_data_41_bias = npz_file['conv4_1/biases']
            self._conv_data_42_weights = npz_file['conv4_2/weights']
            self._conv_data_42_bias = npz_file['conv4_2/biases']
            self._conv_data_43_weights = npz_file['conv4_3/weights']
            self._conv_data_43_bias = npz_file['conv4_3/biases']
            self._conv_data_51_weights = npz_file['conv5_1/weights']
            self._conv_data_51_bias = npz_file['conv5_1/biases']
            self._conv_data_52_weights = npz_file['conv5_2/weights']
            self._conv_data_52_bias = npz_file['conv5_2/biases']
            self._conv_data_53_weights = npz_file['conv5_3/weights']
            self._conv_data_53_bias = npz_file['conv5_3/biases']
        print('VggL5 parameters loaded successfully!')


def _test_load_data():
    ext = VggL1Extractor()
    # test_image = np.random.randint(0,255, (210, 30, 3), dtype=np.uint8)
    # ext.extract_feature(test_image)


if __name__ == '__main__':
    _test_load_data()

