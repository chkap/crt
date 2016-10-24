
import numpy as np
import tensorflow as tf

from feature_extractor import FeatureExtractor


class VggL1Extractor(FeatureExtractor):
    VGG_MODEL_PATH = '/home/chenkai/workspace/caffe_model/vgg16_D/VGG_16_layers_py3.npz'
    VGG_MEAN = [103.939, 116.779, 123.68]

    def __init__(self):
        super(VggL1Extractor, self).__init__()
        self._feature_width = 0
        self._feature_height = 0
        self._output_channel_size = 64
        self._resolution = 1

        self._graph = None
        self._session = None
        self._input_holder = None
        self._output_feature = None
        self._conv_data_11_weights = None
        self._conv_data_11_bias = None
        self._conv_data_12_weights = None
        self._conv_data_12_bias = None

        self._load_data()

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
            _mean = tf.Variable(self.VGG_MEAN, trainable=False)
            _sub_mean = self._input_holder - _mean

            _conv_11_w = tf.Variable(self._conv_data_11_weights)
            _conv_11_b = tf.Variable(self._conv_data_11_bias)
            _conv_11_output = tf.nn.conv2d(_sub_mean, _conv_11_w, [1,1,1,1], padding='SAME') + _conv_11_b
            _conv_11_act = tf.nn.relu(_conv_11_output)

            _conv_12_w = tf.Variable(self._conv_data_12_weights)
            _conv_12_b = tf.Variable(self._conv_data_12_bias)
            _conv_12_output = tf.nn.conv2d(_conv_11_act, _conv_12_w, [1,1,1,1], padding='SAME') + _conv_12_b
            _conv_12_act = tf.nn.relu(_conv_12_output)

            self._output_feature = _conv_12_act
            self._session = tf.Session(graph=self._graph)
            self._session.run(tf.initialize_all_variables())

    def _load_data(self):
        with np.load(self.VGG_MODEL_PATH) as npz_file:
            self._conv_data_11_weights = npz_file['conv1_1/weights']
            self._conv_data_11_bias = npz_file['conv1_1/biases']
            self._conv_data_12_weights = npz_file['conv1_2/weights']
            self._conv_data_12_bias = npz_file['conv1_2/biases']
        print('CNN parameters loaded successfully!')

    def get_channel_num(self):
        return self._output_channel_size

    def get_resolution(self):
        return self._resolution

    def extract_feature(self, input_image):
        input_width = input_image.shape[1]
        input_height = input_image.shape[0]
        assert input_image.shape[2] == 3
        if input_height != self._feature_height or input_width != self._feature_width:
            self._build_network(input_height, input_width)
        assert self._session
        feed_dict = {self._input_holder: input_image[np.newaxis,:,:,:]}
        output_feature = self._session.run(self._output_feature, feed_dict=feed_dict)
        return output_feature[0,:,:,:]

    def extract_multiple_features(self, images_list):
        assert len(images_list)> 0
        input_width = images_list[0].shape[1]
        input_height = images_list[0].shape[0]

        if input_height != self._feature_height or input_width != self._feature_width:
            self._build_network(input_height, input_width)

        _merge_list = []
        for image in images_list:
            _merge_list.append(image[np.newaxis, :, :, :])
        merged = np.concatenate(_merge_list)
        feed_dict = {self._input_holder: merged}
        output_feature = self._session.run(self._output_feature, feed_dict=feed_dict)
        return output_feature


def _test_load_data():
    ext = VggL1Extractor()
    # test_image = np.random.randint(0,255, (210, 30, 3), dtype=np.uint8)
    # ext.extract_feature(test_image)

if __name__ == '__main__':
    _test_load_data()

