from __future__ import print_function

import numpy as np

import tensorflow as tf

from config import ConvRegressionCfg


class ConvRegression(object):

    def __init__(self, init_features, conv_size):
        self._regularization_coef = ConvRegressionCfg.REGULARIZATION_COEF
        self._learning_rate = ConvRegressionCfg.SGD_LEARNING_RATE
        self._momentum = ConvRegressionCfg.SGD_MOMENTUM
        self._verbose = ConvRegressionCfg.VERBOSE
        self._global_step = None
        self._input_holder = None
        self._response_holder = None
        self._output_response = None
        self.graph = None
        self.session = None

        self._pred_loss = None
        self._regu_loss = None
        self._total_loss = None

        self._pred_loss_list = list()
        self._regu_loss_list = list()
        self._total_loss_list = list()

        input_size = init_features.shape
        input_mean = np.mean(init_features)
        self._build_graph(input_size, conv_size, input_mean)

    def _build_graph(self, input_size, conv_size, input_mean):
        self.graph = tf.Graph()
        with self.graph.as_default():
            _input_shape = (None, input_size[0], input_size[1], input_size[2])
            self._input_holder = tf.placeholder(tf.float32, _input_shape, name='input_feature')
            _output_shape = (None, input_size[0], input_size[1], 1)
            self._response_holder = tf.placeholder(tf.float32, _output_shape, name='label_response')
            self._global_step = tf.Variable(0, trainable=False, name='global_step')

            _weight_shape = [conv_size[0], conv_size[1], input_size[2], 1]
            _weight_std = min(1/input_mean, 1)
            _weight_init = tf.random_normal(shape=_weight_shape, stddev=_weight_std)
            _weight = tf.Variable(_weight_init, name='conv_weight')
            _bias = tf.Variable(0.0, name='conv_bias')

            _conv_out = tf.nn.conv2d(self._input_holder, _weight, [1, 1, 1, 1], 'SAME')
            self._output_response = tf.add(_conv_out, _bias)

            _mean_loss = tf.reduce_mean(self._output_response - self._response_holder, reduction_indices=0)
            self._pred_loss = tf.nn.l2_loss(_mean_loss, name='l2_loss')
            self._regu_loss = 0.5*self._regularization_coef * (tf.nn.l2_loss(_weight) + tf.mul(_bias, _bias))
            self._total_loss = self._pred_loss + self._regu_loss

            self._train_op = tf.train.MomentumOptimizer(self._learning_rate, self._momentum)
            self.session = tf.Session(graph=self.graph)
            self.session.run(tf.initialize_all_variables())

    def train(self, features, response, step_num):
        feed_dict = {self._input_holder: features,
                     self._response_holder: response}
        i = 0
        while i < step_num:
            if self._verbose:
                fetches = [self._train_op, self._pred_loss, self._regu_loss, self._total_loss]
                _, pred_loss, regu_loss, total_loss = self.session.run(fetches, feed_dict=feed_dict)
                print('pred_loss:{:.4e}, regu_loss: {:.4e}, total_loss:{:.4e}'.format(pred_loss,
                                                                                      regu_loss,
                                                                                      total_loss))
                self._pred_loss_list.append(pred_loss)
                self._regu_loss_list.append(regu_loss)
                self._total_loss_list.append(total_loss)
            else:
                self.session.run(self._train_op, feed_dict=feed_dict)

            i += 1

    def inference(self, features):
        feed_dict = {self._input_holder: features}
        response = self.session.run(self._output_response, feed_dict=feed_dict)
        return response

    def close(self):
        if self.session is not None:
            self.session.close()
            self.session = None

