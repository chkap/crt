from __future__ import print_function

import numpy as np

import tensorflow as tf

from conv_reg_config import ConvRegressionCfg
import display


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
        self._weight = None
        self._bias = None
        self.graph = None
        self.session = None

        self._loss_weight_a = ConvRegressionCfg.LOSS_WEIGHT_A
        self._loss_weight_b = ConvRegressionCfg.LOSS_WEIGHT_B
        self._loss_threshold = ConvRegressionCfg.LOSS_THRESHOLD
        self._pred_loss = None
        self._regu_loss = None
        self._total_loss = None

        self._pred_loss_list = list()
        self._regu_loss_list = list()
        self._total_loss_list = list()

        self._show_response_fid = ConvRegressionCfg.SHOW_RESPONSE_FID
        self._show_step = ConvRegressionCfg.SHOW_STEP

        input_size = init_features.shape
        input_mean = np.mean(init_features)
        self._build_graph(input_size, conv_size, input_mean)

    def _build_graph(self, input_size, conv_size, input_mean):
        assert len(input_size) == 4 and len(conv_size) == 2
        self.graph = tf.Graph()
        with self.graph.as_default():
            _input_shape = (None, input_size[1], input_size[2], input_size[3])
            self._input_holder = tf.placeholder(tf.float32, _input_shape, name='input_feature')
            _output_shape = (None, input_size[1]-conv_size[0]+1, input_size[2]-conv_size[1]+1, 1)
            self._response_holder = tf.placeholder(tf.float32, _output_shape, name='label_response')
            self._global_step = tf.Variable(0, trainable=False, name='global_step')

            _weight_shape = [conv_size[0], conv_size[1], input_size[3], 1]
            _weight_size = conv_size[0]*conv_size[1]*input_size[3]
            _weight_std = min(1/input_mean/_weight_size/4, 1)
            # _weight_init = tf.zeros(shape=_weight_shape, dtype=tf.float32)
            _weight_init = tf.random_normal(_weight_shape, stddev=_weight_std)
            self._weight = tf.Variable(_weight_init, name='conv_weight')
            self._bias = tf.Variable(0.0, name='conv_bias')

            _conv_out = tf.nn.conv2d(self._input_holder, self._weight, [1, 1, 1, 1], 'VALID')
            self._output_response = tf.add(_conv_out, self._bias)

            _weight_map = self._loss_weight_a * tf.exp(self._loss_weight_b*self._response_holder)
            _diff_map = self._output_response - self._response_holder
            _sign_map = (tf.sign(tf.abs(_diff_map) - self._loss_threshold) + 1) / 2
            _sum_map = tf.mul(tf.mul(_weight_map, _sign_map), _diff_map)
            _l2_loss = tf.reduce_sum(_sum_map * _sum_map, reduction_indices=[1,2,3])
            self._pred_loss = tf.reduce_mean(_l2_loss, reduction_indices=0)
            # self._pred_loss = tf.nn.l2_loss(_mean_loss, name='l2_loss')
            self._regu_loss = 0.5*self._regularization_coef * \
                              (tf.reduce_sum(self._weight*self._weight) + tf.mul(self._bias, self._bias))
            self._total_loss = self._pred_loss + self._regu_loss

            self._train_op = tf.train.AdamOptimizer(self._learning_rate) \
                .minimize(self._total_loss, global_step=self._global_step)
            self.session = tf.Session(graph=self.graph)
            self.session.run(tf.initialize_all_variables())

            tf.train.SummaryWriter('./log', graph=self.graph)

    def get_global_step(self):
        if self.session:
            global_step = self.session.run(self._global_step)
            return global_step
        else:
            return -1

    def train(self, features, response, max_step_num, loss_th):
        feed_dict = {self._input_holder: features,
                     self._response_holder: response}
        i = 0
        while i < max_step_num:
            if self._verbose:
                fetches = [self._train_op, self._pred_loss, self._regu_loss, self._total_loss, self._weight,
                           self._bias, self._output_response, self._global_step]
                _, pred_loss, regu_loss, total_loss, weight, bias, res, step = self.session.run(fetches, feed_dict=feed_dict)
                print('step:{:5d}, pred_loss:{:.4e}, regu_loss: {:.4e}, total_loss:{:.4e}'.format(step,
                                                                                                  pred_loss,
                                                                                                  regu_loss,
                                                                                                  total_loss))
                self._pred_loss_list.append(pred_loss)
                self._regu_loss_list.append(regu_loss)
                self._total_loss_list.append(total_loss)
                if step % self._show_step == 0 and self._show_response_fid:
                    display.show_map(res[0,:,:,0], self._show_response_fid)
            else:
                _, total_loss = self.session.run((self._train_op, self._total_loss), feed_dict=feed_dict)
            if total_loss < loss_th:
                break
            i += 1
        # if i >= max_step_num:
        #     print('Warning, total_loss larger than loss_th even after {:d}steps'.format(i))

    def inference(self, features):
        feed_dict = {self._input_holder: features}
        response = self.session.run(self._output_response, feed_dict=feed_dict)
        return response

    def close(self):
        if self.session is not None:
            self.session.close()
            self.session = None

