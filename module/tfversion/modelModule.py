
from typing import Union, List
import tensorflow as tf


class ModelModule:
    def __init__(self, inputs: Union[tf.Tensor, List[tf.Tensor]], outputs: Union[tf.Tensor, List[tf.Tensor]],
                 standard_outputs: Union[tf.Tensor, List[tf.Tensor]], loss: tf.Tensor,
                 optimizer: Union[tf.train.Optimizer, tf.keras.optimizers.Optimizer] = tf.keras.optimizers.Adam(0.001),
                 net_configs: Union[tf.Tensor, List[tf.Tensor]] = None, model_save_path: str = None,
                 metrics: Union[tf.Tensor, List[tf.Tensor]] = None):
        '''
        :param inputs:  the model inputs, a tensor or tensor list
        :param outputs:  the model outputs, a tensor or tensor list, usually call it predict
        :param standard_outputs: the model standard outputs, a tensor or tensor list, usually call it y
        :param loss:  the model loss, for model train, a tensor
        :param optimizer: the model optimizer with parameters, like adam(learning_rate=0.001)
        :param net_configs:  the model other net configs with tensor that should be feed by user
        :param model_save_path: the model path for save model
        :param metrics:  the model metrics, like accuracy, MSE and so on
        '''
        self._inputs = inputs
        self._outputs = outputs
        self._standard_outputs = standard_outputs
        self._loss = loss
        self._optimizer = optimizer
        # self._learning_rate = learning_rate
        self._net_configs = net_configs
        self._metrics = metrics
        self._model_save_path = model_save_path
        self._train_ops = optimizer.minimize(loss)

    @property
    def train_ops(self):
        return self._train_ops

    @property
    def inputs(self):
        return self._inputs

    @property
    def outputs(self):
        return self._outputs

    @property
    def standard_outputs(self):
        return self._standard_outputs

    @property
    def loss(self):
        return self._loss

    @property
    def optimizer(self):
        return self._optimizer

    # @property
    # def learning_rate(self):
    #     return self._learning_rate

    @property
    def net_configs(self):
        return self._net_configs

    @property
    def model_save_path(self):
        return self._model_save_path

    @property
    def metrics(self):
        return self._metrics

    def batch_fit(self, sess: tf.Session, tr_inputs_feed, tr_outputs_feed, tr_net_configs_feed=None,
                  v_inputs_feed=None, v_outputs_feed=None, v_net_configs_feed=None):
        '''

        :param sess:  a tf.Session for train
        :param tr_inputs_feed:  train inputs feed value with the same sort in self.inputs
        :param tr_outputs_feed:  train standard outputs feed value with the same sort in self.standard_outputs
        :param tr_net_configs_feed:  train net configs feed value with the same sort in self.net_configs
        :param v_inputs_feed:  same with tr_inputs_feed ,but for validation
        :param v_outputs_feed: same with tr_outputs_feed ,but for validation
        :param v_net_configs_feed: same with tr_net_configs_feed ,but for validation
        :return:
            the self.metrics value with the feed value, wrapped with a dict, the train metrics value's key is 'tr_metrics'
            the validation if exist,the validation metrics value's key is 'v_metrics'
        '''
        result = {}
        feed = {}
        try:
            if isinstance(self.inputs, list):
                for i in range(len(self.inputs)):
                    feed[self.inputs[i]] = tr_inputs_feed[i]
            else:
                feed[self.inputs] = tr_inputs_feed
        except:
            raise RuntimeError("train inputs feed Error, maybe the len is not equal")
        try:
            if isinstance(self.standard_outputs, list):
                for i in range(len(self.standard_outputs)):
                    feed[self.standard_outputs[i]] = tr_outputs_feed[i]
            else:
                feed[self.standard_outputs] = tr_outputs_feed
        except:
            raise RuntimeError("train outputs feed Error, maybe the len is not equal")
        try:
            if self.net_configs is not None:
                if isinstance(self.net_configs, list):
                    for i in range(len(self.net_configs)):
                        feed[self.net_configs[i]] = tr_net_configs_feed[i]
                else:
                    feed[self.net_configs] = tr_net_configs_feed
        except:
            raise RuntimeError("train net configs feed Error, maybe the len is not equal or not provide")
        # print(feed)
        _, tr_metrics = sess.run([self.train_ops, self.metrics], feed_dict=feed)
        result["tr_metrics"] = tr_metrics
        if v_inputs_feed is not None and v_outputs_feed is not None:
            feed = {}
            try:
                if isinstance(self.inputs, list):
                    for i in range(len(self.inputs)):
                        feed[self.inputs[i]] = v_inputs_feed[i]
                else:
                    feed[self.inputs] = v_inputs_feed
            except:
                raise RuntimeError("validation inputs feed Error, maybe the len is not equal")
            try:
                if isinstance(self.standard_outputs, list):
                    for i in range(len(self.standard_outputs)):
                        feed[self.standard_outputs[i]] = v_outputs_feed[i]
                else:
                    feed[self.standard_outputs] = v_outputs_feed
            except:
                raise RuntimeError("validation outputs feed Error, maybe the len is not equal")
            try:
                if self.net_configs is not None:
                    if v_net_configs_feed is None:
                        raise RuntimeError("validation net config not provide!")
                    if isinstance(self.net_configs, list):
                        for i in range(len(self.net_configs)):
                            feed[self.net_configs[i]] = v_net_configs_feed[i]
                    else:
                        feed[self.net_configs] = v_net_configs_feed
            except:
                raise RuntimeError("validation net configs feed Error, maybe the len is not equal or not provide")
            v_metrics = sess.run(self.metrics,feed_dict=feed)
            result["v_metrics"] = v_metrics
        return result

    def evaluation(self, sess: tf.Session, test_inputs_feed, test_outputs_feed, test_net_configs_feed=None):
        '''
        :param sess: tf.Session for test
        :param test_inputs_feed: same to batch_fit function's parameter of tr_inputs_feed
        :param test_outputs_feed:  same to batch_fit function's parameter of tr_outputs_feed
        :param test_net_configs_feed:  same to batch_fit function's parameter of tr_net_configs_feed
        :return:
            a result dict, the key is 'test_metrics'
        '''
        result = {}
        if self.model_save_path is not None:
            saver = tf.train.Saver()
            saver.restore(sess,self.model_save_path)
        feed = {}
        try:
            if isinstance(self.inputs, list):
                for i in range(len(self.inputs)):
                    feed[self.inputs[i]] = test_inputs_feed[i]
            else:
                feed[self.inputs] = test_inputs_feed
        except:
            raise RuntimeError("test inputs feed Error, maybe the len is not equal")
        try:
            if isinstance(self.standard_outputs, list):
                for i in range(len(self.standard_outputs)):
                    feed[self.standard_outputs[i]] = test_outputs_feed[i]
            else:
                feed[self.standard_outputs] = test_outputs_feed
        except:
            raise RuntimeError("test outputs feed Error, maybe the len is not equal")
        try:
            if self.net_configs is not None:
                if test_net_configs_feed is None:
                    raise RuntimeError("test net config not provide!")
                if isinstance(self.net_configs, list):
                    for i in range(len(self.net_configs)):
                        feed[self.net_configs[i]] = test_net_configs_feed[i]
                else:
                    feed[self.net_configs] = test_net_configs_feed
        except:
            raise RuntimeError("validation net configs feed Error, maybe the len is not equal or not provide")
        test_metrics = sess.run(self.metrics, feed_dict=feed)
        result["test_metrics"] = test_metrics
        return result

    def predict(self, sess: tf.Session, inputs_feed, net_configs_feed=None):
        '''
        :param sess: tf.Session
        :param inputs_feed: same to batch_fit function's parameter of tr_inputs_feed
        :param net_configs_feed: same to batch_fit function's parameter of tr_net_configs_feed
        :return:
            a result dict, the key is 'predict_outputs'
        '''
        result = {}
        if self.model_save_path is not None:
            saver = tf.train.Saver()
            saver.restore(sess, self.model_save_path)
        feed = {}
        try:
            if isinstance(self.inputs, list):
                for i in range(len(self.inputs)):
                    feed[self.inputs[i]] = inputs_feed[i]
            else:
                feed[self.inputs] = inputs_feed
        except:
            raise RuntimeError("predict inputs feed Error, maybe the len is not equal")
        try:
            if self.net_configs is not None:
                if net_configs_feed is None:
                    raise RuntimeError("predict net config not provide!")
                if isinstance(self.net_configs, list):
                    for i in range(len(self.net_configs)):
                        feed[self.net_configs[i]] = net_configs_feed[i]
                else:
                    feed[self.net_configs] = net_configs_feed
        except:
            raise RuntimeError("predict net configs feed Error, maybe the len is not equal or not provide")
        predict_outputs = sess.run(self.outputs, feed_dict=feed)
        result["predict_outputs"] = predict_outputs
        return result

