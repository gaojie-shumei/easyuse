
from typing import Union, List
import tensorflow as tf
import numpy as np


class ModelModule:
    def __init__(self, inputs: Union[tf.Tensor, List[tf.Tensor]], outputs: Union[tf.Tensor, List[tf.Tensor]],
                 standard_outputs: Union[tf.Tensor, List[tf.Tensor]], loss: tf.Tensor,
                 optimizer: Union[tf.train.Optimizer, tf.keras.optimizers.Optimizer] = tf.keras.optimizers.Adam(0.001),
                 net_configs: Union[tf.Tensor, List[tf.Tensor]] = None, model_save_path: str = None,
                 metrics: Union[tf.Tensor, List[tf.Tensor]] = None, var_list: List[tf.Tensor]=None):
        '''
        :param inputs:  the model inputs, a tensor or tensor list
        :param outputs:  the model outputs, a tensor or tensor list, usually call it predict
        :param standard_outputs: the model standard outputs, a tensor or tensor list, usually call it y
        :param loss:  the model loss, for model train, a tensor
        :param optimizer: the model optimizer with parameters, like adam(learning_rate=0.001)
        :param net_configs:  the model other net configs with tensor that should be feed by user
        :param model_save_path: the model path for save model
        :param metrics:  the model metrics, like accuracy, MSE and so on
        :param var_list: the vars need to be trained
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
        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            self._train_ops = optimizer.minimize(loss,var_list=var_list)

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
                  v_inputs_feed=None, v_outputs_feed=None, v_net_configs_feed=None, batch_size=64):
        '''

        :param sess:  a tf.Session for train
        :param tr_inputs_feed:  train inputs feed value with the same sort in self.inputs
        :param tr_outputs_feed:  train standard outputs feed value with the same sort in self.standard_outputs
        :param tr_net_configs_feed:  train net configs feed value with the same sort in self.net_configs
        :param v_inputs_feed:  same with tr_inputs_feed ,but for validation
        :param v_outputs_feed: same with tr_outputs_feed ,but for validation
        :param v_net_configs_feed: same with tr_net_configs_feed ,but for validation
        :param batch_size: this batch_size only for validation
        :return:
            the self.metrics value with the feed value, wrapped with a dict, the train metrics value's key is 'tr_metrics'
            the validation if exist,the validation metrics value's key is 'v_metrics'
        '''
        result = {}
        try:
            feed = self.__feed(tr_inputs_feed, tr_outputs_feed, tr_net_configs_feed)
        except RuntimeError as e:
            raise RuntimeError("train:" + e)
        # print(feed)
        _, tr_metrics = sess.run([self.train_ops, self.metrics], feed_dict=feed)
        result["tr_metrics"] = tr_metrics
        if v_inputs_feed is not None and v_outputs_feed is not None:
            try:
                length = self.__getfeedlength(v_inputs_feed)
            except RuntimeError as e:
                raise RuntimeError("validation:" + e)
            position = 0
            v_metrics = None
            while position < length:
                batch_inputs_feed, batch_outputs_feed, position = self.__next_batch(v_inputs_feed, v_outputs_feed,
                                                                                    batch_size, position)
                try:
                    feed = self.__feed(v_inputs_feed, v_outputs_feed, v_net_configs_feed)
                except RuntimeError as e:
                    raise RuntimeError("validation:" + e)
                if v_metrics is None:
                    v_metrics = sess.run(self.metrics, feed_dict=feed)
                else:
                    v_metrics = np.r_[v_metrics, sess.run(self.metrics, feed_dict=feed)]
            v_metrics = np.mean(v_metrics, axis=0)
            result["v_metrics"] = v_metrics
        return result

    def evaluation(self, sess: tf.Session, test_inputs_feed, test_outputs_feed, test_net_configs_feed=None,
                   batch_size=64):
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
        else:
            raise RuntimeError("evaluation: the model not be train or not save to path with giving a model_save_path")
        try:
            length = self.__getfeedlength(test_inputs_feed)
        except RuntimeError as e:
            raise RuntimeError("evaluation:" + e)

        position = 0
        test_metrics = None
        while position < length:
            batch_inputs_feed, batch_outputs_feed, position = self.__next_batch(test_inputs_feed, test_outputs_feed,
                                                                                batch_size, position)
            try:
                feed = self.__feed(batch_inputs_feed, batch_outputs_feed, test_net_configs_feed)
            except RuntimeError as e:
                raise RuntimeError("test:" + e)
            if test_metrics is None:
                test_metrics = sess.run(self.metrics, feed_dict=feed)
            else:
                test_metrics = np.r_[test_metrics, sess.run(self.metrics, feed_dict=feed)]
        test_metrics = np.mean(np.array(test_metrics), axis=0)
        result["test_metrics"] = test_metrics
        return result

    def predict(self, sess: tf.Session, inputs_feed, net_configs_feed=None, batch_size=64):
        '''
        :param sess: tf.Session
        :param inputs_feed: same to batch_fit function's parameter of tr_inputs_feed
        :param net_configs_feed: same to batch_fit function's parameter of tr_net_configs_feed
        :param batch_size: batch size
        :return:
            a result dict, the key is 'predict_outputs'
        '''
        result = {}
        if self.model_save_path is not None:
            saver = tf.train.Saver()
            saver.restore(sess, self.model_save_path)
        else:
            raise RuntimeError("predict: the model not be train or not save to path with giving a model_save_path")
        try:
            length = self.__getfeedlength(inputs_feed)
        except RuntimeError as e:
            raise RuntimeError("predict:" + e)
        predict_outputs = None
        position = 0
        while position < length:
            batch_inputs_feed, batch_outputs_feed, position = self.__next_batch(inputs_feed, None, batch_size, position)
            try:
                feed = self.__feed(batch_inputs_feed, None, net_configs_feed=net_configs_feed)
            except RuntimeError as e:
                raise RuntimeError("predict:" + e)
            if predict_outputs is None:
                predict_outputs = sess.run(self.outputs, feed_dict=feed)
            else:
                predict_outputs = np.r_[predict_outputs, sess.run(self.outputs, feed_dict=feed)]
        result["predict"] = predict_outputs
        return result

    def __getfeedlength(self, inputs_feed):
        if isinstance(self.inputs, list):
            if isinstance(inputs_feed[0], list):
                length = len(inputs_feed[0])
            elif isinstance(inputs_feed[0], np.ndarray):
                length = inputs_feed[0].shape[0]
            else:
                raise RuntimeError("your type is " + str(type(inputs_feed[0])) +
                                   "one input feed should be list or numpy.ndarray")
        else:
            if isinstance(inputs_feed, list):
                length = len(inputs_feed)
            elif isinstance(inputs_feed, np.ndarray):
                length = inputs_feed.shape[0]
            else:
                raise RuntimeError("your type is " + str(type(inputs_feed)) +
                                   "one input feed should be list or numpy.ndarray")
        return length

    def __feed(self, inputs_feed, outputs_feed=None, net_configs_feed=None):
        '''

        :param inputs_feed: same to batch_fit
        :param outputs_feed:  same to batch_fit
        :param net_configs_feed:  same to batch_fit
        :return:
          the feed for network
        '''
        feed = {}
        try:
            if isinstance(self.inputs, list):
                for i in range(len(self.inputs)):
                    feed[self.inputs[i]] = inputs_feed[i]
            else:
                feed[self.inputs] = inputs_feed
        except:
            raise RuntimeError("inputs feed Error, maybe the len is not equal")
        try:
            if outputs_feed is not None:
                if isinstance(self.standard_outputs, list):
                    for i in range(len(self.standard_outputs)):
                        feed[self.standard_outputs[i]] = outputs_feed[i]
                else:
                    feed[self.standard_outputs] = outputs_feed
        except:
            raise RuntimeError("outputs feed Error, maybe the len is not equal")
        try:
            if self.net_configs is not None:
                if net_configs_feed is None:
                    raise RuntimeError("net configs feed should be provided")
                if isinstance(self.net_configs, list):
                    for i in range(len(self.net_configs)):
                        feed[self.net_configs[i]] = net_configs_feed[i]
                else:
                    feed[self.net_configs] = net_configs_feed
        except:
            raise RuntimeError("net configs feed Error, maybe the len is not equal or not provide")
        return feed

    def __next_batch(self, inputs_feed, outputs_feed=None, batch_size=64, position=0):
        '''
        this function only for validation and test, not to train
        :param inputs_feed:  same to v_inputs_feed in batch_fit, test_inputs_feed in evaluation, inputs_feed in predict
        :param outputs_feed: same to v_outputs_feed in batch_fit, test_outputs_feed in evaluation
        :param batch_size: batch size
        :param position: the position for this batch
        :return:
         a tuple of (batch_inputs_feed, batch_outputs_feed, position)
        '''
        if isinstance(self.inputs, list):
            batch_inputs_feed = []
            if len(self.inputs)!=len(inputs_feed):
                # print("next batch inputs_feed length error")
                raise RuntimeError("next batch inputs_feed length error")
            for i in range(len(self.inputs)):
                if isinstance(inputs_feed[i], list):
                    if position + batch_size < len(inputs_feed[i]):
                        batch_inputs_feed.append(inputs_feed[i][position:position + batch_size])
                    else:
                        batch_inputs_feed.append(inputs_feed[i][position:])
                elif isinstance(inputs_feed[i], np.ndarray):
                    if position + batch_size < inputs_feed[i].shape[0]:
                        batch_inputs_feed.append(inputs_feed[i][position:position + batch_size])
                    else:
                        batch_inputs_feed.append(inputs_feed[i][position:])
                else:
                    # print("one input feed must be list or numpy.ndarray")
                    raise RuntimeError("your type is " + str(type(inputs_feed[i])) +
                                    ",but one input feed must be list or numpy.ndarray")
            # pass
        else:
            if isinstance(inputs_feed, list):
                if position+batch_size<len(inputs_feed):
                    batch_inputs_feed = inputs_feed[position:position+batch_size]
                else:
                    batch_inputs_feed = inputs_feed[position:]
            elif isinstance(inputs_feed, np.ndarray):
                if position + batch_size < inputs_feed.shape[0]:
                    batch_inputs_feed = inputs_feed[position:position + batch_size]
                else:
                    batch_inputs_feed = inputs_feed[position:]
            else:
                raise RuntimeError("your type is " + str(type(inputs_feed)) +
                                ",but one input feed must be list or numpy.ndarray")

        if outputs_feed is not None:
            if isinstance(self.standard_outputs, list):
                batch_outputs_feed = []
                for i in range(len(self.standard_outputs)):
                    if isinstance(outputs_feed[i], list):
                        if position + batch_size < len(outputs_feed[i]):
                            batch_outputs_feed.append(outputs_feed[i][position:position + batch_size])
                        else:
                            batch_outputs_feed.append(outputs_feed[i][position:])
                    elif isinstance(outputs_feed[i], np.ndarray):
                        if position + batch_size < outputs_feed[i].shape[0]:
                            batch_outputs_feed.append(outputs_feed[i][position:position + batch_size])
                        else:
                            batch_outputs_feed.append(outputs_feed[i][position:])
                    else:
                        raise RuntimeError("your type is " + str(type(outputs_feed[i])) +
                                        ",but one output feed must be list or numpy.ndarray")
            else:
                if isinstance(outputs_feed, list):
                    if position+batch_size < len(outputs_feed):
                        batch_outputs_feed = outputs_feed[position:position+batch_size]
                    else:
                        batch_outputs_feed = outputs_feed[position:]
                elif isinstance(outputs_feed, np.ndarray):
                    if position+batch_size < outputs_feed.shape[0]:
                        batch_outputs_feed = outputs_feed[position:position+batch_size]
                    else:
                        batch_outputs_feed = outputs_feed[position:]
                else:
                    raise RuntimeError("your type is " + str(type(outputs_feed)) +
                                    ",but one output feed must be list or numpy.ndarray")
        else:
            batch_outputs_feed = None
        position += batch_size
        return batch_inputs_feed, batch_outputs_feed, position
