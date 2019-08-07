
from typing import Union, List
import tensorflow as tf
import numpy as np


class ModelModule:
    def __init__(self, inputs: Union[tf.Tensor, List[tf.Tensor]], outputs: Union[tf.Tensor, List[tf.Tensor]],
                 standard_outputs: Union[tf.Tensor, List[tf.Tensor]], loss: tf.Tensor, train_ops: tf.Tensor,
                 net_configs: Union[tf.Tensor, List[tf.Tensor]] = None, model_save_path: str = None,
                 metrics: Union[tf.Tensor, List[tf.Tensor]] = None):
        '''
        :param inputs:  the model inputs, a tensor or tensor list
        :param outputs:  the model outputs, a tensor or tensor list, usually call it predict
        :param standard_outputs: the model standard outputs, a tensor or tensor list, usually call it y
        :param loss:  the model loss, for model train, a tensor
        :param train_ops: the train ops
        :param net_configs:  the model other net configs with tensor that should be feed by user
        :param model_save_path: the model path for save model
        :param metrics:  the model metrics, like accuracy, MSE and so on
        '''
        self._inputs = inputs
        self._outputs = outputs
        self._standard_outputs = standard_outputs
        self._loss = loss
        # self._learning_rate = learning_rate
        self._net_configs = net_configs
        self._metrics = metrics
        self._model_save_path = model_save_path
        self._train_ops = train_ops

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
                  v_inputs_feed=None, v_outputs_feed=None, v_net_configs_feed=None, batch_size=64,
                  return_outputs=False):
        '''

        :param sess:  a tf.Session for train
        :param tr_inputs_feed:  train inputs feed value with the same sort in self.inputs
        :param tr_outputs_feed:  train standard outputs feed value with the same sort in self.standard_outputs
        :param tr_net_configs_feed:  train net configs feed value with the same sort in self.net_configs
        :param v_inputs_feed:  same with tr_inputs_feed ,but for validation
        :param v_outputs_feed: same with tr_outputs_feed ,but for validation
        :param v_net_configs_feed: same with tr_net_configs_feed ,but for validation
        :param batch_size: this batch_size only for validation
        :param return_outputs: return the outputs or not
        :return:
            a result with self.loss,self.metrics is not None ,self.metrics will append in result, if return_output
            is True,the output also in result, the keys will be 'tr_loss','tr_metrics','tr_outputs'
            the validation if exist                             'v_loss','v_metrics','v_outputs'
        '''
        result = {}
        try:
            feed = self.__feed(tr_inputs_feed, tr_outputs_feed, tr_net_configs_feed)
        except RuntimeError as e:
            raise RuntimeError("train:" + e)
        # print(sess, self.train_ops, self.loss, self.metrics)
        tr_metrics = None

        if self.metrics is not None:
            _, tr_loss, tr_metrics = sess.run([self.train_ops, self.loss, self.metrics], feed_dict=feed)
            result["tr_metrics"] = tr_metrics
        else:
            _, tr_loss = sess.run([self.train_ops, self.loss], feed_dict=feed)
        if return_outputs:
            tr_outputs = sess.run(self.outputs, feed_dict=feed)
            result["tr_outputs"] = tr_outputs
        result["tr_loss"] = tr_loss
        if v_inputs_feed is not None and v_outputs_feed is not None:
            try:
                length = self.__getfeedlength(v_inputs_feed)
            except RuntimeError as e:
                raise RuntimeError("validation:" + e)
            position = 0
            v_loss = []
            v_metrics = None
            v_outputs = None
            while position < length:
                # print("validation position:{}".format(position))
                batch_inputs_feed, batch_outputs_feed, position, actual_length = self.__next_batch(v_inputs_feed,
                                                                                                   v_outputs_feed,
                                                                                                   batch_size, position)
                try:
                    feed = self.__feed(batch_inputs_feed, batch_outputs_feed, v_net_configs_feed)
                except RuntimeError as e:
                    raise RuntimeError("validation:" + e)
                b_v_metrics = None
                if self.metrics is not None:
                    b_v_loss, b_v_metrics = sess.run([self.loss, self.metrics], feed_dict=feed)
                else:
                    b_v_loss = sess.run(self.loss, feed_dict=feed)
                v_loss.append(b_v_loss)
                if v_metrics is None:
                    v_metrics = b_v_metrics
                else:
                    if isinstance(self.metrics, list):
                        for i in range(len(self.metrics)):
                            v_metrics[i] = np.r_[v_metrics[i], b_v_metrics[i]]
                    else:
                        v_metrics = np.r_[v_metrics, b_v_metrics]
                if return_outputs:
                    b_v_outputs = sess.run(self.outputs, feed_dict=feed)
                    if v_outputs is None:
                        v_outputs = b_v_outputs
                    else:
                        if isinstance(self.outputs, list):
                            for i in range(len(self.outputs)):
                                v_outputs[i] = np.r_[v_outputs[i], b_v_outputs[i]]
                        else:
                            v_outputs = np.r_[v_outputs, b_v_outputs]

            v_loss = np.mean(np.array(v_loss))
            if v_metrics is not None:
                if isinstance(self.metrics, list):
                    for i in range(len(self.metrics)):
                        v_metrics[i] = np.mean(v_metrics[i], axis=0)
                else:
                    v_metrics = np.mean(v_metrics, axis=0)
                result["v_metrics"] = v_metrics
            result["v_loss"] = v_loss
            if return_outputs:
                result["v_outputs"] = v_outputs
        return result

    def evaluation(self, sess: tf.Session, test_inputs_feed, test_outputs_feed, test_net_configs_feed=None,
                   batch_size=64, is_in_train=False, return_outputs=False):
        '''
        :param sess: tf.Session for test
        :param test_inputs_feed: same to batch_fit function's parameter of tr_inputs_feed
        :param test_outputs_feed:  same to batch_fit function's parameter of tr_outputs_feed
        :param test_net_configs_feed:  same to batch_fit function's parameter of tr_net_configs_feed
        :param is_in_train: is also train and only test it is correct
        :param return_outputs: return the outputs or not
        :return:
            a result dict of self.loss, if self.metrics is not None,self.metrics will append to result,if return_outputs
            is True, the self.outputs will be in result, the key is 'test_loss','test_metrics','test_outputs'
        '''
        result = {}
        if self.model_save_path is not None:
            saver = tf.train.Saver()
            saver.restore(sess, self.model_save_path)
        else:
            if is_in_train:
                pass
            else:
                raise RuntimeError("evaluation:the model not be train or not save with giving a model_save_path")
        try:
            length = self.__getfeedlength(test_inputs_feed)
        except RuntimeError as e:
            raise RuntimeError("evaluation:" + e)

        position = 0
        test_metrics = None
        test_loss = []
        test_outputs = None
        while position < length:
            batch_inputs_feed, batch_outputs_feed, position, actual_length = self.__next_batch(test_inputs_feed,
                                                                                               test_outputs_feed,
                                                                                               batch_size, position)
            try:
                feed = self.__feed(batch_inputs_feed, batch_outputs_feed, test_net_configs_feed)
            except RuntimeError as e:
                raise RuntimeError("test:" + e)
            b_test_metrics = None
            if self.metrics is not None:
                b_test_loss, b_test_metrics = sess.run([self.loss, self.metrics], feed_dict=feed)
            else:
                b_test_loss = sess.run(self.loss, feed_dict=feed)
            test_loss.append(b_test_loss)
            if test_metrics is None:
                test_metrics = b_test_metrics
            else:
                if isinstance(self.metrics, list):
                    for i in range(len(self.metrics)):
                        test_metrics[i] = np.r_[test_metrics[i], b_test_metrics[i]]
                else:
                    test_metrics = np.r_[test_metrics, b_test_metrics]
            if return_outputs:
                b_test_outputs = sess.run(self.outputs, feed_dict=feed)
                if test_outputs is None:
                    test_outputs = b_test_outputs
                else:
                    if isinstance(self.outputs, list):
                        for i in range(len(self.outputs)):
                            test_outputs[i] = np.r_[test_outputs[i], b_test_outputs[i]]
                    else:
                        test_outputs = np.r_[test_outputs, b_test_outputs]
        if self.metrics is not None:
            if isinstance(self.metrics, list):
                for i in range(len(self.metrics)):
                    test_metrics[i] = np.mean(test_metrics[i], axis=0)
            else:
                test_metrics = np.mean(np.array(test_metrics), axis=0)
            result["test_metrics"] = test_metrics
        if return_outputs:
            result["test_outputs"] = test_outputs
        test_loss = np.mean(np.array(test_loss))
        result["test_loss"] = test_loss
        return result

    def predict(self, sess: tf.Session, inputs_feed, net_configs_feed=None, batch_size=64, is_in_train=False):
        '''
        :param sess: tf.Session
        :param inputs_feed: same to batch_fit function's parameter of tr_inputs_feed
        :param net_configs_feed: same to batch_fit function's parameter of tr_net_configs_feed
        :param batch_size: batch size
        :param is_in_train: is also train and only test it is correct
        :return:
            a result dict, the key is 'predict'
        '''
        result = {}
        if self.model_save_path is not None:
            saver = tf.train.Saver()
            saver.restore(sess, self.model_save_path)
        else:
            if is_in_train:
                pass
            else:
                raise RuntimeError("predict: the model not be train or not save with giving a model_save_path")
        try:
            length = self.__getfeedlength(inputs_feed)
        except RuntimeError as e:
            raise RuntimeError("predict:" + e)
        predict_outputs = None
        position = 0
        while position < length:
            batch_inputs_feed, batch_outputs_feed, position, actual_length = self.__next_batch(inputs_feed, None,
                                                                                               batch_size, position)
            try:
                feed = self.__feed(batch_inputs_feed, None, net_configs_feed=net_configs_feed)
            except RuntimeError as e:
                raise RuntimeError("predict:" + e)
            if predict_outputs is None:
                predict_outputs = sess.run(self.outputs, feed_dict=feed)[0:actual_length]
            else:
                predict_outputs = np.r_[predict_outputs, sess.run(self.outputs, feed_dict=feed)[0:actual_length]]
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
         a tuple of (batch_inputs_feed, batch_outputs_feed, position, actual_length)
        '''
        actual_length = None
        if isinstance(self.inputs, list):
            batch_inputs_feed = []
            if len(self.inputs) != len(inputs_feed):
                # print("next batch inputs_feed length error")
                raise RuntimeError("next batch inputs_feed length error")
            for i in range(len(self.inputs)):
                if isinstance(inputs_feed[i], list):
                    if position + batch_size <= len(inputs_feed[i]):
                        batch_inputs_feed.append(inputs_feed[i][position:position + batch_size])
                    else:
                        temp = list(inputs_feed[i][position:])
                        # batch_inputs_feed.append(inputs_feed[i][position:])
                        res = batch_size - len(inputs_feed[i][position:])
                        actual_length = len(inputs_feed[i][position:])
                        while res > len(inputs_feed[i]):
                            temp = temp + inputs_feed[i]
                            res -= len(inputs_feed[i])
                        if res > 0:
                            temp = temp + inputs_feed[i][0:res]
                        batch_inputs_feed.append(temp)
                elif isinstance(inputs_feed[i], np.ndarray):
                    if position + batch_size < inputs_feed[i].shape[0]:
                        batch_inputs_feed.append(inputs_feed[i][position:position + batch_size])
                    else:
                        temp = np.array(inputs_feed[i][position:])
                        # batch_inputs_feed.append(inputs_feed[i][position:])
                        res = batch_size - inputs_feed[i][position:].shape[0]
                        actual_length = inputs_feed[i][position:].shape[0]
                        while res > inputs_feed[i].shape[0]:
                            temp = np.r_[temp, inputs_feed[i]]
                            res -= inputs_feed[i].shape[0]
                        if res > 0:
                            temp = np.r_[temp, inputs_feed[i][0:res]]
                        batch_inputs_feed.append(temp)
                        # batch_inputs_feed.append(inputs_feed[i][position:])
                else:
                    # print("one input feed must be list or numpy.ndarray")
                    raise RuntimeError("your type is " + str(type(inputs_feed[i])) +
                                       ",but one input feed must be list or numpy.ndarray")
            # pass
        else:
            if isinstance(inputs_feed, list):
                if position+batch_size < len(inputs_feed):
                    batch_inputs_feed = inputs_feed[position:position+batch_size]
                else:
                    batch_inputs_feed = list(inputs_feed[position:])
                    actual_length = len(inputs_feed[position:])
                    res = batch_size - len(inputs_feed[position:])
                    while res >= len(inputs_feed):
                        batch_inputs_feed = batch_inputs_feed + inputs_feed
                        res = res - len(inputs_feed)
                    if res > 0:
                        batch_inputs_feed = batch_inputs_feed + inputs_feed[0:res]
            elif isinstance(inputs_feed, np.ndarray):
                if position + batch_size < inputs_feed.shape[0]:
                    batch_inputs_feed = inputs_feed[position:position + batch_size]
                else:
                    batch_inputs_feed = np.array(inputs_feed[position:])
                    actual_length = inputs_feed[position:].shape[0]
                    res = batch_size - inputs_feed[position:].shape[0]
                    while res >= inputs_feed.shape[0]:
                        batch_inputs_feed = np.r_[batch_inputs_feed, inputs_feed]
                        res = res - inputs_feed.shape[0]
                    if res > 0:
                        batch_inputs_feed = np.r_[batch_inputs_feed, inputs_feed[0:res]]
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
                            temp = list(outputs_feed[i][position:])
                            res = batch_size - len(outputs_feed[i][position:])
                            actual_length = len(outputs_feed[i][position:])
                            while res > len(outputs_feed[i]):
                                temp = temp + outputs_feed[i]
                                res -= len(outputs_feed[i])
                            if res > 0:
                                temp = temp + outputs_feed[i][0:res]
                            batch_outputs_feed.append(temp)
                    elif isinstance(outputs_feed[i], np.ndarray):
                        if position + batch_size < outputs_feed[i].shape[0]:
                            batch_outputs_feed.append(outputs_feed[i][position:position + batch_size])
                        else:
                            temp = np.array(outputs_feed[i][position:])
                            res = batch_size - outputs_feed[i][position:].shape[0]
                            actual_length = outputs_feed[i][position:].shape[0]
                            while res > outputs_feed[i].shape[0]:
                                temp = np.r_[temp, outputs_feed[i]]
                                res -= outputs_feed[i].shape[0]
                            if res > 0:
                                temp = np.r_[temp, outputs_feed[i][0:res]]
                            batch_outputs_feed.append(temp)
                    else:
                        raise RuntimeError("your type is " + str(type(outputs_feed[i])) +
                                           ",but one output feed must be list or numpy.ndarray")
            else:
                if isinstance(outputs_feed, list):
                    if position+batch_size < len(outputs_feed):
                        batch_outputs_feed = outputs_feed[position:position+batch_size]
                    else:
                        batch_outputs_feed = list(outputs_feed[position:])
                        actual_length = len(outputs_feed[position:])
                        res = batch_size - len(outputs_feed[position:])
                        while res >= len(outputs_feed):
                            batch_outputs_feed = batch_outputs_feed + outputs_feed
                            res = res - len(outputs_feed)
                        if res > 0:
                            batch_outputs_feed = batch_outputs_feed + outputs_feed[0:res]
                elif isinstance(outputs_feed, np.ndarray):
                    if position+batch_size < outputs_feed.shape[0]:
                        batch_outputs_feed = outputs_feed[position:position+batch_size]
                    else:
                        batch_outputs_feed = np.array(outputs_feed[position:])
                        actual_length = outputs_feed[position:].shape[0]
                        res = batch_size - outputs_feed[position:].shape[0]
                        while res >= outputs_feed.shape[0]:
                            batch_outputs_feed = np.r_[batch_outputs_feed, outputs_feed]
                            res = res - outputs_feed.shape[0]
                        if res > 0:
                            batch_outputs_feed = np.r_[batch_outputs_feed, outputs_feed[0:res]]
                else:
                    raise RuntimeError("your type is " + str(type(outputs_feed)) +
                                       ",but one output feed must be list or numpy.ndarray")
        else:
            batch_outputs_feed = None
        position += batch_size
        return batch_inputs_feed, batch_outputs_feed, position, actual_length
