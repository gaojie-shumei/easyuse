from typing import Union, List
import tensorflow as tf
import numpy as np
import random
from os import path as os_path


class ModelModule:
    def __init__(self, inputs: Union[tf.Tensor, List[tf.Tensor]], outputs: Union[tf.Tensor, List[tf.Tensor]],
                 standard_outputs: Union[tf.Tensor, List[tf.Tensor]], loss: tf.Tensor, train_ops: tf.Tensor,
                 net_configs: Union[tf.Tensor, List[tf.Tensor]] = None, model_save_path: str = None,
                 metrics: Union[tf.Tensor, List[tf.Tensor]] = None, num_parallel_calls=0):
        '''
        :param inputs:  the model inputs, a tensor or tensor list
        :param outputs:  the model outputs, a tensor or tensor list, usually call it predict
        :param standard_outputs: the model standard outputs, a tensor or tensor list, usually call it y
        :param loss:  the model loss, for model train, a tensor
        :param train_ops: the train ops
        :param net_configs:  the model other net configs with tensor that should be feed by user
        :param model_save_path: the model path for save model
        :param metrics:  the model metrics, like accuracy, MSE and so on
        :param num_parallel_calls: data parallel num, usually use multi GPU
        '''
        self._inputs = inputs
        self._outputs = outputs
        self._standard_outputs = standard_outputs
        self._loss = loss
        self._net_configs = net_configs
        self._metrics = metrics
        self._model_save_path = model_save_path
        self._train_ops = train_ops
        self._num_parallel_calls = num_parallel_calls

    @property
    def num_parallel_calls(self):
        return self._num_parallel_calls

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
    def net_configs(self):
        return self._net_configs

    @property
    def model_save_path(self):
        return self._model_save_path

    @property
    def metrics(self):
        return self._metrics

    def fit(self, sess: tf.Session, epoch: int, tr_inputs_feed, tr_outputs_feed, tr_net_configs_feed=None,
            v_inputs_feed=None, v_outputs_feed=None, v_net_configs_feed=None, batch_size=64,return_outputs=False,
            show_result=True, start_save_model_epoch=None, model_name='model'):
        '''

        :param sess:  a tf.Session for train
        :param epoch: the train num
        :param tr_inputs_feed:  train inputs feed value with the same sort in self.inputs
        :param tr_outputs_feed:  train standard outputs feed value with the same sort in self.standard_outputs
        :param tr_net_configs_feed:  train net configs feed value with the same sort in self.net_configs
        :param v_inputs_feed:  same with tr_inputs_feed ,but for validation
        :param v_outputs_feed: same with tr_outputs_feed ,but for validation
        :param v_net_configs_feed: same with tr_net_configs_feed ,but for validation
        :param batch_size: this batch_size only for validation
        :param return_outputs: return the outputs or not
        :param show_result: one epoch to show result in console
        :param start_save_model_epoch: which epoch to save model
        :param model_name: model_name  'model' is the default
        :return:
            a result with self.loss,self.metrics is not None ,self.metrics will append in result, if return_output
            is True,the output also in result, the keys will be 'tr_loss','tr_metrics','tr_outputs'
            the validation if exist and do_validation is True   'v_loss','v_metrics','v_outputs'
        '''
        results = []
        for i in range(epoch):
            save_model = False
            if i >= start_save_model_epoch:
                save_model = True
            generator = self.__generator_batch(batch_size, tr_inputs_feed, tr_outputs_feed, shuffle=True)
            for batch_inputs_feed, batch_outputs_feed, batch_len, is_one_epoch in generator:
                result = self.batch_fit(sess, batch_inputs_feed, batch_outputs_feed, tr_net_configs_feed, v_inputs_feed,
                                        v_outputs_feed, v_net_configs_feed, batch_size, return_outputs, is_one_epoch,
                                        save_model, model_name)
                if is_one_epoch:
                    results.append(result)
                    if show_result:
                        print("epoch=", i, ",result=", result)
        return results

    def batch_fit(self, sess: tf.Session, tr_inputs_feed, tr_outputs_feed, tr_net_configs_feed=None,
                  v_inputs_feed=None, v_outputs_feed=None, v_net_configs_feed=None, batch_size=64,
                  return_outputs=False, do_validation=False, save_model=False, model_name='model'):
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
        :param do_validation: do validation or not
        :param save_model: True save model, False not
        :param model_name: model name 'model' as the default
        :return:
            a result with self.loss,self.metrics is not None ,self.metrics will append in result, if return_output
            is True,the output also in result, the keys will be 'tr_loss','tr_metrics','tr_outputs'
            the validation if exist and do_validation is True   'v_loss','v_metrics','v_outputs'
        '''
        result = {}
        global_step = tf.train.get_or_create_global_step()
        feed = self.__feed(tr_inputs_feed, tr_outputs_feed, tr_net_configs_feed)
        sess.run(self.train_ops, feed_dict=feed)
        if self.metrics is not None:
            if return_outputs:
                tr_run = sess.run([self.loss, self.metrics, self.outputs], feed_dict=feed)
            else:
                tr_run = sess.run([self.loss, self.metrics], feed_dict=feed)
        else:
            if return_outputs:
                tr_run = sess.run([self.loss, self.outputs], feed_dict=feed)
            else:
                tr_run = sess.run([self.loss], feed_dict=feed)
        result["tr_loss"] = tr_run[0]
        if self.metrics is not None:
            result["tr_metrics"] = tr_run[1]
            if return_outputs:
                result["tr_outputs"] = tr_run[2]
        else:
            if return_outputs:
                result["tr_outputs"] = tr_run[1]
        if do_validation and v_inputs_feed is not None and v_outputs_feed is not None:
            generator = self.__generator_batch(batch_size, v_inputs_feed, v_outputs_feed)
            v_loss, v_metrics, v_outputs, count = 0, None, None, 0
            for batch_inputs_feed, batch_outputs_feed, batch_len, _ in generator:
                feed = self.__feed(batch_inputs_feed, batch_outputs_feed, v_net_configs_feed)
                if self.metrics is not None:
                    if return_outputs:
                        v_run = sess.run([self.loss, self.metrics, self.outputs], feed_dict=feed)
                    else:
                        v_run = sess.run([self.loss, self.metrics], feed_dict=feed)
                else:
                    if return_outputs:
                        v_run = sess.run([self.loss, self.outputs], feed_dict=feed)
                    else:
                        v_run = sess.run([self.loss], feed_dict=feed)
                count += 1
                v_loss += v_run[0]
                if self.metrics is not None:
                    if isinstance(self.metrics, list):
                        v_metrics = self.__type2concat(v_metrics, v_run[1])
                    else:
                        if v_metrics is None:
                            v_metrics = v_run[1]
                        else:
                            v_metrics += v_run[1]
                    if return_outputs:
                        outputs = v_run[2]
                        if (self.num_parallel_calls > 0 and batch_size * self.num_parallel_calls != batch_len) or \
                                (self.num_parallel_calls == 0 and batch_size != batch_len):
                            if isinstance(self.outputs, list):
                                for i in range(len(self.outputs)):
                                    outputs[i] = outputs[i][0:batch_len]
                            else:
                                outputs = outputs[0:batch_len]
                        v_outputs = self.__type2concat(v_outputs, outputs)
                else:
                    if return_outputs:
                        outputs = v_run[1]
                        if (self.num_parallel_calls > 0 and batch_size * self.num_parallel_calls != batch_len) or \
                                (self.num_parallel_calls == 0 and batch_size != batch_len):
                            if isinstance(self.outputs, list):
                                for i in range(len(self.outputs)):
                                    outputs[i] = outputs[i][0:batch_len]
                            else:
                                outputs = outputs[0:batch_len]
                        v_outputs = self.__type2concat(v_outputs, outputs)
            v_loss = self.__type2mean(self.loss, v_loss, count)
            result["v_loss"] = v_loss
            if self.metrics is not None:
                v_metrics = self.__type2mean(self.metrics, v_metrics, count)
                result["v_metrics"] = v_metrics
            if return_outputs:
                result["v_outputs"] = v_outputs
        if save_model and self.model_save_path is not None:
            saver = tf.train.Saver()
            saver.save(sess, os_path.join(self.model_save_path, model_name+".ckpt"), global_step=global_step)
        return result

    def evaluation(self, sess: tf.Session, test_inputs_feed, test_outputs_feed, test_net_configs_feed=None,
                   batch_size=64, is_in_train=False, return_outputs=False):
        '''
        :param sess: tf.Session for test
        :param test_inputs_feed: same to batch_fit function's parameter of tr_inputs_feed
        :param test_outputs_feed:  same to batch_fit function's parameter of tr_outputs_feed
        :param test_net_configs_feed:  same to batch_fit function's parameter of tr_net_configs_feed
        :param batch_size: batch size
        :param is_in_train: is also train and only test it is correct
        :param return_outputs: return the outputs or not
        :return:
            a result dict of self.loss, if self.metrics is not None,self.metrics will append to result,if return_outputs
            is True, the self.outputs will be in result, the key is 'test_loss','test_metrics','test_outputs'
        '''
        result = {}
        if is_in_train:
            pass
        elif self.model_save_path is not None:
            if tf.train.latest_checkpoint(self.model_save_path) is not None:
                saver = tf.train.Saver()
                saver.restore(sess, tf.train.latest_checkpoint(self.model_save_path))
            else:
                raise RuntimeError("evaluation:the model not save")
        else:
            raise RuntimeError("evaluation:the model not be train or not save with giving a model_save_path")
        test_loss, test_metrics, test_outputs, count = 0, None, None, 0
        generator = self.__generator_batch(batch_size, test_inputs_feed, test_outputs_feed)
        for batch_inputs_feed, batch_outputs_feed, batch_len, _ in generator:
            feed = self.__feed(batch_inputs_feed, batch_outputs_feed, test_net_configs_feed)
            if self.metrics is not None:
                if return_outputs:
                    test_run = sess.run([self.loss, self.metrics, self.outputs], feed_dict=feed)
                else:
                    test_run = sess.run([self.loss, self.metrics], feed_dict=feed)
            else:
                if return_outputs:
                    test_run = sess.run([self.loss, self.outputs], feed_dict=feed)
                else:
                    test_run = sess.run([self.loss], feed_dict=feed)
            count += 1
            test_loss += test_run[0]
            if self.metrics is not None:
                if isinstance(self.metrics, list):
                    test_metrics = self.__type2concat(test_metrics, test_run[1])
                else:
                    if test_metrics is None:
                        test_metrics = test_run[1]
                    else:
                        test_metrics += test_run[1]
                if return_outputs:
                    outputs = test_run[2]
                    if (self.num_parallel_calls > 0 and batch_size * self.num_parallel_calls != batch_len) or \
                            (self.num_parallel_calls == 0 and batch_size != batch_len):
                        if isinstance(self.outputs, list):
                            for i in range(len(self.outputs)):
                                outputs[i] = outputs[i][0:batch_len]
                        else:
                            outputs = outputs[0:batch_len]
                    test_outputs = self.__type2concat(test_outputs, outputs)
            else:
                if return_outputs:
                    outputs = test_run[1]
                    if (self.num_parallel_calls > 0 and batch_size * self.num_parallel_calls != batch_len) or \
                            (self.num_parallel_calls == 0 and batch_size != batch_len):
                        if isinstance(self.outputs, list):
                            for i in range(len(self.outputs)):
                                outputs[i] = outputs[i][0:batch_len]
                        else:
                            outputs = outputs[0:batch_len]
                    test_outputs = self.__type2concat(test_outputs, outputs)
        test_loss = self.__type2mean(self.loss, test_loss, count)
        result["test_loss"] = test_loss
        if self.metrics is not None:
            test_metrics = self.__type2mean(self.metrics, test_metrics, count)
            result["test_metrics"] = test_metrics
        if return_outputs:
            result["test_outputs"] = test_outputs
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
        if is_in_train:
            pass
        elif self.model_save_path is not None:
            if tf.train.latest_checkpoint(self.model_save_path) is not None:
                saver = tf.train.Saver()
                saver.restore(sess, tf.train.latest_checkpoint(self.model_save_path))
            else:
                raise RuntimeError("predict:the model not save")
        else:
            raise RuntimeError("predict:the model not be train or not save with giving a model_save_path")
        predict_outputs = None
        generator = self.__generator_batch(batch_size, inputs_feed)
        for batch_inputs_feed, _, batch_len, _ in generator:
            feed = self.__feed(batch_inputs_feed, None, net_configs_feed)
            outputs = sess.run(self.outputs, feed_dict=feed)
            if (self.num_parallel_calls > 0 and batch_size * self.num_parallel_calls != batch_len) or \
                    (self.num_parallel_calls == 0 and batch_size != batch_len):
                if isinstance(self.outputs, list):
                    for i in range(len(self.outputs)):
                        outputs[i] = outputs[i][0:batch_len]
                else:
                    outputs = outputs[0:batch_len]

            predict_outputs = self.__type2concat(predict_outputs, outputs)
        result["predict"] = predict_outputs
        return result

    def __feed(self, inputs_feed, outputs_feed=None, net_configs_feed=None):
        '''

        :param inputs_feed: self.inputs feed
        :param outputs_feed:  self.standard_outputs feed
        :param net_configs_feed:  self.net_configs feed
        :return:
          the feed for network
        '''
        feed = {}
        feed.update(self.__type2feed(self.inputs, inputs_feed))
        if outputs_feed is not None:
            feed.update(self.__type2feed(self.standard_outputs, outputs_feed))
        if self.net_configs is not None:
            feed.update(self.__type2feed(self.net_configs, net_configs_feed))
        return feed

    @staticmethod
    def __type2feed(self_placeholder, feed_data):
        '''
        :param self_placeholder:
        :param feed_data:
        :return:
            the feed dict for the placeholder
        '''
        feed = {}
        try:
            if self_placeholder is not None:
                if feed_data is None:
                    raise RuntimeError("feed data not provide")
                if isinstance(self_placeholder, list):
                    for i in range(len(self_placeholder)):
                        feed[self_placeholder[i]] = feed_data[i]
                else:
                    feed[self_placeholder] = feed_data
        except:
            raise RuntimeError("feed data error")
        return feed

    @staticmethod
    def __type2mean(self_placeholder, result, count):
        '''
        :param self_placeholder: mean of this placeholder
        :param result: result of this placeholder
        :param count: the concat num
        :return:
            the mean result
        '''
        if isinstance(self_placeholder, list):
            for i in range(len(self_placeholder)):
                result[i] = np.mean(np.array(result[i]), axis=0)
        else:
            result /= count
        return result

    @staticmethod
    def __type2len(self_placeholder, feed_data):
        '''
        :param self_placeholder: the placeholder in self.inputs,self.standard_outputs,self.net_configs
        :param feed_data: the data feed to self_placeholder
        :return:
            the feed_data length
        '''
        if isinstance(self_placeholder, list):
            if isinstance(feed_data[0], list):
                length = len(feed_data[0])
            elif isinstance(feed_data[0], np.ndarray):
                length = feed_data[0].shape[0]
            else:
                raise TypeError("only support list and numpy.ndarray type")
        else:
            if isinstance(feed_data, list):
                length = len(feed_data)
            elif isinstance(feed_data, np.ndarray):
                length = feed_data.shape[0]
            else:
                raise TypeError("only support list and numpy.ndarray type")
        return length

    @staticmethod
    def __type2concat(data1, data2):
        '''
        :param data1: front data
        :param data2: rear data
        :return:
            concat data
        '''
        if data1 is None:
            return data2
        if data2 is None:
            return data1
        if isinstance(data1, list):
            if isinstance(data2, list):
                data = data1 + data2
            else:
                raise TypeError("data1 and data2 should be same type")
        elif isinstance(data1, np.ndarray):
            if isinstance(data2, np.ndarray):
                data = np.r_[data1, data2]
            else:
                raise TypeError("data1 and data2 should be same type")
        else:
            raise TypeError("only support list and numpy.ndarray")
        return data

    @staticmethod
    def __type2clone(data):
        '''
        :param data:
        :return: clone data
        '''
        if isinstance(data, list):
            return list(data)
        elif isinstance(data, np.ndarray):
            return np.array(data)
        else:
            raise TypeError("only support list and numpy.ndarray type")
        return data

    def __type2batch(self, self_placeholder, feed_data, position, batch_size):
        '''
        :param self_placeholder: the placeholder in self.inputs,self.standard_outputs,self.net_configs
        :param feed_data: the data feed to self_placeholder
        :param position: the data position
        :param batch_size: the batch size
        :return:
            batch feed data, batch len
        '''
        batch_feed_data = []
        batch_len = batch_size
        length = self.__type2len(self_placeholder,feed_data)
        if isinstance(self_placeholder, list):
            if position + batch_size > length:
                batch_len = length - position
                for i in range(len(self_placeholder)):
                    temp = feed_data[i][position:]
                    res = batch_size - batch_len
                    while res > length:
                        temp = self.__type2concat(temp, feed_data[i])
                        res -= length
                    if res > 0:
                        temp = self.__type2concat(temp, feed_data[i][0:res])
                    batch_feed_data.append(self.__type2clone(temp))
            else:
                for i in range(len(self_placeholder)):
                    batch_feed_data.append(self.__type2clone(feed_data[i][position:position+batch_size]))
        elif isinstance(self_placeholder,tf.Tensor):
            if position + batch_size > length:
                batch_feed_data = feed_data[position:]
                res = batch_size - self.__type2len(self_placeholder,feed_data[position:])
                batch_len = self.__type2len(self_placeholder,feed_data[position:])
                while res > length:
                    batch_feed_data = self.__type2concat(batch_feed_data, feed_data)
                    res -= length
                if res > 0:
                    batch_feed_data = self.__type2concat(batch_feed_data, feed_data[0:res])
            else:
                batch_feed_data = feed_data[position:position + batch_size]

        else:
            raise TypeError("self_placeholder only support to List[Tensor] and Tensor type")

        return batch_feed_data, batch_len

    def __generator_batch(self, batch_size, inputs_feed, outputs_feed=None, shuffle=False):
        if self.num_parallel_calls > 0:
            batch_size = batch_size * self.num_parallel_calls
        position = 0
        length = self.__type2len(self.inputs, inputs_feed)
        if shuffle:
            shuffle_index = random.sample(range(length), length)
            if isinstance(self.inputs, list):
                for i in range(len(self.inputs)):
                    inputs_feed[i] = np.array(inputs_feed[i])[shuffle_index]
            else:
                inputs_feed = np.array(inputs_feed)[shuffle_index]
            if outputs_feed is not None:
                if isinstance(self.standard_outputs, list):
                    for i in range(len(self.standard_outputs)):
                        outputs_feed[i] = np.array(outputs_feed[i])[shuffle_index]
                else:
                    outputs_feed = np.array(outputs_feed)[shuffle_index]
        while position < length:
            batch_inputs_feed, batch_len = self.__type2batch(self.inputs, inputs_feed, position, batch_size)
            if outputs_feed is not None:
                batch_outputs_feed, _ = self.__type2batch(self.standard_outputs, outputs_feed, position, batch_size)
            else:
                batch_outputs_feed = None
            position = position + batch_size
            is_one_epoch = False
            if position >= length:
                is_one_epoch = True
            yield (batch_inputs_feed, batch_outputs_feed, batch_len, is_one_epoch)
