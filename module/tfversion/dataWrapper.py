import tensorflow as tf
from typing import List, Union, Dict
import os.path as ospath
import collections
import numpy as np


class InputSample:
    def __init__(self, guid,input_x: Dict, input_y: Dict=None):
        '''
        :param guid: for unique the sample
        :param input_x: the sample data, e.g:{key:value,key:value}
        :param input_y: the sample answer e.g:{key:value,key:value}
        '''
        if guid is None:
            raise ValueError("the guid should provide")
        if input_x is None:
            raise ValueError("the input_x should provide")
        self.guid = guid
        self.input_x = input_x
        self.input_y = input_y


class PadSample:
    '''
    if a sample class is PadSample
        this sample is only for batch pad, if the real_sample num less than batch size, this class will be used
    '''
    def __int__(self, pad_value):
        self.pad_value = pad_value


class FeatureTypingFunctions:
    def __init__(self, x_fns: Dict, name_to_features: Dict[str, Union[tf.FixedLenFeature, tf.VarLenFeature,
                                                            tf.FixedLenSequenceFeature, tf.SparseFeature]],
                 y_fns: Dict=None, is_real_sample_fn=None):
        '''
        :param x_fns:  the x_fns should be a Dict with same key to net_x in InputFeatures, x_fns value should be in
                        [int64_feature, float_feature, bytes_feature]
        :param name_to_features:
        :param y_fns: same to x_fns
        :param is_real_sample_fn: a function in [int64_feature, float_feature, bytes_feature]
        '''
        if x_fns is None:
            raise ValueError("x_fns should provide")
        if name_to_features is None:
            raise ValueError("name_to_features should be provide")
        if is_real_sample_fn is None:
            is_real_sample_fn = FeatureTypingFunctions.int64_feature
        self.x_fns = x_fns
        self.y_fns = y_fns
        self.is_real_sample_fn = is_real_sample_fn
        self.name_to_features = name_to_features

    @classmethod
    def int64_feature(cls, values):
        f = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
        return f

    @classmethod
    def float_feature(cls, values):
        f = tf.train.Feature(float_list=tf.train.FloatList(value=list(values)))
        return f

    @classmethod
    def bytes_feature(cls, values):
        f = tf.train.Feature(bytes_list=tf.train.BytesList(value=list(values)))
        return f



class InputFeatures:
    def __init__(self, net_x: Dict, net_y: Dict=None, is_real_sample: bool=True):
        '''
        :param net_x: this should be a Dict converted by input_x in class InputSample, if is PadSample,
                      this should be a Dict same to input_x but value for padsample
        :param net_y: this should be a Dict converted by input_y in class InputSample, if is PadSample,
                      this should be a Dict same to input_x but value for padsample
        :param is_real_sample:  is InputSample True, is PadSample False
        '''
        if net_x is None:
            raise ValueError("the net_x should provide")
        self.net_x = net_x
        self.net_y = net_y
        self.is_real_sample = is_real_sample


class TFDataWrapper:
    def __init__(self):
        super(TFDataWrapper, self).__init__()

    def wrapper(self, all_features: List[InputFeatures], batch_size, gpu_num=0, is_train=True,
                drop_remainder=False, num_parallel_calls=None):
        '''
        :param all_features: the all data for network
        :param batch_size: batch size
        :param gpu_num: if use multi gpu, this should provide
        :param is_train:  is train set or not
        :param drop_remainder:  if len(all_features)%batch_size!=0, drop the next data or not
        :param num_parallel_calls: the data process with thread,if None,one thread
        :return:
            tf.data.Dataset,data with tensor,iterator_init
        '''
        if gpu_num>0:
            batch_size = batch_size*gpu_num
        net_data = {}
        for f in all_features:
            for x_key in f.net_x:
                if x_key in net_data:
                    net_data[x_key].append(f.net_x[x_key])
                else:
                    net_data[x_key] = []
                    net_data[x_key].append(f.net_x[x_key])
            for y_key in f.net_y:
                if y_key in net_data:
                    net_data[y_key].append(f.net_y[y_key])
                else:
                    net_data[y_key] = []
                    net_data[y_key].append(f.net_y[y_key])
            if "is_real_sample" not in net_data:
                net_data["is_real_sample"] = []
                net_data["is_real_sample"].append(f.is_real_sample)
            else:
                net_data["is_real_sample"].append(f.is_real_sample)
        # num_samples = len(all_features)
        for key in net_data:
            shape = [len(all_features)]
            temp = net_data[key][0]
            if isinstance(temp, np.ndarray):
                net_data[key] = np.array(net_data[key])
            while isinstance(temp, list) or isinstance(temp, np.ndarray):
                if isinstance(temp, list):
                    while isinstance(temp, list):
                        shape.append(len(temp))
                        temp = temp[0]
                if isinstance(temp, np.ndarray):
                    while isinstance(temp, np.ndarray):
                        shape.append(temp.shape[0])
                        temp = temp[0]
            net_data[key] = tf.constant(net_data[key],shape=shape)
        tf_data = tf.data.Dataset.from_tensor_slices(net_data)
        if is_train:
            tf_data = tf_data.repeat()
            tf_data = tf_data.shuffle(buffer_size=100)
        tf_data = tf_data.map(lambda x:x,num_parallel_calls)
        try:
            tf_data = tf_data.batch(batch_size,drop_remainder)
        except:
            tf_data = tf_data.batch(batch_size)
        it = tf_data.make_initializable_iterator()
        data = it.get_next()
        iterator_init = it.initializer
        return tf_data, data, iterator_init

    def __call__(self, all_features: List[InputFeatures], batch_size, gpu_num=0, is_train=True,
                 drop_remainder=False)->tf.data.Dataset:
        tf_data = self.wrapper(all_features,batch_size,gpu_num,is_train,drop_remainder)
        return tf_data



class TFRecordWrapper:
    def __init__(self, file_path: str, feature_typing_fn: FeatureTypingFunctions):
        '''
        :param file_path: TFRecord file path
        :param feature_typing_fn:a FeatureTypingFunctions for net_x,net_y,is_real_sample
        '''
        if file_path is None or file_path == "":
            raise ValueError("the file_path should provide")
        else:
            file_list = file_path.rsplit("/",maxsplit=1)
            if len(file_list)==1:
                file_list = file_path.rsplit("\\",maxsplit=1)
                if len(file_list)==1:
                    self.file_path = file_path
                else:
                    if ospath.isdir(file_list[0]):
                        self.file_path = file_path
                    else:
                        raise ValueError("the file_path not correct")
            else:
                if ospath.isdir(file_list[0]):
                    self.file_path = file_path
                else:
                    raise ValueError("the file_path not correct")
        if feature_typing_fn is None:
            raise ValueError("feature_typing_fn should provide")
        self.feature_typing_fn = feature_typing_fn
        self.writer = tf.python_io.TFRecordWriter(self.file_path)

    def write(self, input_features: Union[InputFeatures, List[InputFeatures]], is_complete=True):
        '''
        :param input_features: the sample(InputFeatures) list or one for to write to TFRecord
        :param is_complete:  TFRecord is complete
        :return:
        '''
        if self.writer is None:
            writer = tf.python_io.TFRecordWriter(self.file_path)
            self.writer = writer
        else:
            writer = self.writer
        if isinstance(input_features, list):
            for f in input_features:
                features = collections.OrderedDict()
                for x_key in f.net_x:
                    if isinstance(f.net_x[x_key], list):
                        features[x_key] = self.feature_typing_fn.x_fns[x_key](f.net_x[x_key])
                    elif isinstance(f.net_x[x_key], np.ndarray):
                        features[x_key] = self.feature_typing_fn.x_fns[x_key](f.net_x[x_key].tolist())
                    else:
                        features[x_key] = self.feature_typing_fn.x_fns[x_key]([f.net_x[x_key]])
                for y_key in f.net_y:
                    if isinstance(f.net_y[y_key], list):
                        features[y_key] = self.feature_typing_fn.y_fns[y_key](f.net_y[y_key])
                    elif isinstance(f.net_y[y_key], np.ndarray):
                        features[y_key] = self.feature_typing_fn.y_fns[y_key](f.net_y[y_key].tolist())
                    else:
                        features[y_key] = self.feature_typing_fn.y_fns[y_key]([f.net_y[y_key]])
                features["is_real_sample"] = self.feature_typing_fn.is_real_sample_fn([f.is_real_sample])
                tf_sample = tf.train.Example(features=tf.train.Features(feature=features))
                writer.write(tf_sample.SerializeToString())
        else:
            features = collections.OrderedDict()
            for x_key in input_features.net_x:
                if isinstance(input_features.net_x[x_key], list):
                    features[x_key] = self.feature_typing_fn.x_fns[x_key](input_features.net_x[x_key])
                elif isinstance(input_features.net_x[x_key], np.ndarray):
                    features[x_key] = self.feature_typing_fn.x_fns[x_key](input_features.net_x[x_key].tolist())
                else:
                    features[x_key] = self.feature_typing_fn.x_fns[x_key]([input_features.net_x[x_key]])
            for y_key in input_features.net_y:
                if isinstance(input_features.net_y[y_key], list):
                    features[y_key] = self.feature_typing_fn.y_fns[y_key](input_features.net_y[y_key])
                elif isinstance(input_features.net_x[x_key], np.ndarray):
                    features[y_key] = self.feature_typing_fn.y_fns[y_key](input_features.net_y[y_key].tolist())
                else:
                    features[y_key] = self.feature_typing_fn.y_fns[y_key]([input_features.net_y[y_key]])
            features["is_real_sample"] = self.feature_typing_fn.is_real_sample_fn(input_features.is_real_sample)
            tf_sample = tf.train.Example(features=tf.train.Features(feature=features))
            writer.write(tf_sample.SerializeToString())
        if is_complete:
            writer.close()
            self.writer = None

    def __decode_record(self, record):
        '''
        :param record: one data from TFRcord file
        :return: the train example
        '''
        sample = tf.parse_single_example(record, self.feature_typing_fn.name_to_features)
        for name in list(sample.keys()):
            t = sample[name]
            if t.dtype == tf.int64:
                t = tf.to_int32(t)
            sample[name] = t
        return sample

    def read(self, is_train: bool, batch_size, gpu_num=0, drop_remainder=False,num_parallel_calls=None):
        '''
        :param is_train: is train set or not,if is train set, it will be repeat and shuffle
        :param batch_size: batch size for cpu or one GPU
        :param gpu_num: if use gpu to train or test or evalution, it should provide
        :param drop_remainder:  if the set is less than batch_size or batch_size*gpu_num,drop it or not
        :param num_parallel_calls: the data process with thread,if None,one thread
        :return:
            tf.data.Dataset,data with tensor,iterator_init
        '''
        if gpu_num > 0:
            batch_size = batch_size*gpu_num
        tf_record = tf.data.TFRecordDataset(self.file_path)
        if is_train:
            tf_record = tf_record.repeat()
            tf_record = tf_record.shuffle(buffer_size=100)

        tf_record = tf_record.map(lambda record: self.__decode_record(record),num_parallel_calls)
        try:
            tf_record = tf_record.batch(batch_size, drop_remainder)
        except:
            tf_record = tf_record.batch(batch_size)
        it = tf_record.make_initializable_iterator()
        data = it.get_next()
        iterator_init = it.initializer
        return tf_record, data, iterator_init
