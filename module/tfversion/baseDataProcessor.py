from .dataWrapper import *
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

class BaseDataProcessor:
    def __init__(self, features_typing_fn: FeatureTypingFunctions=None):
        self.features_typing_fn = features_typing_fn

    def creat_samples(self,*args,**kwargs)->List[InputSample]:
        '''
        the input_x,input_y of InputSample's key should be same to the self.features_typing_fn
        :param args:
        :param kwargs:
        :return:
        '''
        raise NotImplementedError

    def samples2features(self,samples: List[InputSample],*args,**kwargs)->List[InputFeatures]:
        '''
        the net_x,net_y of InputFeatures' key should be same to the self.features_typing_fn
        :param samples:
        :param args:
        :param kwargs:
        :return:
        '''
        raise NotImplementedError


class MnistDataProcessor(BaseDataProcessor):
    def __init__(self, features_typing_fn: FeatureTypingFunctions=None):
        if features_typing_fn is None:
            try:
                name_to_features={
                    "x": tf.io.FixedLenFeature(shape=[784,], dtype="float"),
                    "y": tf.io.FixedLenFeature(shape=[], dtype=tf.int64),
                    "is_real_sample": tf.io.FixedLenFeature(shape=[], dtype=tf.int64)
                }
            except:
                name_to_features = {
                    "x": tf.FixedLenFeature(shape=[784, ], dtype="float"),
                    "y": tf.FixedLenFeature(shape=[], dtype=tf.int64),
                    "is_real_sample": tf.FixedLenFeature(shape=[], dtype=tf.int64)
                }
            features_typing_fn = FeatureTypingFunctions({"x":FeatureTypingFunctions.float_feature},
                                                        name_to_features=name_to_features
                                                        , y_fns={"y": FeatureTypingFunctions.int64_feature})
        if features_typing_fn is None:
            raise ValueError("class FeatureTypingFunctions:features_typing_fn should provide")
        super(MnistDataProcessor, self).__init__(features_typing_fn)

    def creat_samples(self, xs,ys):
        samples = []
        for index,(x,y) in enumerate(zip(xs,ys)):
            input_x = {"x": x}
            input_y = {"y": y}
            sample = InputSample(index, input_x, input_y)
            samples.append(sample)
        return samples

    def samples2features(self,samples: List[InputSample]):
        input_features = []
        for sample in samples:
            feature = InputFeatures(sample.input_x, sample.input_y, True)
            input_features.append(feature)
        return input_features