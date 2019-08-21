from bert import tokenization
import pandas as pd
import numpy as np
import random
import math
from module.tfversion import baseDataProcessor
import datautil.nlpDataUtil as nlpDataUtil
import tensorflow as tf


class GammaWord2VecDataProcessor(baseDataProcessor.BaseDataProcessor):
    def __init__(self, max_len, word2vec_size=768, features_typing_fn=None):
        if features_typing_fn is None:
            x_fns = {"x": baseDataProcessor.FeatureTypingFunctions.float_feature,
                     "length": baseDataProcessor.FeatureTypingFunctions.int64_feature}
            y_fns = {"y": baseDataProcessor.FeatureTypingFunctions.int64_feature}
            name_to_features = {
                "x": tf.FixedLenFeature(shape=[max_len * word2vec_size], dtype="float"),
                "length": tf.FixedLenFeature(shape=[], dtype=tf.int64),
                "is_real_sample": tf.FixedLenFeature(shape=[], dtype=tf.int64),
                "y": tf.FixedLenFeature(shape=[], dtype=tf.int64)
            }
            features_typing_fn = baseDataProcessor.FeatureTypingFunctions(x_fns, name_to_features, y_fns)
        super(GammaWord2VecDataProcessor, self).__init__(features_typing_fn)

    def creat_samples(self, data, label, datautil: nlpDataUtil.NLPDataUtil, max_length):
        xs, _, lengths = datautil.padding(data, max_length=max_length)
        xs, _ = datautil.format(xs)
        samples = []
        for index, (x, length, y) in enumerate(zip(xs, lengths, label)):
            sample = baseDataProcessor.InputSample(index, {"x": x, "length": length},{"y": y})
            samples.append(sample)
        return samples

    def samples2features(self, samples: baseDataProcessor.List[baseDataProcessor.InputSample]):
        features = []
        for sample in samples:
            feature = baseDataProcessor.InputFeatures(sample.input_x, sample.input_y, True)
            features.append(feature)
        return features

def read_classification_data(jsonPath, depreated_text="DirtyDeedsDoneDirtCheap", data_augmentation_label=2,
                             test_percent=0.5, keyword_path="../data/keyword.xlsx"):
    '''读取classification 数据  id  text  label'''
    df = pd.read_json(jsonPath, orient="records", encoding=None, lines=True)
    '''获取负面关键词'''
    keyword = pd.read_excel(keyword_path)
    keyword = keyword.fillna("")
    keyword = np.array(keyword).reshape(-1)
    keyword = keyword[keyword!=""]
    # print(np.isnan(keyword))
    print(keyword)
    '''将分隔符去除'''
    if depreated_text is not None and depreated_text != "":
        df["text"].replace(depreated_text + "(:|：)*([0-9]*)", "", regex=True, inplace=True)
    '''DataFrame to numpy  按照 id,text,label的格式来'''
    data = np.c_[np.array(df["id"]), np.array(df["text"]), np.array(df["label"])]
    # print("data:",data)
    '''删除text为空的数据'''
    data = np.delete(data, np.where(data[:, 1] == "")[0].reshape(-1), axis=0)
    '''删除未知label数据'''
    delete_index = []
    for i in range(data.shape[0]):
        if data[i, 2] not in [0, 1, 2]:
            delete_index.append(i)
    data = np.delete(data, delete_index, axis=0)
    '''获取需要数据增强的类别数据'''
    need_data_augmentation = data[data[:, 2] == data_augmentation_label]
    print("need_data_augmentation.shape", need_data_augmentation.shape)
    np.random.shuffle(need_data_augmentation)
    print("need_data_augmentation.shape", need_data_augmentation.shape)
    test_num = math.ceil(need_data_augmentation.shape[0]*test_percent)
    test_data = need_data_augmentation[0:test_num]
    other_data = data[data[:, 2] != data_augmentation_label]
    '''数据增强的数据确定'''
    need_data_augmentation = need_data_augmentation[test_num:]
    print("need_data_augmentation.shape", need_data_augmentation.shape)
    '''数据增强'''
    data_augmentation_data = data_augmentation(need_data_augmentation, keyword, split_regex=" ")
    '''其他类别数据'''
    # other_data = data[data[:, 2] != data_augmentation_label]
    # print(other_data.shape,data_augmentation_data.shape)
    '''训练数据拼接'''
    if data_augmentation_data.shape[0] <= other_data.shape[0]:
        sample_index = random.sample(range(other_data.shape[0]), data_augmentation_data.shape[0])
        data = np.r_[other_data[sample_index], data_augmentation_data]
    else:
        data = np.r_[other_data, data_augmentation_data[0:other_data.shape[0]]]
    if data.shape[0] > 25600:
        data = data[-25600:]
    '''数据打乱'''
    np.random.shuffle(data)
    test_text = test_data[:, 1].tolist()
    test_label = test_data[:, 2].astype(np.int32).tolist()
    train_text = data[:, 1].tolist()
    train_label = data[:, 2].astype(np.int32).tolist()
    return train_text, train_label, test_text, test_label


def data_augmentation(need_data_augmentation: np.ndarray, keyword: np.ndarray, split_regex=" "):
    data = []
    '''数据增强    1、对文本的词语顺序进行打乱增强
                   2、在当前文本中插入负面关键词并打乱顺序增强
    '''
    for i in range(need_data_augmentation.shape[0]):
        for _ in range(3):
            data.append(need_data_augmentation[i])
        id = need_data_augmentation[i, 0]
        text = need_data_augmentation[i, 1]
        label = need_data_augmentation[i, 2]
        textsplit = text.split(split_regex)
        for word in keyword:
            if isinstance(word, str)==False:
                continue
            ts = np.array(textsplit).tolist()
            insert_index = random.randint(0, len(ts))
            ts.insert(insert_index, word)
            data.append(np.array([id, " ".join(ts), label]))
            shuffle_index = random.sample(range(len(ts)), len(ts))
            ts = np.array(ts)[shuffle_index]
            data.append(np.array([id, " ".join(ts), label]))
    data = np.array(data)
    return data


def generator_batch(batch_size, data_x: list, data_y: list=None,num_parallel_calls=0, shuffle=True, random_state=random.randint(0, 1000)):
    if num_parallel_calls > 0:
        batch_size = batch_size * num_parallel_calls
    position = 0
    while position < len(data_x):
        temp_data_x = data_x[position:]
        if data_y is not None:
            temp_data_y = data_y[position:]
        if shuffle:
            random.seed(random_state)
            random.shuffle(temp_data_x)
            if data_y is not None:
                random.seed(random_state)
                random.shuffle(temp_data_y)
        data_x = data_x[0:position] + temp_data_x
        if data_y is not None:
            data_y = data_y[0:position] + temp_data_y
        if position + batch_size >= len(data_x):
            if batch_size <= len(data_x):
                batch_x = data_x[position:] + data_x[0:batch_size-len(data_x[position:])]
                if data_y is not None:
                    batch_y = data_y[position:] + data_y[0:batch_size-len(data_x[position:])]
                else:
                    batch_y = None
            else:
                res = batch_size - len(data_x[position:])
                batch_x = data_x[position:]
                if data_y is not None:
                    batch_y = data_y[position:]
                else:
                    batch_y = None
                while res > len(data_x):
                    batch_x = batch_x + data_x
                    if data_y is not None:
                        batch_y = batch_y + data_y
                    res -= len(data_x)
                if res > 0:
                    batch_x = batch_x + data_x[0:res]
                    if data_y is not None:
                        batch_y = batch_y + data_y[0:res]
        else:
            batch_x = temp_data_x[0:batch_size]
            if data_y is not None:
                batch_y = temp_data_y[0:batch_size]
            else:
                batch_y = None
        if shuffle:
            random.seed(random_state)
            random.shuffle(batch_x)
            if data_y is not None:
                random.seed(random_state)
                random.shuffle(batch_y)
        position += batch_size
        if position > len(data_x):
            flag = 1
        else:
            flag = 0
        yield (batch_x, batch_y,flag)


def next_batch(batch_size, data_x: list, data_y: list=None, position=0, shuffle=True,
               random_state=random.randint(0, 1000)):
    temp_data_x = data_x[position:]
    if data_y is not None:
        temp_data_y = data_y[position:]
    if shuffle:
        random.seed(random_state)
        random.shuffle(temp_data_x)
        if data_y is not None:
            random.seed(random_state)
            random.shuffle(temp_data_y)
    data_x = data_x[0:position] + temp_data_x
    if data_y is not None:
        data_y = data_y[0:position] + temp_data_y
    if position + batch_size >= len(data_x):
        if batch_size <= len(data_x):
            batch_x = data_x
            if data_y is not None:
                batch_y = data_y
            else:
                batch_y = None
        else:
            res = batch_size
            batch_x = []
            if data_y is not None:
                batch_y = []
            else:
                batch_y = None
            while res > len(data_x):
                batch_x = batch_x + data_x
                if data_y is not None:
                    batch_y = batch_y + data_y
                res -= len(data_x)
            if res > 0:
                batch_x = batch_x + data_x[0:res]
                if data_y is not None:
                    batch_y = batch_y + data_y[0:res]
    else:
        batch_x = temp_data_x[0:batch_size]
        if data_y is not None:
            batch_y = temp_data_y[0:batch_size]
        else:
            batch_y = None
    if shuffle:
        random.seed(random_state)
        random.shuffle(batch_x)
        if data_y is not None:
            random.seed(random_state)
            random.shuffle(batch_y)
    position += batch_size
    return data_x, data_y, batch_x, batch_y, position


def convert_single_sample(sample: list, tokenizer: tokenization.FullTokenizer):
    '''
    :sample: a text
    :tokenizer: same to ner_bert_data_convert paramer of tokenizer
    '''
    tokens = []
    segment_ids = []
    # labels = []
    tokens.append("[CLS]")
    segment_ids.append(0)  # use to token_type_ids,tag the sentence is the first or second,first is 0,second is 1
    token = tokenizer.tokenize(sample)
    for j in range(len(token)):
        tokens.append(token[j])
        segment_ids.append(0)
    input_ids = tokenizer.convert_tokens_to_ids(tokens)  # set word to ids

    # create mask,if mask=1,it show the position has word,if mask=0,then not
    input_mask = [1] * len(input_ids)
    actual_length = len(input_ids)
    return tokens, input_ids, input_mask, segment_ids, actual_length


def convert_batch_data(batch_data, tokenizer:tokenization.FullTokenizer,bert_max_len=512):
    batch_tokens, batch_input_ids, batch_input_mask, batch_segment_ids, actual_lengths = [], [], [], [], []
    for i in range(len(batch_data)):
        tokens, input_ids, input_mask, segment_ids, actual_length = convert_single_sample(batch_data[i],tokenizer)
        batch_tokens.append(np.array(tokens).tolist())
        batch_input_ids.append(np.array(input_ids).tolist())
        batch_input_mask.append(np.array(input_mask).tolist())
        batch_segment_ids.append(np.array(segment_ids).tolist())
        actual_lengths.append(actual_length)
    actual_lengths = np.array(actual_lengths)
    for i in range(len(batch_data)):
        if bert_max_len-actual_lengths[i]>=0:
            batch_input_ids[i] = np.array(batch_input_ids[i]+[0]*(bert_max_len-actual_lengths[i]))
            batch_input_mask[i] = np.array(batch_input_mask[i] + [0] * (bert_max_len - actual_lengths[i]))
            batch_segment_ids[i] = np.array(batch_segment_ids[i]+[0]*(bert_max_len-actual_lengths[i]))
        else:
            batch_input_ids[i] = np.array(batch_input_ids[i][0:bert_max_len])
            batch_input_mask[i] = np.array(batch_input_mask[i][0:bert_max_len])
            batch_segment_ids[i] = np.array(batch_segment_ids[i][0:bert_max_len])
            actual_lengths[i] = bert_max_len
    return batch_input_ids, batch_input_mask, batch_segment_ids, actual_lengths
# if __name__=="__main__":
    # data, label = read_data("./data/sub.csv")
    # data,label,batch_data,batch_label,position = next_batch(5,data,label,shuffle=False)
    # print(data[0:5])
    # print("@"*33)
    # print(batch_data)
    # print(data[7966])
    # bert_base_model_dir = "../bert/base_model/cased_L-12_H-768_A-12"
    #
    # tokenizer = tokenization.FullTokenizer(vocab_file=bert_base_model_dir+"/vocab.txt",do_lower_case=False)
    # print(tokenizer.tokenize(data[7966,2]))