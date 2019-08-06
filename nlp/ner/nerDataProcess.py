import pandas as pd
import numpy as np
from bert import tokenization
import random


def read_data(dataPath, pad_word="[PAD]", use_bert=True):
    df = pd.read_json(dataPath, orient="records", lines=True)
    data = np.array(df[["id", "text", "label"]])
    unique_label = []
    for i in range(data.shape[0]):
        unique_label = list(set(unique_label + list(set(data[i, 2]))))
    unique_label.sort()
    textlist = data[:, 1].tolist()
    labellist = data[:, 2].tolist()
    if use_bert:
        '''
            [FIX]表示本为一个词，但是被bert分成了多个词时除第一个词以外的词的label
        '''
        unique_label = [pad_word, "[FIX]"] + unique_label
    else:
        unique_label = [pad_word] + unique_label
    return textlist, labellist, unique_label


def convert_one_sample(sample: list, sample_label: list=None, tokenizer: tokenization.FullTokenizer=None, max_len=512,
                       unique_label: list = None):
    if tokenizer is None:
        raise RuntimeError("tokenizer should be provide")
    if sample_label is not None and unique_label is None:
        raise RuntimeError("if sample label not None, the unique label also should be not None")

    tokens = []
    if sample_label is not None:
        label = []
    else:
        label = None
    for i in range(len(sample)):
        token = tokenizer.tokenize(sample[i])
        for j in range(len(token)):
            tokens.append(token[j])
            if j == 0:
                if sample_label is not None:
                    label.append(unique_label.index(sample_label[i]))
            else:
                if sample_label is not None:
                    label.append(unique_label.index("[FIX]"))

    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    if len(input_ids) > max_len:
        input_ids = input_ids[0:max_len]
        if sample_label is not None:
            label = label[0:max_len]
    input_mask = [1] * len(input_ids)
    segment_ids = [0] * len(input_ids)
    return tokens, input_ids, input_mask, segment_ids, label, len(input_ids)


def convert_batch_data(batch_sample: list, batch_sample_label: list=None, tokenizer: tokenization.FullTokenizer=None,
                       max_len=512, unique_label: list=None, pad_word="[PAD]"):
    if tokenizer is None:
        raise RuntimeError("tokenizer should be provide")
    if batch_sample_label is not None and unique_label is None:
        raise RuntimeError("if sample label not None, the unique label also should be not None")

    batch_input_ids = []
    batch_tokens = []
    batch_input_mask = []
    batch_segment_ids = []
    if batch_sample_label is not None:
        batch_label = []
    else:
        batch_label = None

    use_max_len = 0
    squence_lengths = []
    for i in range(len(batch_sample)):
        if batch_sample_label is not None:
            tokens, input_ids, input_mask, segment_ids, label, lengths = convert_one_sample(batch_sample[i],
                                                                                   batch_sample_label[i], tokenizer,
                                                                                   max_len, unique_label)
            batch_label.append(label)
        else:
            tokens, input_ids, input_mask, segment_ids, label,lengths = convert_one_sample(batch_sample[i], None, tokenizer,
                                                                                   max_len, unique_label)
        squence_lengths.append(lengths)
        if len(input_ids) > use_max_len:
            use_max_len = len(input_ids)
        batch_tokens.append(tokens)
        batch_input_ids.append(input_ids)
        batch_input_mask.append(input_mask)
        batch_segment_ids.append(segment_ids)
    squence_lengths = np.array(squence_lengths)
    for i in range(len(batch_sample)):
        if batch_sample_label is not None:
            batch_label[i] = np.array(batch_label[i] + [unique_label.index(pad_word)] * (use_max_len-len(batch_label[i])))
        batch_input_ids[i] = np.array(batch_input_ids[i] + [0] * (use_max_len - len(batch_input_ids[i])))
        batch_input_mask[i] = np.array(batch_input_mask[i] + [0] * (use_max_len - len(batch_input_mask[i])))
        batch_segment_ids[i] = np.array([0] * use_max_len)
        batch_tokens[i] = np.array(batch_tokens[i] + [pad_word] * (use_max_len - len(batch_tokens[i])))
    if batch_sample_label is not None:
        batch_label = np.array(batch_label).astype(np.int32)
    batch_input_ids = np.array(batch_input_ids).astype(np.int32)
    batch_input_mask = np.array(batch_input_mask).astype(np.int32)
    batch_segment_ids = np.array(batch_segment_ids).astype(np.int32)
    batch_tokens = np.array(batch_tokens)
    return batch_tokens, batch_input_ids, batch_input_mask, batch_segment_ids, batch_label, squence_lengths


def generator_batch(textlist: list, batch_size, shuffle=True, random_state=random.randint(0,1000),labellist: list=None):
    position = 0
    while position < len(textlist):
        temp_t = textlist[position:]
        if labellist is not None:
            temp_l = labellist[position:]
        else:
            temp_l = None
        if shuffle:
            random.seed(random_state)
            random.shuffle(temp_t)
            if labellist is not None:
                random.seed(random_state)
                random.shuffle(temp_l)
        textlist = textlist[0:position] + temp_t
        if labellist is not None:
            labellist = labellist[0:position] + temp_l
        if batch_size < len(temp_t):
            batch_sample = textlist[position:position + batch_size]
            if labellist is not None:
                batch_sample_label = labellist[position:position + batch_size]
        else:
            batch_sample = textlist[position:]
            if labellist is not None:
                batch_sample_label = labellist[position:]
        if labellist is None:
            batch_sample_label = None
        position += batch_size
        yield (batch_sample, batch_sample_label)


def next_batch(textlist: list, position, batch_size, shuffle=True, random_state=random.randint(0, 1000),
               labellist: list=None):
    temp_t = textlist[position:]
    if labellist is not None:
        temp_l = labellist[position:]
    else:
        temp_l = None
    if shuffle:
        random.seed(random_state)
        random.shuffle(temp_t)
        if labellist is not None:
            random.seed(random_state)
            random.shuffle(temp_l)
    textlist = textlist[0:position] + temp_t
    if labellist is not None:
        labellist = labellist[0:position] + temp_l
    if batch_size<len(temp_t):
        batch_sample = textlist[position:position+batch_size]
        if labellist is not None:
            batch_sample_label = labellist[position:position+batch_size]
    else:
        batch_sample = textlist[position:]
        if labellist is not None:
            batch_sample_label = labellist[position:]
    if labellist is None:
        batch_sample_label = None
    position += batch_size
    return textlist, labellist, batch_sample, batch_sample_label, position


# if __name__ == '__main__':
#     dataPath = "data/ner.json"
#     read_data(dataPath)
