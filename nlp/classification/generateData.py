from bert import tokenization
import pandas as pd
import numpy as np
import random

def read_data(csvPath):
    df = pd.read_csv(csvPath)
    data = np.array(df)
    label = data[:, 3].tolist()
    data = data[:,2].tolist()
    return data,label

def next_batch(batch_size,data_x:list,data_y:list=None,position=0,shuffle=True,random_state=random.randint(0,1000)):
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
    if batch_size>=len(temp_data_x):
        batch_x = temp_data_x
        if data_y is not None:
            batch_y = temp_data_y
        else:
            batch_y = None
    else:
        batch_x = temp_data_x[0:batch_size]
        if data_y is not None:
            batch_y = temp_data_y[0:batch_size]
        else:
            batch_y = None
    position += batch_size
    return data_x,data_y,batch_x,batch_y,position

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
def convert_batch_data(batch_data,tokenizer:tokenization.FullTokenizer):
    batch_tokens, batch_input_ids, batch_input_mask, batch_segment_ids, actual_lengths = [], [], [], [], []
    max_len = 0
    for i in range(len(batch_data)):
        tokens, input_ids, input_mask, segment_ids, actual_length = convert_single_sample(batch_data[i],tokenizer)
        if max_len < actual_length:
            max_len = actual_length
        batch_tokens.append(np.array(tokens).tolist())
        batch_input_ids.append(np.array(input_ids).tolist())
        batch_input_mask.append(np.array(input_mask).tolist())
        batch_segment_ids.append(np.array(segment_ids).tolist())
        actual_lengths.append(actual_length)
    actual_lengths = np.array(actual_lengths)
    for i in range(len(batch_data)):
        batch_input_ids[i] = np.array(batch_input_ids[i]+[0]*(max_len-actual_lengths[i]))
        batch_input_mask[i] = np.array(batch_input_mask[i] + [0] * (max_len - actual_lengths[i]))
        batch_segment_ids[i] = np.array(batch_segment_ids[i]+[0]*(max_len-actual_lengths[i]))
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