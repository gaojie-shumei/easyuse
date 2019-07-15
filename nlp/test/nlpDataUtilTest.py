'''
Created on 2019年7月10日

@author: gaojiexcq
'''
from datautil import nlpDataUtil
train_path = "../ner/data/conll2003/eng.train"
v_path = "../ner/data/conll2003/eng.testa"
test_path = "../ner/data/conll2003/eng.testb"


'''
读取数据
data,data_pos_tag,data_chunk_tag是x数据
label是y数据
label_index 是所有的类别，包括实体和非实体
'''
def read_data(filepath,encoding="utf-8",position=0,read_data_size=None,padding_str="<pad>"):
    data = []
    data_pos_tag = []
    data_chunk_tag = []
    label = []
    label_index = []
    
    sub_data = []
    sub_data_pos_tag = []
    sub_data_chunk_tag = []
    sub_label = []
    with open(filepath,mode="r+",encoding=encoding) as f:
        f.seek(position)
        count = 0
        while True:
            line = f.readline()
            if line=="":
                break
            line = line.strip()
            if position!=0 and line.strip()!="":
                line_split = line.split(" ")
                sub_data.append(line_split[0])
                sub_data_pos_tag.append(line_split[1])
                sub_data_chunk_tag.append(line_split[2])
                sub_label.append(line_split[3])
                label_index.append(line_split[3])
            elif len(sub_data)>0:
                data.append(sub_data)
                data_pos_tag.append(sub_data_pos_tag)
                data_chunk_tag.append(sub_data_chunk_tag)
                label.append(sub_label)
                sub_data = []
                sub_data_pos_tag = []
                sub_data_chunk_tag = []
                sub_label = []
                count+=1
            position = 1
            if read_data_size!=None and count==read_data_size:
                break
        position = f.tell()
        label_index = list(set(label_index))
        label_index.sort()
        label_index = [padding_str]+label_index
    return data,data_pos_tag,data_chunk_tag,label,position,label_index


#word2vec test
data,data_pos_tag,data_chunk_tag,label,position,label_index = read_data(train_path,encoding="utf-8",position=0,
                                                                        read_data_size=None,padding_str="<pad>")

datautil = nlpDataUtil.NLPDataUtil(use_for_bert=True,word2vec_path=None,label_setlist=label_index)
data_model = datautil.word2vec(sentences=data, size=5, min_count=1)
print(len(data_model.wv.vocab))
 
data,data_pos_tag,data_chunk_tag,label,position,label_index = read_data(v_path,encoding="utf-8",position=0,
                                                                        read_data_size=None,padding_str="<pad>")
 
data_model = datautil.word2vec(sentences=data, size=5, min_count=1)
print(len(data_model.wv.vocab))


#next batch
data,label,batch_data,batch_label,position = datautil.next_batch(batch_size=2, data_x=data, data_y=label, 
                                                                 position=0, shuffle=True, random_state=1)
print(batch_data,"\n",batch_label)

#padding
pad_data,pad_label,actual_lengths = datautil.padding(batch_data, batch_label)
print(pad_data,"\n",pad_label,"\n",actual_lengths)
  
#format
batch_x,batch_y = datautil.format(pad_data, pad_label)
print(batch_x,"\n",batch_y)

from bert import tokenization
base_dir = "../bert/base_model/cased_L-12_H-768_A-12"
tokenizer = tokenization.FullTokenizer(vocab_file=base_dir + "/vocab.txt",do_lower_case=False)
batch_tokens,batch_input_ids,batch_input_mask,batch_segment_ids,batch_labels_index,\
actual_lengths = datautil.ner_bert_data_convert(batch_data, batch_label, tokenizer)
print(batch_tokens)
print("#"*33)
print(batch_input_ids)
print("#"*33)
print(batch_input_mask)
print("#"*33)
print(batch_segment_ids)
print("#"*33)
print(batch_labels_index)
print("#"*33)
print(actual_lengths)
print("#"*33)




