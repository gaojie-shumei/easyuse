'''
Created on 2019年7月12日

@author: gaojiexcq
'''
from nlp.ner.tensorflow import nerNet
from datautil import nlpDataUtil
from nlp.test.read_data_for_conll2003 import *
import tensorflow as tf
from bert import modeling
from bert import tokenization
import tensorflow.contrib.layers as tcl

def model_fn(input,istrain):
    '''
    :input: a tensor,shape [batch,max_len,word_embedding_size]=[None,None,word_embedding_size], usually a placeholder
    '''
    out = tf.layers.dense(input,units=40,activation=tf.nn.relu)
    out = tf.layers.batch_normalization(out,training=istrain)
    return out
istrain = tf.placeholder(dtype=tf.bool)
model_fn_placeholders = {"istrain":istrain}
train_path = "../../ner/data/conll2003/eng.train"
v_path = "../../ner/data/conll2003/eng.testa"
test_path = "../../ner/data/conll2003/eng.testb"

data,data_pos_tag,data_chunk_tag,label,position,label_index = read_data(train_path,encoding="utf-8",position=0,
                                                                        read_data_size=None,padding_str="<pad>")

v_data,v_data_pos_tag,v_data_chunk_tag,v_label,v_position,v_label_index = read_data(v_path,encoding="utf-8",position=0,
                                                                        read_data_size=None,padding_str="<pad>")

t_data,t_data_pos_tag,t_data_chunk_tag,t_label,t_position,t_label_index = read_data(v_path,encoding="utf-8",position=0,
                                                                        read_data_size=None,padding_str="<pad>")

datautil = nlpDataUtil.NLPDataUtil(use_for_bert=True, label_setlist=label_index)


'''
in git the bert base model is not exist,you should download it on network
'''
bert_based_model_dir = "../../bert/base_model/cased_L-12_H-768_A-12"

#bert model paramers set   placeholder
input_ids = tf.placeholder(shape=[None,None], dtype=tf.int32, name = "input_ids")
input_mask = tf.placeholder(shape=[None,None], dtype=tf.int32, name = "input_mask")
segment_ids = tf.placeholder(shape=[None,None], dtype=tf.int32, name = "segment_ids")
y = tf.placeholder(shape=[None,None],dtype=tf.int32,name="y")
actual_lengths_tensor = tf.placeholder(shape=[None],dtype=tf.int32,name="actual_lengths")

tokenizer = tokenization.FullTokenizer(vocab_file=bert_based_model_dir+"/vocab.txt",do_lower_case=False)

nernet = nerNet.NERNet(datautil=datautil,y=y,actual_lengths_tensor=actual_lengths_tensor,use_crf=False,tokenizer=tokenizer,
                       bert_base_model_path=bert_based_model_dir+"/bert_model.ckpt",checkpoint="./model/nerNetTest/")

regularizer_fn = tcl.l2_regularizer(0.0001)
nernet.create_NERModel_based_bert(
    bert_config = modeling.BertConfig.from_json_file(bert_based_model_dir+"/bert_config.json"), 
    bert_is_train=True, 
    input_ids=input_ids, 
    input_mask=input_mask, 
    segment_ids=segment_ids, 
    use_one_hot_embeddings=False, 
    model_fn=model_fn, 
    model_fn_params={"istrain":istrain},
    model_fn_placeholders= model_fn_placeholders,
    regularizer_fn=regularizer_fn
)
print("\ntrain\n")
nernet.train(data, label, train_num=100, learn_rate=2e-5, batch_size=32,v_data=v_data,v_label=v_label,
             model_fn_placeholder_feed_tr={"istrain":True},model_fn_placeholder_feed_pre={"istrain":False},step_for_show=2)

print("\ntest\n")
nernet.test(t_data, t_label, batch_size=32, model_fn_placeholder_feed_pre={"istrain":False})

t_data,predict_labels = nernet.predict(t_data, batch_size=32, model_fn_placeholder_feed_pre={"istrain":False})

print("\npredict\n")
print(t_data[0])
print(t_label[0])
print(predict_labels[0])



