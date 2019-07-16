'''
Created on 2019年7月12日

@author: gaojiexcq
'''
from nlp.ner.tfversion import nerNet
from datautil import nlpDataUtil
from nlp.test.read_data_for_conll2003 import *
import tensorflow as tf
from bert import modeling
from bert import tokenization
import tensorflow.contrib.layers as tcl
from tensorflow.contrib import rnn as tcr

#getRNNcell
'''
num_units:隐藏状态的size
use_peepholes：是否使用窥视孔连接    （LSTM的变种）
cell_clip:将隐藏状态控制在正负cell_clip之内
num_proj:可以简单理解为一个全连接，表示投射（projection）操作之后输出的维度，要是为None的话，表示不进行投射操作。
proj_clip:将投影矩阵的结果控制在正负proj_clip之内
forget_bias:对忘记门的限制
activation：激活函数
layer_norm:是否进行层标准化   输出为    norm*norm_gain + norm_shift
norm_gain:层标准化参数
norm_shift:层标准化参数
'''
def get_lstm_cell(num_units,use_peepholes=False,cell_clip=None,num_proj=None,proj_clip=None,forget_bias=1.0,activation=tf.nn.tanh,
                  layer_norm=False,norm_gain=1.0,norm_shift=0.0,input_keep_prob=1.0,state_keep_prob=1.0,output_keep_prob=1.0):
#     cell = tcr.LayerNormLSTMCell(num_units,use_peepholes=use_peepholes,cell_clip=cell_clip,num_proj=num_proj,proj_clip=proj_clip,
#                                  forget_bias=forget_bias,activation=activation,layer_norm=layer_norm,norm_gain=norm_gain,norm_shift=norm_shift)
    cell = tcr.LayerNormBasicLSTMCell(num_units,forget_bias=forget_bias,activation=activation,layer_norm=layer_norm,
                                      norm_gain=norm_gain,norm_shift=norm_shift)
#     cell = tcr.LSTMCell(num_units,use_peepholes=use_peepholes, cell_clip=cell_clip,num_proj=num_proj, proj_clip=proj_clip,
#                         forget_bias=forget_bias,activation=activation)
    cell = tcr.DropoutWrapper(cell,input_keep_prob=input_keep_prob,state_keep_prob=state_keep_prob,output_keep_prob=output_keep_prob)
    return cell



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

keep_prob = tf.placeholder("float")
istrain = tf.placeholder(dtype=tf.bool)
model_fn_placeholders = {"istrain":istrain,"keep_prob":keep_prob}
def model_fn(input,istrain,keep_prob,isbn,isln):
    '''
    :input: a tensor,shape [batch,max_len,word_embedding_size]=[None,None,word_embedding_size], usually a placeholder
    :istrain: for batch normalization
    '''
    regularizer_fn = tcl.l2_regularizer(0.0001)
    #bn
    if isbn:
        out = tf.layers.batch_normalization(input,training=istrain)
    else:
        out = tf.identity(input)
    #BiLSTM  编码
    with tf.variable_scope("encoder_lstm"):
        fw_cell = tf.nn.rnn_cell.MultiRNNCell([get_lstm_cell(128, use_peepholes=True, forget_bias=0.7, layer_norm=isln, norm_gain=0.8, 
                                                             norm_shift=1e-5, output_keep_prob=keep_prob) for _ in range(2)])
        bw_cell = tf.nn.rnn_cell.MultiRNNCell([get_lstm_cell(128, use_peepholes=True, forget_bias=0.7, layer_norm=isln, norm_gain=0.8, 
                                                             norm_shift=1e-5, output_keep_prob=keep_prob) for _ in range(2)])
        out,states = tf.nn.bidirectional_dynamic_rnn(fw_cell, bw_cell, out, sequence_length=actual_lengths_tensor,dtype="float")
    
    #拼接或者相加 
    if isinstance(out, tuple):
        out = tf.concat(out,axis=-1)
    
    #单向LSTM 解码
    with tf.variable_scope("decoder_lstm"):
        cell = tf.nn.rnn_cell.MultiRNNCell([get_lstm_cell(512,use_peepholes=True,forget_bias=0.7,layer_norm=isln,norm_gain=0.8, 
                                                          norm_shift=1e-5, output_keep_prob=keep_prob) for _ in range(2)])
        out,states = tf.nn.dynamic_rnn(cell,out,sequence_length=actual_lengths_tensor,dtype="float")
    return out

tokenizer = tokenization.FullTokenizer(vocab_file=bert_based_model_dir+"/vocab.txt",do_lower_case=False)

nernet = nerNet.NERNet(datautil=datautil,y=y,actual_lengths_tensor=actual_lengths_tensor,use_crf=True,tokenizer=tokenizer,
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
    model_fn_params={"istrain":istrain,"keep_prob":keep_prob,"isbn":True,"isln":True},
    model_fn_placeholders= model_fn_placeholders,
    regularizer_fn=regularizer_fn
)
print("\ntrain\n")
nernet.train(data, label, train_num=100, learn_rate=2e-5, batch_size=128,v_data=v_data,v_label=v_label,
             model_fn_placeholder_feed_tr={"istrain":True,"keep_prob":0.5},model_fn_placeholder_feed_pre={"istrain":False,"keep_prob":1},step_for_show=10)

print("\ntest\n")
nernet.test(t_data, t_label, batch_size=128, model_fn_placeholder_feed_pre={"istrain":False,"keep_prob":1})

t_data,predict_labels = nernet.predictNER(t_data, batch_size=128, model_fn_placeholder_feed_pre={"istrain":False,"keep_prob":1})

print("\npredict\n")
print(t_data[0])
print(t_label[0])
print(predict_labels[0])



