'''
Created on 2019年7月16日

@author: gaojiexcq
'''
from nlp.bert.tfversion import bertTfApi
from datautil import nlpDataUtil
import tensorflow as tf
from bert import modeling
from bert import tokenization
from nlp.test.read_data_for_conll2003 import *
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

datautil = nlpDataUtil.NLPDataUtil(use_for_bert=True)

bert_based_model_dir = "../../bert/base_model/cased_L-12_H-768_A-12"
bert_config = modeling.BertConfig.from_json_file(bert_based_model_dir+"/bert_config.json")
bert_is_train = True
input_ids = tf.placeholder(dtype=tf.int32, shape=[None,None], name="input_ids")
input_mask = tf.placeholder(shape=[None,None], dtype=tf.int32, name = "input_mask")
segment_ids = tf.placeholder(shape=[None,None], dtype=tf.int32, name = "segment_ids")

bertfortf = bertTfApi.BertForTensorFlow(bert_config, bert_is_train, input_ids, 
        input_mask, segment_ids, use_one_hot_embeddings=False, scope=None)
model,restore_vars = bertfortf.create_bert_model()

out = (model.get_sequence_output() + model.get_embedding_output())/2
output_size = out.get_shape().as_list()[-1]
istrain = tf.placeholder(tf.bool)
keep_prob = tf.placeholder("float")
actual_lengths_tf = tf.placeholder(dtype=tf.int32, shape=[None], name="actual_length")
'''
isbn: if 'True' 表示要批标准化
isregularizer:if 'True' 表示要正则化
'''
def model(x,isbn,isregularizer,isln):
    regularizer = None
    if isregularizer:
        regularizer = None
    #bn
    if isbn:
        outputs = tf.layers.batch_normalization(x,training=istrain)
    else:
        outputs = tf.identity(x)
        
    #BiLSTM  编码
    with tf.variable_scope("encoder_lstm"):
        fw_cell = tf.nn.rnn_cell.MultiRNNCell([get_lstm_cell(num_units=128, use_peepholes=True, forget_bias=0.7, layer_norm=isln, norm_gain=0.8, 
                                                             norm_shift=1e-5, output_keep_prob=1) for _ in range(2)])
        bw_cell = tf.nn.rnn_cell.MultiRNNCell([get_lstm_cell(num_units=128, use_peepholes=True, forget_bias=0.7, layer_norm=isln, norm_gain=0.8, 
                                                             norm_shift=1e-5, output_keep_prob=1) for _ in range(2)])
        outputs,states = tf.nn.bidirectional_dynamic_rnn(fw_cell, bw_cell, outputs, sequence_length=actual_lengths_tf,dtype="float")
    
    #拼接或者相加 
    if isinstance(outputs, tuple):
        outputs = tf.concat(outputs,axis=-1)
    
    #转为输出要求的维度
    result = tf.layers.dense(tf.layers.flatten(outputs),units=256,activation=tf.nn.relu,kernel_regularizer=regularizer)
    
    #单向LSTM 解码
    with tf.variable_scope("decoder_lstm"):
        cell = tf.nn.rnn_cell.MultiRNNCell([get_lstm_cell(decoder_num_units=output_size,use_peepholes=True,forget_bias=0.7,layer_norm=isln,norm_gain=0.8, 
                                                          norm_shift=1e-5, output_keep_prob=keep_prob) for _ in range(2)])
        outputs,states = tf.nn.dynamic_rnn(cell,outputs,sequence_length=actual_lengths_tf,dtype="float")
    
    
    return outputs,result

outputs,result = model(out,True,False,True)
loss = tf.reduce_mean(outputs-out)

def train(data,train_num,learn_rate,batch_size):
    tokenizer = tokenization.FullTokenizer(vocab_file=bert_based_model_dir + "/vocab.txt",do_lower_case=False)
    
    with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
        train = tf.train.AdamOptimizer(learn_rate)
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        bertfortf.load_bert_pretrained_model(sess, bert_based_model_dir+"/bert_model.ckpt", restore_vars)
        step = 0
        for i in range(train_num):
            position = 0
            while(position<len(data)):
                #get batch data
                data,label,batch_data,batch_label,\
                position = datautil.next_batch(batch_size=batch_size, data_x=data, data_y=None, 
                                               position=position, shuffle=True)
                #get bert data
                batch_tokens,batch_input_ids,batch_input_mask,batch_segment_ids,batch_labels_index,\
                actual_lengths = datautil.ner_bert_data_convert(batch_data, tokenizer, None)
                
                feed = {input_ids:batch_input_ids,input_mask:batch_input_mask,segment_ids:batch_segment_ids,
                             actual_lengths_tf:actual_lengths,istrain:True,keep_prob:0.5}
                _,loss_ = sess.run([train,loss],feed_dict=feed)
                if step %10 ==0:
                    print("epcho=",i,",step=",step,",loss=",loss_)
                step += 1


