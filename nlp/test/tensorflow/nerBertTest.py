'''
Created on 2019年7月11日

@author: gaojiexcq
'''
import tensorflow as tf
from bert import modeling
from datautil import nlpDataUtil
from bert import tokenization
from nlp.test.read_data_for_conll2003 import *

train_path = "../../ner/data/conll2003/eng.train"
v_path = "../../ner/data/conll2003/eng.testa"
test_path = "../../ner/data/conll2003/eng.testb"

data,data_pos_tag,data_chunk_tag,label,position,label_index = read_data(train_path,encoding="utf-8",position=0,
                                                                        read_data_size=None,padding_str="<pad>")

datautil = nlpDataUtil.NLPDataUtil(use_for_bert=True, label_setlist=label_index)

output_size = len(datautil.label_setlist)


bert_based_model_dir = "../../bert/base_model/cased_L-12_H-768_A-12"

#bert model paramers set   placeholder
input_ids = tf.placeholder(shape=[None,None], dtype=tf.int32, name = "input_ids")
input_mask = tf.placeholder(shape=[None,None], dtype=tf.int32, name = "input_mask")
segment_ids = tf.placeholder(shape=[None,None], dtype=tf.int32, name = "segment_ids")
y = tf.placeholder(shape=[None,None],dtype=tf.int32,name="y")

#tf session config
configsession = tf.ConfigProto()
configsession.gpu_options.allow_growth = True

def model():

    #bert config load
    bert_config = modeling.BertConfig.from_json_file(bert_based_model_dir+"/bert_config.json")
    
    #bert model load 
    bert_model = modeling.BertModel(
            config = bert_config,
            is_training = True,
            input_ids = input_ids,
            input_mask= input_mask,
            token_type_ids= segment_ids,
            use_one_hot_embeddings=False
        )
    restore_vars = tf.global_variables()
    out = bert_model.get_sequence_output()
    out = tf.layers.dense(out, units=output_size,activation=tf.nn.softmax)
    return out,restore_vars

predict,restore_vars = model()
loss = tf.losses.sparse_softmax_cross_entropy(labels=y, logits=predict)
accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(predict, axis=-1, output_type=tf.int32),y),"float"))


def train(data,label,train_num,learn_rate,batch_size):
    tokenizer = tokenization.FullTokenizer(vocab_file=bert_based_model_dir + "/vocab.txt",do_lower_case=False)
    
    #get validation data start 
    v_data,_,_,v_label,_,_ = read_data(v_path,encoding="utf-8",position=0,
                                                                        read_data_size=None,padding_str="<pad>")
    #get batch data
    _,_,v_data,v_label,_ = datautil.next_batch(batch_size=batch_size, data_x=v_data, data_y=v_label, 
                                               position=0, shuffle=False)
    #get bert data
    v_tokens,v_input_ids,v_input_mask,v_segment_ids,v_labels_index,\
    v_lengths = datautil.ner_bert_data_convert(v_data, v_label, tokenizer)
    
    #get validation data end
    
    
    optimizer = tf.train.AdamOptimizer(learn_rate)
    train = optimizer.minimize(loss)
    #saver create
    saver = tf.train.Saver(var_list=restore_vars,max_to_keep=1)
    init = tf.global_variables_initializer()
    with tf.Session(config=configsession) as sess:
        sess.run(init)
        saver.restore(sess, bert_based_model_dir+"/bert_model.ckpt")
        print(1)
        step = 0
        for i in range(train_num):
            position = 0
            while(position<len(data)):
                #get batch data
                data,label,batch_data,batch_label,\
                position = datautil.next_batch(batch_size=batch_size, data_x=data, data_y=label, 
                                               position=position, shuffle=True)
                #get bert data
                batch_tokens,batch_input_ids,batch_input_mask,batch_segment_ids,batch_labels_index,\
                actual_lengths = datautil.ner_bert_data_convert(batch_data, batch_label, tokenizer)
                
                feed = {input_ids:batch_input_ids,input_mask:batch_input_mask,segment_ids:batch_segment_ids,
                             y:batch_labels_index}
                _,loss_tr,acc_tr = sess.run([train,loss,accuracy],feed_dict = feed)
                if step % 10 ==0:
                    feed = {input_ids:v_input_ids,input_mask:v_input_mask,segment_ids:v_segment_ids,
                             y:v_labels_index}
                    loss_v,acc_v = sess.run([loss,accuracy],feed_dict=feed)
                    print("i=",i,",step=",step,",loss_tr=",loss_tr,",loss_v=",loss_v,",acc_tr=",acc_tr,",acc_v=",acc_v)
                step += 1
                
                

if __name__=="__main__":
    train_num = 100
    learn_rate = 2e-5
    batch_size = 32
    train(data, label, train_num, learn_rate, batch_size)
    