'''
Created on 2019年7月12日

@author: gaojiexcq
'''
import tensorflow as tf
from bert import modeling
import os

pathname = ".。/../bert/base_model/cased_L-12_H-768_A-12/bert_model.ckpt" # 模型地址
bert_config = modeling.BertConfig.from_json_file(".。/../bert/base_model/cased_L-12_H-768_A-12/bert_config.json")# 配置文件地址。
configsession = tf.ConfigProto()
configsession.gpu_options.allow_growth = True
sess = tf.Session(config=configsession)
input_ids = tf.placeholder(shape=[64, 128], dtype=tf.int32, name="input_ids")
input_mask = tf.placeholder(shape=[64, 128], dtype=tf.int32, name="input_mask")
segment_ids = tf.placeholder(shape=[64, 128], dtype=tf.int32, name="segment_ids")

model = modeling.BertModel(
        config=bert_config,
        is_training=True,
        input_ids=input_ids,
        input_mask=input_mask,
        token_type_ids=segment_ids,
        use_one_hot_embeddings=False)

out = model.get_sequence_output()
tvar = tf.global_variables()

out = tf.layers.dense(out, units=10,activation=None,name="softmax")
with sess.as_default():
    
    saver = tf.train.Saver(tvar)
    sess.run(tf.global_variables_initializer())# 这里尤其注意，先初始化，在加载参数，否者会把bert的参数重新初始化。这里和demo1是有区别的
    saver.restore(sess, pathname)
    print(1)