'''
Created on 2019年7月12日

@author: gaojiexcq
'''
import tensorflow as tf
from bert import modeling

class BertForTensorFlow:
    '''
    the word embedding can get in bert model
    '''
    def __init__(self,bert_config:modeling.BertConfig,bert_is_train:bool,input_ids,
                 input_mask=None,segment_ids=None,use_one_hot_embeddings:bool=False):
        '''
        :bert_config: bert config file, not clear yet
        :bert_is_train: train or not train
        :input_ids: the text(one sample) split with bert.tokenization and convert it to ids
        :input_mask: if the position has word,it is 1,else 0
        :segment_ids: if it is the sample's first sentence then 0,the second then 1,max is second
        :use_one_hot_embedding: use or not use
        '''
        self.bert_config = bert_config
        self.bert_is_train = bert_is_train
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.use_one_hot_embeddings = use_one_hot_embeddings
        self.bert_model = None
    
    def create_bert_model(self):
        model = modeling.BertModel(
            config=self.bert_config,
            is_training=self.bert_is_train,
            input_ids=self.input_ids,
            input_mask=self.input_mask,
            token_type_ids=self.segment_ids,
            use_one_hot_embeddings=self.use_one_hot_embeddings)
        self.bert_model = model
        restore_vars = tf.global_variables()
        return model,restore_vars
    
    def load_bert_pretrained_model(self,sess:tf.Session,model_path:str,restore_vars:list):
        '''
        :sess: a tf.Session for loading model
        :model_path: the bert pretrained model checkpoint usually name bert_model.ckpt
        :restore_vars: a list of tf.Variable the bert pretrained model's parampers
        '''
        saver = tf.train.Saver(var_list=restore_vars)
        saver.restore(sess, model_path)
        return
    
    