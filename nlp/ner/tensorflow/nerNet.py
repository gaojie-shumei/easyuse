'''
Created on 2019年7月12日

@author: gaojiexcq
'''
import tensorflow as tf
from nlp.bert.tensorflow import bertTfApi
from datautil.nlpDataUtil import NLPDataUtil
from tensorflow.contrib import crf as tcc
from bert import modeling
from bert import tokenization
import numpy as np
import os
import os.path as ospath
class NERNet:
    '''
    the ner label mode should be BIO or IOB or BIOES
    '''
    def __init__(self,datautil:NLPDataUtil,y,actual_lengths_tensor,use_crf=True,tokenizer:tokenization.FullTokenizer=None,
                 bert_base_model_path=None,sessionConfig:tf.ConfigProto=None,checkpoint=None):
        '''
        :datautil: a NLPDataUtil class,has use_for_bert property,if not use bert,it should have the word2vec_model
        :y: the label of ner  it's the standard answer,usually is a placeholder with shape [batch,max_len]=[None,None]
        :actual_lengths_tensor: actual lengths for input data ,shape [batch] = [None]
        :use_crf: use or not use a crf to label the data
        :tokenizer: a bert tokenizer for split the sample, if the datautil.use_for_bert is True,then it should be privide
        :bert_base_model_path: if the datautil.use_for_bert is True,then it should be privide,use for load bert pretrained model
        :sessionConfig: the config of tf.Session
        :checkpoint: the model base dir
        '''
        self.datautil = datautil
        self.use_crf = use_crf
        self.y = y  
        self.predict = None  # the predict of label
        self.output_size = len(datautil.label_setlist)
        self.actual_lengths_tensor = actual_lengths_tensor
        if datautil.use_for_bert:
            if tokenizer is None or bert_base_model_path is None:
                raise RuntimeError("the datautil show that bert is used,so the tokenizer and the bert base model path need to be privided")
            else:
                self.tokenizer = tokenizer
                self.bert_base_model_path = bert_base_model_path
        else:
            self.tokenizer = None
            self.bert_base_model_path = None
        self.loss = None
        self.accuracy = None
        self.restore_vars = None
        self.bertfortf = None
        self.input_ids = None
        self.input_mask = None
        self.segment_ids = None
        self.input = None
        self.entitylist = self.get_entity()
        self.model_fn_placeholders = None
        self.sessionConfig = sessionConfig
        self.checkpoint = checkpoint
        
    def create_NERModel(self,input,model_fn=None,model_fn_params:dict=None,model_fn_placeholders:dict=None,regularizer_fn=None):
        '''
        :input: a tensor,usually is a placeholder, like the mnist example's input x
        :model_fn: your own model function write by tensorflow, output a tensor has the same dim of input
                   ,and it's output can't activate with softmax
        :model_fn_params: your own  model function's paramters include all placeholder used,it should privide with dict
        :model_fn_placeholders: the placeholder use for your own model function
        :regularizer_fn: a regularizer function,usually use tf.contrib.layers.l2_regularizer(scale)
        '''
        
        if self.use_crf:
            if model_fn is not None:
                if model_fn_params is not None:
                    out = model_fn(input,**model_fn_params)
                else:
                    out = model_fn(input)
                out = tf.layers.dense(out, units=self.output_size,activation=tf.nn.relu,
                                      kernel_regularizer=regularizer_fn)
            else:
                out = self.base_model(input,activation=tf.nn.relu, regularizer_fn=regularizer_fn)
            log_likehood,transition_params = tcc.crf_log_likelihood(out, tag_indices=self.y, 
                                                                    sequence_lengths = self.actual_lengths_tensor)
            loss = tf.reduce_mean(-log_likehood)
            out, best_score = tcc.crf_decode(out, transition_params, self.actual_lengths_tensor)
            accuracy = tf.reduce_mean(tf.cast(tf.equal(self.y,out),"float"))
        else:
            if model_fn is not None:
                if model_fn_params is not None:
                    out = model_fn(input,**model_fn_params)
                else:
                    out = model_fn(input)
                out = tf.layers.dense(out, units=self.output_size,activation=tf.nn.softmax,
                                      kernel_regularizer=regularizer_fn)
            else:
                out = self.base_model(input,activation=tf.nn.softmax, regularizer_fn=regularizer_fn)
            loss = tf.losses.sparse_softmax_cross_entropy(labels=self.y, logits=out)
            out = tf.argmax(out, axis=-1, output_type=tf.int32)
            accuracy = tf.reduce_mean(tf.cast(tf.equal(out,self.y),"float"))
        self.loss = loss
        self.accuracy = accuracy
        self.predict = out
        self.input = input
        self.model_fn_placeholders = model_fn_placeholders
        return self
    
    def create_NERModel_based_bert(self,bert_config:modeling.BertConfig,bert_is_train:bool,
                                   input_ids,input_mask,segment_ids,
                                   use_one_hot_embeddings:bool=False,model_fn=None,model_fn_params:dict=None,
                                   model_fn_placeholders:dict=None,regularizer_fn=None):
        '''
        :bert_config: bert config file, not clear yet
        :bert_is_train: train or not train
        :input_ids: tf.placeholder[batch,max_len]=[None,None] the text(one sample) split with bert.tokenization and convert it to ids
        :input_mask: tf.placeholder[batch,max_len]=[None,None] if the position has word,it is 1,else 0
        :segment_ids: tf.placeholder[batch,max_len]=[None,None] if it is the sample's first sentence then 0,the second then 1,max is second
        :use_one_hot_embedding: use or not use
        :model_fn: the model function will be used after get the word2vec with bert,and it's output can't activate with softmax
        :model_fn_params: your own  model function's paramters include all placeholder used,it should privide with dict
        :model_fn_placeholders: the placeholder use for your own model function
        :regularizer_fn:   a regularizer function,usually use tf.contrib.layers.l2_regularizer(scale)
        '''
        bertfortf = bertTfApi.BertForTensorFlow(bert_config=bert_config, 
                                            bert_is_train=bert_is_train, 
                                            input_ids=input_ids, 
                                            input_mask=input_mask, 
                                            segment_ids=segment_ids, 
                                            use_one_hot_embeddings=use_one_hot_embeddings)
        model,restore_vars = bertfortf.create_bert_model()
        out = model.get_sequence_output()
        self.create_NERModel(out, model_fn, model_fn_params,model_fn_placeholders, regularizer_fn)
        self.restore_vars = restore_vars
        self.bertfortf = bertfortf
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        return self
    
    def base_model(self,input,activation=None,regularizer_fn=None):
        out = tf.layers.dense(input, units=self.output_size,activation=activation,
                              kernel_regularizer=regularizer_fn)
        return out
    
    def train(self,data:list,label:list,train_num,learn_rate,batch_size,v_data:list=None,v_label:list=None,
              model_fn_placeholder_feed_tr:dict=None,model_fn_placeholder_feed_pre:dict=None,
              optimizer=None,optimizer_params=None,step_for_show=10):
        '''
        :data: the train data
        :label: the label of data
        :train_num: the train num
        :learn_rate: the learn rate
        :batch_size: the batch size
        :v_data: the validation data
        :v_label: the validation label
        :model_fn_placeholder_feed_tr: the feed to the model_fn_placeholders with keys for train
        :model_fn_placeholder_feed_pre: the feed to the model_fn_placeholders with keys for predict
        :optimizer: your custom optimizer function
        :optimizer_params: your custom optimizer function paramters except learn rate,a dict
        :step_for_show: how many step to show the train answer
        '''
        if optimizer is not None:
            train = optimizer(learning_rate = learn_rate,**optimizer_params).minimize(self.loss)
        else:
            train = tf.train.AdamOptimizer(learning_rate=learn_rate).minimize(self.loss)
        init = tf.global_variables_initializer()
        
        with tf.Session(config=self.sessionConfig) as sess:
            saver = tf.train.Saver()
            sess.run(init)
            if self.restore_vars is not None:
                if self.bert_base_model_path is None:
                    raise RuntimeError("the bert pretrained model not privide")
                else:
                    if self.bertfortf is not None:
                        self.bertfortf.load_bert_pretrained_model(sess, self.bert_base_model_path, self.restore_vars)
                    else:
                        raise RuntimeError("the bertTfApi is not privide")
            datautil = self.datautil
            if datautil.use_for_bert == True:
                tokenizer = self.tokenizer
            step = 0
            max_F = 0
            print(self.entitylist)
            for i in range(train_num):
                position = 0
                while(position<len(data)):
                    #get batch data
                    data,label,batch_data,batch_label,\
                    position = datautil.next_batch(batch_size=batch_size, data_x=data, data_y=label, 
                                                   position=position, shuffle=True)
                    if datautil.use_for_bert == True:
                        #get bert data
                        batch_tokens,batch_input_ids,batch_input_mask,batch_segment_ids,batch_labels_index,\
                        actual_lengths = datautil.ner_bert_data_convert(batch_data, batch_label, tokenizer)
                        feed = {
                            self.input_ids:batch_input_ids,
                            self.input_mask:batch_input_mask,
                            self.segment_ids:batch_segment_ids,
                            self.y:batch_labels_index,
                            self.actual_lengths_tensor:actual_lengths
                        }        
                        normal_tags = batch_labels_index
                    else:
                        pad_data, pad_y_data, actual_lengths = datautil.padding(batch_data, batch_label)
                        batch_x,batch_y = datautil.format(pad_data, pad_y_data)
                        feed = {
                            self.input:batch_x,
                            self.y:batch_y,
                            self.actual_lengths_tensor:actual_lengths
                        }
                        normal_tags = batch_y
                    if self.model_fn_placeholders is not None:
#                         assert(len(self.model_fn_placeholders.keys())==len(model_fn_placeholder_feed.keys()))
                        for key in self.model_fn_placeholders:
                            if key not in model_fn_placeholder_feed_tr:
                                raise RuntimeError("the model_fn_placeholder_feed_tr can't find the correct key for placeholder")
                            else:
                                feed.update({self.model_fn_placeholders[key]:model_fn_placeholder_feed_tr[key]})
                    _,loss_tr,acc_tr = sess.run([train,self.loss,self.accuracy],feed_dict=feed)
                    if v_data is None:
                        predict_tags = sess.run(self.predict,feed_dict=feed)
                        _,_,_,P_tr,R_tr,F_tr,acc_tr = self.evaluation(predict_tags, normal_tags, actual_lengths)
                        avg_F = np.mean(F_tr)
                    else:
                        v_position = 0
                        P,R,F,acc = None,None,None,None
                        while(v_position<len(v_data)):
                            #get batch data
                            v_data,v_label,batch_data,batch_label,\
                            v_position = datautil.next_batch(batch_size=batch_size, data_x=v_data, data_y=v_label, 
                                                           position=v_position, shuffle=False)
                            if datautil.use_for_bert == True:
                                #get bert data
                                batch_tokens,batch_input_ids,batch_input_mask,batch_segment_ids,batch_labels_index,\
                                actual_lengths = datautil.ner_bert_data_convert(batch_data, batch_label, tokenizer)
                                feed = {
                                    self.input_ids:batch_input_ids,
                                    self.input_mask:batch_input_mask,
                                    self.segment_ids:batch_segment_ids,
                                    self.y:batch_labels_index,
                                    self.actual_lengths_tensor:actual_lengths
                                }        
                                normal_tags = batch_labels_index
                            else:
                                pad_data, pad_y_data, actual_lengths = datautil.padding(batch_data, batch_label)
                                batch_x,batch_y = datautil.format(pad_data, pad_y_data)
                                feed = {
                                    self.input:batch_x,
                                    self.y:batch_y,
                                    self.actual_lengths_tensor:actual_lengths
                                }
                                normal_tags = batch_y
                            if self.model_fn_placeholders is not None:
        #                         assert(len(self.model_fn_placeholders.keys())==len(model_fn_placeholder_feed.keys()))
                                for key in self.model_fn_placeholders:
                                    if key not in model_fn_placeholder_feed_pre:
                                        raise RuntimeError("the model_fn_placeholder_feed_pre can't find the correct key for placeholder")
                                    else:
                                        feed.update({self.model_fn_placeholders[key]:model_fn_placeholder_feed_pre[key]})
                            predict_tags = sess.run(self.predict,feed_dict=feed)
                            _,_,_,P_v,R_v,F_v,acc_v = self.evaluation(predict_tags, normal_tags, actual_lengths)
                            if P is None:
                                P,R,F,acc = P_v,R_v,F_v,acc_v
                            else:
                                P,R,F,acc = np.c_[P,P_v],np.c_[R,R_v],np.c_[F,F_v],np.c_[acc,acc_v]
                        P,R,F,acc = np.mean(P, axis=-1),np.mean(R,axis=-1),np.mean(F,axis=-1),np.mean(acc)
                        avg_F = np.mean(F)
                    if avg_F>max_F:
                        max_F = avg_F
                        if self.checkpoint is not None:
                            if ospath.isdir(self.checkpoint)==False:
                                os.makedirs(self.checkpoint)
                            saver.save(sess,ospath.join(self.checkpoint,"model.ckpt"))
                        if step%step_for_show!=0:
                            if v_data is None:
                                print("epcho=",i,",step=",step,",loss_train=",loss_tr,",P_train=",P_tr,",R_train=",R_tr,
                                      ",F_train=",F_tr,",acc_train=",acc_tr,",avg_F=",avg_F,",max_F=",max_F)
                            else:
                                print("epcho=",i,",step=",step,",loss_train=",loss_tr,",P_validation=",P,",R_validation=",R,
                                      ",F_validation=",F,",acc_validation=",acc,",avg_F=",avg_F,",max_F=",max_F)
                    if step%step_for_show==0:
                        if v_data is None:
                            print("epcho=",i,",step=",step,",loss_train=",loss_tr,",P_train=",P_tr,",R_train=",R_tr,
                                  ",F_train=",F_tr,",acc_train=",acc_tr,",avg_F=",avg_F,",max_F=",max_F)
                        else:
                            print("epcho=",i,",step=",step,",loss_train=",loss_tr,",P_validation=",P,",R_validation=",R,
                                  ",F_validation=",F,",acc_validation=",acc,",avg_F=",avg_F,",max_F=",max_F)
                    step += 1    
        return        
        
        

    def test(self,test_data:list,test_label:list,batch_size,model_fn_placeholder_feed_pre:dict=None):
        '''
        :test_data: the test data, a two dim list
        :test_data: the test label, a two dim list
        :batch_size: batch size
        :model_fn_placeholder_feed_pre: the feed to the model_fn_placeholders with keys for predict
        '''
        P,R,F,acc = None,None,None,None
        datautil = self.datautil
        if self.checkpoint is None or self.checkpoint=="":
            raise ValueError("the model path can't be None")
        saver = tf.train.Saver()
        with tf.Session(config=self.sessionConfig) as sess:
            try:
                saver.restore(sess, tf.train.latest_checkpoint(self.checkpoint))
            except:
                raise RuntimeError("the model restore failed")
            position = 0
            while(position<len(test_data)):
                #get batch data
                test_data,test_label,batch_data,batch_label,position = datautil.next_batch(batch_size=batch_size, data_x=test_data, 
                                               data_y=test_label,position=position, shuffle=False)
                if datautil.use_for_bert:
                    batch_tokens,batch_input_ids,batch_input_mask,batch_segment_ids,batch_labels_index,\
                    actual_lengths = datautil.ner_bert_data_convert(batch_data,batch_label, tokenizer=self.tokenizer)
                    feed = {
                        self.input_ids:batch_input_ids,
                        self.input_mask:batch_input_mask,
                        self.segment_ids:batch_segment_ids,
                        self.y:batch_labels_index,
                        self.actual_lengths_tensor:actual_lengths
                    }
                    normal_tags = batch_labels_index
                else:
                    pad_data, pad_y_data, actual_lengths = datautil.padding(batch_data,batch_label)
                    batch_x,batch_y = datautil.format(pad_data,pad_y_data)
                    feed = {
                        self.input:batch_x,
                        self.y:batch_y,
                        self.actual_lengths_tensor:actual_lengths
                    }
                    normal_tags = batch_y
                if self.model_fn_placeholders is not None:
        #                         assert(len(self.model_fn_placeholders.keys())==len(model_fn_placeholder_feed.keys()))
                    for key in self.model_fn_placeholders:
                        if key not in model_fn_placeholder_feed_pre:
                            raise RuntimeError("the model_fn_placeholder_feed_pre can't find the correct key for placeholder")
                        else:
                            feed.update({self.model_fn_placeholders[key]:model_fn_placeholder_feed_pre[key]})
                predict_tags = sess.run(self.predict,feed_dict=feed)
                _,_,_,P_t,R_t,F_t,acc_t = self.evaluation(predict_tags, normal_tags, actual_lengths)
                if P is None:
                    P,R,F,acc = P_t,R_t,F_t,acc_t
                else:
                    P,R,F,acc = np.c_[P,P_t],np.c_[R,R_t],np.c_[F,F_t],np.c_[acc,acc_t]
            P,R,F,acc = np.mean(P, axis=-1),np.mean(R,axis=-1),np.mean(F,axis=-1),np.mean(acc)
            print(self.entitylist)
            print("test_P=",P,",test_R=",R,",test_F=",F,",test_acc=",acc)
            return P,R,F,acc
        pass

    def predict(self,test_data:list,batch_size,model_fn_placeholder_feed_pre:dict=None):
        '''
        :test_data: the test data, a two dim list
        :batch_size: batch size
        :model_fn_placeholder_feed_pre: the feed to the model_fn_placeholders with keys for predict
        '''
        predict_labels = []
        datautil = self.datautil
        if self.checkpoint is None or self.checkpoint=="":
            raise ValueError("the model path can't be None")
        saver = tf.train.Saver()
        with tf.Session(config=self.sessionConfig) as sess:
            try:
                saver.restore(sess, tf.train.latest_checkpoint(self.checkpoint))
            except:
                raise RuntimeError("the model restore failed")
            position = 0
            while(position<len(test_data)):
                #get batch data
                test_data,_,batch_data,_,position = datautil.next_batch(batch_size=batch_size, data_x=test_data, 
                                               position=position, shuffle=False)
                if datautil.use_for_bert:
                    batch_tokens,batch_input_ids,batch_input_mask,batch_segment_ids,batch_labels_index,\
                    actual_lengths = datautil.ner_bert_data_convert(batch_data, tokenizer=self.tokenizer)
                    feed = {
                        self.input_ids:batch_input_ids,
                        self.input_mask:batch_input_mask,
                        self.segment_ids:batch_segment_ids,
                        self.actual_lengths_tensor:actual_lengths
                    }
                else:
                    pad_data, _, actual_lengths = datautil.padding(batch_data)
                    batch_x,_ = datautil.format(pad_data)
                    feed = {
                        self.input:batch_x,
                        self.actual_lengths_tensor:actual_lengths
                    }
                if self.model_fn_placeholders is not None:
        #                         assert(len(self.model_fn_placeholders.keys())==len(model_fn_placeholder_feed.keys()))
                    for key in self.model_fn_placeholders:
                        if key not in model_fn_placeholder_feed_pre:
                            raise RuntimeError("the model_fn_placeholder_feed_pre can't find the correct key for placeholder")
                        else:
                            feed.update({self.model_fn_placeholders[key]:model_fn_placeholder_feed_pre[key]})
                predict_tags = sess.run(self.predict,feed_dict=feed)
                predict_label = datautil.index2label(predict_tags, actual_lengths)
                predict_labels.append(predict_label)
        return test_data,predict_labels
#         pass

    def evaluation(self,predict_tags,normal_tags,actual_lengths):
        '''
        the evaluation function  this maybe has some fault!
        :predict_tags: the label result, if crf,it's crf decode_tag to numpy,if softmax,it's np.argmax(out) to numpy
        :normal_tags: is the data which same to the feed data to self.y
        :actual_lengths: the input's actual length
        
        :return TP is each entity the correct num(label this entity,actually also yes)
                FP is each entity the incorrect num (label this entity,but actually not)
                normal_P is each entity actual num
                P is each entity TP/(TP+FP)
                R is each entity TP/normal_P
                F is each entity (2*P*R)/(P+R)
                acc is the all acc sum(TP)/sum(normal_P)
        '''
        entitylist = self.entitylist
        labellist = self.datautil.label_setlist
        x_sequence_length = actual_lengths
# def evaluation(labellist,entitylist,predict_tags,normal_tags,x_sequence_length):
        TP = np.zeros(shape=[len(entitylist)],dtype="float")#预测准确的      P -》 P
        FP = np.zeros(shape=[len(entitylist)],dtype="float")#将错误的预测成准确的  N -》 P
    #     FN = np.zeros(shape=[len(entitylist)],dtype="float")#本来正确但预测错误的 P -》 N
    #     TN = np.zeros(shape=[len(entitylist)],dtype="float")#本来错误同样预测成错误的 N -》 N
        #判定是否保存模型的准确率计算相关参数
        correct = 0
        normal_P = np.zeros(shape=[len(entitylist)],dtype="float")#实际准确的
        for i in range(predict_tags.shape[0]):
            hasentity = np.zeros(shape=[len(entitylist)],dtype="float")#判断该条数据是否有对应实体，有为1，没有为0
            predict = predict_tags[i,:]
            normal = normal_tags[i,:]
            #判断实体是否存在
            for j in range(x_sequence_length[i]):
                for k in range(len(entitylist)):
                    if entitylist[k] in labellist[normal[j]]:
                        hasentity[k] = 1
            predict_correct = np.array(hasentity.tolist())
            predict_fault = np.zeros(shape=[len(entitylist)],dtype="float")
            
            for j in range(x_sequence_length[i]):
                for k in range(len(entitylist)):
                    if entitylist[k] in labellist[predict[j]] and normal[j]!=predict[j]:
                        predict_fault[k] = 1
                    if entitylist[k] in labellist[normal[j]] and normal[j]!=predict[j]:
                        predict_correct[k] = 0
                if normal[j]==predict[j]:
                    correct += 1
            TP = TP + predict_correct
            FP = FP + predict_fault
            normal_P = normal_P + hasentity
        #P = TP/(TP+FP)   R=TP/normal_P   F = (2*P*R)/(P+R)   acc = (TP+TN)/(TP+FP+TN+FN) = corecct/all
        #针对本数据   acc = sum(TP)/sum(normal_P)
        predict_P = TP + FP
        for i in range(predict_P.shape[0]):
            if predict_P[i] ==0:
                predict_P[i] = 1
            if normal_P[i]==0:
                normal_P[i]=1
        P = TP/predict_P
        R = TP/normal_P
        P_R = P + R
        for i in range(P_R.shape[0]):
            if P_R[i]==0:
                P_R[i]=1
        F = (2*P*R)/P_R
    #     acc = (np.sum(TP)+other_correct_entity)/(np.sum(normal_P)+other_entity)
        acc = correct/np.sum(x_sequence_length)
        return TP,FP,normal_P,P,R,F,acc
    
    
    
    
    def get_entity(self):
        label_setlist = self.datautil.label_setlist
        pad_word = self.datautil.pad_word
        entityset = set()
        for label in label_setlist:
            if len(label)==1:
                if label=="S":
                    entityset.add(label)
            else:
                if label not in [pad_word,"[CLS]","[SEP]","[X]"]:
                    entityset.add(label[1:])
        entitylist = list(entityset)
        entitylist.sort()
        print(entitylist)
        return entitylist
    
    
    
    
    
    