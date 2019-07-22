# from datautil import nlpDataUtil
from nlp.bert.tfversion import bertTfApi
from bert import modeling
from bert import tokenization
import tensorflow as tf
from tensorflow.contrib import rnn as tcr
from nlp.classification.generateData import *
class SingleClassification:
    def __init__(self):
        super(SingleClassification, self).__init__()
        self.regularizer = tf.keras.regularizers.l2(0.0001)
        self.fc1 = tf.keras.layers.Dense(units=512, activation=tf.nn.relu, kernel_initializer="he_normal",
                                         kernel_regularizer=self.regularizer)
        self.fc2 = tf.keras.layers.Dense(units=128, activation=tf.nn.relu, kernel_initializer="he_normal",
                                         kernel_regularizer=self.regularizer)
        self.softmax = tf.keras.layers.Dense(units=4, activation=tf.nn.softmax, kernel_initializer="he_normal",
                                             kernel_regularizer=self.regularizer)
        self.bertfortf = None
        self.restore_vars = None

    def createmodel(self, *args, **kwargs):
        '''
        :param args:
            if len(args)==7   顺序固定如下
                 bert_config,    bert config
                 bert_is_train,  train or not train in bert
                 input_ids,      word to ids
                 input_mask,     have word in this position 1,not 0
                 segment_ids,    the first sentence 0, the second 1
                 bert_name,      the bert model load scope
                 keep_prob,      for dropout
            else:  顺序固定如下
                input,      shape[batch,timesteps,word2vec_size]=[None,None,word2vec_size]
                keep_prob,  for dropout
        :param kwargs:
            isbn,       true to bn    flase not bn
            istrain,    tf.bool for LN and BN
        :return:
        '''
        if "isbn" in kwargs:
            isbn = kwargs["isbn"]
        else:
            isbn = False
        if "istrain" in kwargs:
            istrain = kwargs["istrain"]
        else:
            istrain = False
        if len(args)==7:
            bert_config = args[0]
            bert_is_train = args[1]
            input_ids = args[2]
            input_mask = args[3]
            segment_ids = args[4]
            bert_name = args[5]
            # istrain = args[6]
            keep_prob = args[6]
            bertfortf = bertTfApi.BertForTensorFlow(bert_config, bert_is_train, input_ids, input_mask,
                                                    segment_ids, scope=bert_name)
            self.bertfortf = bertfortf
            model,restore_vars = bertfortf.create_bert_model()
            self.restore_vars = restore_vars
            output = model.get_pooled_output()
        else:
            input = args[0]
            # istrain = args[1]
            keep_prob = args[1]
            actual_lengths_tensor = args[3]
            fcell = tf.nn.rnn_cell.MultiRNNCell([self.__get_lstm_cell(256, forget_bias=0.8,
                                                                      output_keep_prob=keep_prob)
                                                 for _ in range(2)])
            bcell = tf.nn.rnn_cell.MultiRNNCell([self.__get_lstm_cell(256, forget_bias=0.8,
                                                                      output_keep_prob=keep_prob)
                                                 for _ in range(2)])
            output, _ = tf.nn.bidirectional_dynamic_rnn(fcell, bcell, input, sequence_length=actual_lengths_tensor,dtype="float")
            if isinstance(output,tuple):
                output = tf.concat(output, axis=-1)
            output = tf.reduce_mean(output, axis=1)
        if isbn:
            output = tf.keras.layers.BatchNormalization(trainable=istrain)(output)
        output = self.fc1(output)
        output = self.fc2(output)
        output = tf.nn.dropout(keep_prob=keep_prob)
        output = self.softmax(output)
        return output

    def __get_lstm_cell(self,num_units, use_peepholes=False, cell_clip=None, num_proj=None, proj_clip=None,
                          forget_bias=1.0, activation=tf.nn.tanh,
                          layer_norm=False, norm_gain=1.0, norm_shift=0.0, input_keep_prob=1.0, state_keep_prob=1.0,
                          output_keep_prob=1.0):
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
            #     cell = tcr.LayerNormLSTMCell(num_units,use_peepholes=use_peepholes,cell_clip=cell_clip,num_proj=num_proj,proj_clip=proj_clip,
            #                                  forget_bias=forget_bias,activation=activation,layer_norm=layer_norm,norm_gain=norm_gain,norm_shift=norm_shift)
        cell = tcr.LayerNormBasicLSTMCell(num_units, forget_bias=forget_bias, activation=activation,
                                          layer_norm=layer_norm,norm_gain=norm_gain, norm_shift=norm_shift)
        #     cell = tcr.LSTMCell(num_units,use_peepholes=use_peepholes, cell_clip=cell_clip,num_proj=num_proj, proj_clip=proj_clip,
        #                         forget_bias=forget_bias,activation=activation)
        cell = tcr.DropoutWrapper(cell, input_keep_prob=input_keep_prob, state_keep_prob=state_keep_prob,
                                  output_keep_prob=output_keep_prob)
        return cell

    def __call__(self, *args, **kwargs):
        self.createmodel(args,kwargs)

#bert model paramers set   placeholder
input_ids = tf.placeholder(shape=[None,None], dtype=tf.int32, name = "input_ids")
input_mask = tf.placeholder(shape=[None,None], dtype=tf.int32, name = "input_mask")
segment_ids = tf.placeholder(shape=[None,None], dtype=tf.int32, name = "segment_ids")
y = tf.placeholder(shape=[None],dtype=tf.int32,name="y")
keep_prob = tf.placeholder("float")
def train(data,label,bert_base_model_dir,train_num,learning_rate,batch_size):
    tokenizer = tokenization.FullTokenizer(vocab_file=bert_base_model_dir+"/vocab.txt",do_lower_case=False)
    singleclass = SingleClassification()
    output = singleclass(modeling.BertConfig.from_json_file(bert_base_model_dir+"/bert_config.json"), True,
                         input_ids, input_mask, segment_ids, "bert", keep_prob)
    loss = tf.losses.sparse_softmax_cross_entropy(y,output)
    accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(output,axis=-1,output_type=tf.int32),y),"float"))
    optimizer = tf.train.AdamOptimizer(learning_rate)
    with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
        train = optimizer.minimize(loss)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        singleclass.bertfortf.load_bert_pretrained_model(sess,bert_base_model_dir+"/bert_model.ckpt",
                                                         singleclass.restore_vars)
    step = 0
    for i in range(train_num):
        position = 0
        while(position<len(data)):
            data,label,batch_data,batch_label,position = next_batch(batch_size,data,label,position)
            batch_input_ids,batch_input_mask,batch_segment_ids,_=convert_batch_data(batch_data,tokenizer)
            sess.run(train,feed_dict = {
                input_ids:batch_input_ids,
                input_mask:batch_input_mask,
                segment_ids:batch_segment_ids,
                keep_prob:0.5,
                y:batch_label
            })
            if step%100==0:
                print(sess.run([loss,accuracy],feed_dict={
                    input_ids:batch_input_ids,
                    input_mask:batch_input_mask,
                    segment_ids:batch_segment_ids,
                    keep_prob:0.5,
                    y:batch_label
                }))
            step += 1

def main():
    datapath = "../data/sub.csv"
    data,label = read_data(datapath)
    bert_base_model_dir = "../../bert/base_model/cased_L-12_H-768_A-12"
    train_num = 100
    learning_rate = 0.0005
    batch_size = 128
    train(data,label,bert_base_model_dir,train_num,learning_rate,batch_size)

if __name__ == '__main__':
    main()