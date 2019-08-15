from datautil import nlpDataUtil
from nlp.classification.generateData import *
import tensorflow as tf
import tensorflow.contrib as tfc
import os
import datetime
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  #只显示error


with tf.device("/cpu:0"):
    word2vec_size = 768
    output_size = 3
    x = tf.placeholder("float",shape=[None,None,word2vec_size],name="x")
    y = tf.placeholder(tf.int32,shape=[None],name="y")
    lengths = tf.placeholder(tf.int32,shape=[None],name="lengths")
    gpu_num = 0
    lstm_units = 256


def model_compute(_x,_y,_lengths,regularizer=None):
    # print(_x.shape,_y.shape)
    with tf.variable_scope("",reuse=tf.AUTO_REUSE):
        fw_cell = tf.nn.rnn_cell.LSTMCell(lstm_units, forget_bias=0.8)
        bw_cell = tf.nn.rnn_cell.LSTMCell(lstm_units, forget_bias=0.8)
        out,_ = tf.nn.bidirectional_dynamic_rnn(fw_cell, bw_cell, _x, sequence_length=_lengths,dtype="float")
        # print(out)
        if isinstance(out, tuple):
            out = tf.concat(out,axis=-1)
        # print(out.shape)
        out = tf.keras.layers.BatchNormalization()(out)
        # print(out.shape)
        out = tf.reduce_mean(out,axis=1)
        # print(out.shape)
        out = tf.layers.dense(out, units=256, activation=tf.nn.relu, kernel_initializer="he_normal",
                              kernel_regularizer=regularizer)
        out = tf.layers.dense(out, units=32, activation=tf.nn.relu, kernel_initializer="he_normal",
                              kernel_regularizer=regularizer)
        out = tf.layers.dense(out, units=output_size, activation=tf.nn.softmax, kernel_initializer="he_normal",
                              kernel_regularizer=regularizer)
        loss = tf.reduce_mean(tf.losses.sparse_softmax_cross_entropy(_y,out))
        acc = tf.reduce_mean(tf.cast(tf.equal(_y,tf.argmax(out,axis=-1,output_type=tf.int32)),"float"))
        return out,loss,acc


def model():
    with tf.device("/cpu:0"):
        regularizer = tfc.layers.l2_regularizer(0.0001)
        if gpu_num > 0:
            _x = tf.split(x,gpu_num,axis=0)
            _y = tf.split(y,gpu_num,axis=0)
            print(_x[0].shape)
            print(_y[0].shape)
            _lengths = tf.split(lengths,gpu_num,axis=0)
            loss = 0
            acc = 0
            out = []
            for i in range(gpu_num):
                with tf.device("/gpu:%d"%i):
                    _out,_loss,_acc = model_compute(_x[i],_y[i],_lengths[i],regularizer)
                    loss += _loss
                    acc += _acc
                    out.append(_out)
            loss /= gpu_num
            acc /= gpu_num
            out = tf.concat(out,axis=0)
        else:
            out,loss,acc = model_compute(x,y,lengths,regularizer)
        return out,loss,acc


with tf.device("/cpu:0"):
    out,loss,acc = model()


def train(train_data,train_label,test_data,test_label,datautil: nlpDataUtil.NLPDataUtil,train_num,learning_rate,batch_size):
    with tf.device("/cpu:0"):
        optimizer = tf.train.AdamOptimizer(learning_rate)
        train_ops = optimizer.minimize(loss)
        learn_mode = tf.keras.backend.learning_phase()
        init = tf.global_variables_initializer()
        config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True, per_process_gpu_memory_fraction=0.8),
                                allow_soft_placement=True)
        with tf.Session(config=config) as sess:
            sess.run(init)
            for i in range(train_num):
                generator = generator_batch(batch_size, train_data, train_label, num_parallel_calls=gpu_num)
                for batch_x,batch_y,flag in generator:
                    pad_x,_,actual_lengths = datautil.padding(batch_x)
                    batch_x,_ = datautil.format(pad_x)
                    start = datetime.datetime.now()
                    sess.run(train_ops,feed_dict={x:batch_x,y:batch_y,lengths:actual_lengths,learn_mode:1})
                    tr_loss,tr_acc = sess.run([loss,acc],feed_dict={x:batch_x,y:batch_y,lengths:actual_lengths,learn_mode:0})
                    if flag == 1:
                        count = 0
                        v_loss,v_acc = 0,0
                        for batch_x,batch_y,flag in generator_batch(batch_size, test_data,test_label,
                                                                    num_parallel_calls=gpu_num,shuffle=False):
                            pad_x, pad_y, actual_lengths = datautil.padding(batch_x, batch_y)
                            batch_x, batch_y = datautil.format(pad_x, pad_y)
                            _loss,_acc = sess.run([loss,acc],feed_dict={x:batch_x,y:batch_y,lengths:actual_lengths,learn_mode:0})
                            v_loss += _loss
                            v_acc += _acc
                            count += 1
                        v_loss /= count
                        v_acc /= count
                        print("i=",i,",tr_loss=",tr_loss,",tr_acc=",tr_acc,",v_loss=",v_loss,",v_acc=",v_acc)
                        end = datetime.datetime.now()
                        print("batch fit time=",(end-start).total_seconds())






def main():
    datapath = "nlp/classification/data/classification.json"
    keyword_path = "nlp/classification/data/keyword.xlsx"
    train_text, train_label, test_text, test_label = read_classification_data(datapath,
                                                                              depreated_text="DirtyDeedsDoneDirtCheap",
                                                                              data_augmentation_label=2,
                                                                              test_percent=0.5,
                                                                              keyword_path=keyword_path)
    print(len(train_label), len(test_label))
    datautil = nlpDataUtil.NLPDataUtil(word2vec_path="nlp/classification/tfversion/model/char.model")
    train_data = []
    test_data = []
    for text in train_text:
        train_data.append(text.split(" "))
    for text in test_text:
        test_data.append(text.split((" ")))
    datautil.word2vec(train_data + test_data, size=word2vec_size, min_count=1, sg=1)
    train(train_data, train_label, test_data, test_label,datautil,
          train_num=100, learning_rate=0.0005, batch_size=64)


if __name__ == '__main__':
    main()
