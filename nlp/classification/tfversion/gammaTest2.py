from datautil import nlpDataUtil
from nlp.classification.generateData import *
import tensorflow as tf
import tensorflow.contrib as tfc
import os
import datetime
from module.tfversion import baseNet,modelModule
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  #只显示error


class GammaNet(baseNet.BaseNet):
    def __init__(self,lstm_units,output_size,regularizer):
        self.lstm_units = lstm_units
        self.output_size = output_size
        self.regularizer = regularizer

    def net(self, inputs):
        with tf.variable_scope("", reuse=tf.AUTO_REUSE):
            _x = inputs["input"]
            _lengths = inputs["lengths"]
            _y = inputs["y"]
            regularizer = self.regularizer
            lstm_units = self.lstm_units
            output_size = self.output_size
            fw_cell = tf.nn.rnn_cell.LSTMCell(lstm_units, forget_bias=0.8)
            bw_cell = tf.nn.rnn_cell.LSTMCell(lstm_units, forget_bias=0.8)
            out, _ = tf.nn.bidirectional_dynamic_rnn(fw_cell, bw_cell, _x, sequence_length=_lengths, dtype="float")
            # print(out)
            if isinstance(out, tuple):
                out = tf.concat(out, axis=-1)
            # print(out.shape)
            out = tf.keras.layers.BatchNormalization()(out)
            # print(out.shape)
            out = tf.reduce_mean(out, axis=1)
            # print(out.shape)
            out = tf.layers.dense(out, units=256, activation=tf.nn.relu, kernel_initializer="he_normal",
                                  kernel_regularizer=regularizer)
            out = tf.layers.dense(out, units=32, activation=tf.nn.relu, kernel_initializer="he_normal",
                                  kernel_regularizer=regularizer)
            out = tf.layers.dense(out, units=output_size, activation=tf.nn.softmax, kernel_initializer="he_normal",
                                  kernel_regularizer=regularizer)
            loss = tf.reduce_mean(tf.losses.sparse_softmax_cross_entropy(_y, out))
            acc = tf.reduce_mean(tf.cast(tf.equal(_y, tf.argmax(out, axis=-1, output_type=tf.int32)), "float"))
            outputs = {"output":out,"loss":loss,"acc":acc}
            return outputs


with tf.device("/cpu:0"):
    word2vec_size = 768
    output_size = 3
    gpu_num = 0
    lstm_units = 256
    learning_rate = 0.0005
    model_save_path = "model"
    model_name = "classification"


def creat_model():
    with tf.device("/cpu:0"):
        x = tf.placeholder("float", shape=[None, None, word2vec_size], name="x")
        y = tf.placeholder(tf.int32, shape=[None], name="y")
        lengths = tf.placeholder(tf.int32, shape=[None], name="lengths")

        regularizer = tfc.layers.l2_regularizer(0.0001)
        net = GammaNet(lstm_units,output_size,regularizer)
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
                    outputs = net({"input":_x[i],"lengths":_lengths[i],"y":_y[i]})
                    loss += outputs["loss"]
                    acc += outputs["acc"]
                    out.append(outputs["output"])
            loss /= gpu_num
            acc /= gpu_num
            out = tf.concat(out,axis=0)
        else:
            outputs = net({"input": x, "lengths": lengths, "y": y})
            out = outputs["output"]
            loss = outputs["loss"]
            acc = outputs["acc"]
        optimizer = tf.train.AdamOptimizer(learning_rate)
        train_ops = optimizer.minimize(loss)
        learn_mode = tf.keras.backend.learning_phase()
        model = modelModule.ModelModule([x,lengths],out,y,loss,train_ops,learn_mode,
                                        model_save_path=model_save_path, metrics=acc, num_parallel_calls=gpu_num)
        return model


with tf.device("/cpu:0"):
    model = creat_model()


def train(train_data,train_label,test_data,test_label,datautil: nlpDataUtil.NLPDataUtil,train_num,batch_size):
    with tf.device("/cpu:0"):
        test_x,_,test_lengths = datautil.padding(test_data)
        test_x,_ = datautil.format(test_x)
        test_y = test_label
        v_inputs_feed = [test_x, test_lengths]
        v_outputs_feed = test_y
        v_net_configs_feed = 0
        init = tf.global_variables_initializer()
        config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True, per_process_gpu_memory_fraction=0.8),
                                allow_soft_placement=True)
        with tf.Session(config=config) as sess:
            sess.run(init)
            pad_x, _, actual_lengths = datautil.padding(train_data)
            train_x, _ = datautil.format(pad_x)
            tr_inputs_feed = [train_x, actual_lengths]
            tr_outputs_feed = train_label
            tr_net_configs_feed = 1
            model.fit(sess, train_num, tr_inputs_feed, tr_outputs_feed, tr_net_configs_feed, v_inputs_feed,
                      v_outputs_feed, v_net_configs_feed, batch_size, False, True, start_save_model_epoch=10,
                      model_name=model_name)
            # for i in range(train_num):
            #     generator = generator_batch(batch_size, train_data, train_label,num_parallel_calls=gpu_num)
            #     for batch_x,batch_y,flag in generator:
            #         pad_x,_,actual_lengths = datautil.padding(batch_x)
            #         batch_x,_ = datautil.format(pad_x)
            #         start = datetime.datetime.now()
            #         do_validation = False
            #         if flag == 1:
            #             do_validation = True
            #         tr_inputs_feed = [batch_x,actual_lengths]
            #         tr_outputs_feed = batch_y
            #         tr_net_configs_feed = 1
            #         result = model.batch_fit(sess, tr_inputs_feed,tr_outputs_feed,tr_net_configs_feed,v_inputs_feed,
            #                                  v_outputs_feed,v_net_configs_feed,batch_size,do_validation=do_validation)
            #         if flag == 1:
            #             print("i=",i,",result=",result)
            #             end = datetime.datetime.now()
            #             print("batch fit time=",(end-start).total_seconds())






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
          train_num=100, batch_size=64)


if __name__ == '__main__':
    main()
