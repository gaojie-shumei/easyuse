from module.tfversion import baseNet,modelModule
import tensorflow as tf
from bert import modeling,tokenization
from nlp.classification.generateData import *
import warnings
warnings.filterwarnings(action="ignore")


class GammaClassNet(baseNet.BaseNet):
    def __init__(self, lstm_units, l2_scale, drop_rate, output_size, use_bert=True):
        super(GammaClassNet, self).__init__()
        self.l2 = None
        if l2_scale!=0:
            self.l2 = tf.keras.regularizers.l2(l2_scale)
        self.lstm_units = lstm_units
        self.l2_scale = l2_scale
        self.drop_rate = drop_rate
        self.output_size = output_size
        self.use_bert = use_bert
        self.lstm = tf.keras.layers.LSTM(lstm_units, kernel_initializer="he_normal", kernel_regularizer=self.l2,
                                         dropout=drop_rate, recurrent_dropout=drop_rate, return_sequences=True)
        self.masking = tf.keras.layers.Masking(0)
        self.bilstm = tf.keras.layers.Bidirectional(self.lstm, merge_mode="concat")
        self.fc = tf.keras.layers.Dense(lstm_units/2, activation=tf.keras.activations.relu,
                                             kernel_initializer="he_normal", kernel_regularizer=self.l2)
        self.softmax = tf.keras.layers.Dense(output_size, activation=tf.keras.activations.softmax,
                                             kernel_initializer="he_normal", kernel_regularizer=self.l2)

    def net(self, inputs):
        if self.use_bert:
            bert_config = inputs["bert_config"]
            bert_is_train = inputs["bert_is_train"]
            input_ids = inputs["input_ids"]
            input_mask = inputs["input_mask"]
            segment_ids = inputs["segment_ids"]
            bert_scope = inputs["bert_scope"]
            model = modeling.BertModel(config=bert_config, is_training=bert_is_train, input_ids=input_ids,
                                       input_mask=input_mask, token_type_ids=segment_ids, scope=bert_scope)
            output = model.get_sequence_output()
            bert_vars = tf.global_variables(bert_scope)
        else:
            output = inputs["input"]
            bert_vars = None
        output = self.masking(output)
        output = self.bilstm(output)
        output = tf.reduce_mean(output, axis=1)
        output = self.fc(output)
        output = self.softmax(output)
        print(output.get_shape().as_list())
        outputs = {
            "output": output,
            "bert_vars": bert_vars
        }
        return outputs

lstm_units = 256
l2_scale = 0
drop_rate = 0
output_size = 3
use_bert = True
word2vec_size = 128
bert_model_base_dir = "../../bertapi/base_model/cased_L-12_H-768_A-12"


def class_model():
    if use_bert:
        input_ids = tf.placeholder(tf.int32, shape=[None, None], name="input_ids")
        input_mask = tf.placeholder(tf.int32, shape=[None, None], name="input_mask")
        segment_ids = tf.placeholder(tf.int32, shape=[None, None], name="segment_ids")
    else:
        input = tf.placeholder("float", shape=[None, None, word2vec_size], name="input")
    y = tf.placeholder("float", shape=[None, output_size], name="y")
    is_train = tf.keras.backend.learning_phase()  #1 for train 0 for test
    lr = tf.placeholder("float")
    bert_config = modeling.BertConfig.from_json_file(bert_model_base_dir+"/bert_config.json")
    net = GammaClassNet(lstm_units, l2_scale, drop_rate, output_size, use_bert)
    if use_bert:
        net_inputs = {
            "bert_config": bert_config,
            "bert_is_train": True,
            "input_ids": input_ids,
            "input_mask": input_mask,
            "segment_ids": segment_ids,
            "bert_scope": "bert"
        }
    else:
        net_inputs = {
            "input": input
        }
    outputs = net(net_inputs)
    output = outputs["output"]
    bert_vars = outputs["bert_vars"]
    loss = tf.pow(tf.abs((y - output)), 2)*tf.log(output)
    accuracy = tf.reduce_mean(tf.keras.metrics.categorical_accuracy(y, output))

    if use_bert:
        model_inputs = [input_ids, input_mask, segment_ids]
    else:
        model_inputs = [input]
    model_outputs = output
    model_y = y
    net_configs = [lr, is_train]
    optimizer = tf.train.AdamOptimizer(lr)
    model = modelModule.ModelModule(model_inputs, model_outputs, model_y, loss, optimizer, net_configs,
                                    metrics=accuracy)
    return model, bert_vars


model, bert_vars = class_model()


def train(train_text, train_label, test_text, test_label, train_num, learning_rate, batch_size):
    print("train")
    if use_bert:
        tokenizer = tokenization.FullTokenizer(bert_model_base_dir+"/vocab.txt", do_lower_case=True)
    test_label = tf.keras.utils.to_categorical(test_label, output_size)
    test_input_ids, test_input_mask, test_segment_ids, _ = convert_batch_data(test_text, tokenizer)
    v_inputs_feed = [test_input_ids, test_input_mask, test_segment_ids]
    v_outputs_feed = test_label
    v_net_configs_feed = [learning_rate, 0]
    if bert_vars is not None:
        restore_saver = tf.train.Saver(bert_vars)
    saver = tf.train.Saver()
    init = tf.global_variables_initializer()
    config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
    with tf.Session(config=config) as sess:
        print("tf.Session()")
        sess.run(init)
        if bert_vars is not None:
            restore_saver.restore(sess, bert_model_base_dir+"/bert_model.ckpt")
        print("restore saver")
        step = 0
        for i in range(train_num):
            position = 0
            while position < len(train_text):
                # if i == 0:
                    # print("batch")
                train_text, train_label, batch_x, batch_y, position = next_batch(batch_size, train_text, train_label,
                                                                                 position)
                batch_y = tf.keras.utils.to_categorical(batch_y, output_size)
                batch_input_ids, batch_input_mask, batch_segment_ids, _ = convert_batch_data(batch_x, tokenizer)
                tr_inputs_feed = [batch_input_ids, batch_input_mask, batch_segment_ids]
                tr_net_configs_feed = [learning_rate, 1]
                # if i == 0:
                    # print("batch fit start")
                result = model.batch_fit(sess, tr_inputs_feed, batch_y, tr_net_configs_feed, v_inputs_feed,
                                         v_outputs_feed, v_net_configs_feed, batch_size)
                # if i == 0:
                    # print("batch fit end")
                if step % 25 == 0:
                    print("i={},step={},result={}".format(i, step, result))
                step += 1
            if result["v_metrics"] > 0.5 and result["tr_metrics"] > 0.9:
                predict = model.predict(sess, v_inputs_feed, v_net_configs_feed, batch_size, is_in_train=True)
                predict["predict"] = np.argmax(predict["predict"], axis=-1)
                print("predict=", predict["predict"].tolist())
                print("      y=", test_label)

    return


def main():
    datapath = "../data/classification.json"
    keyword_path = "../data/keyword.xlsx"
    train_text, train_label, test_text, test_label = read_classification_data(datapath,
                                                                              depreated_text="DirtyDeedsDoneDirtCheap",
                                                                              data_augmentation_label=2,
                                                                              test_percent=0.5,
                                                                              keyword_path=keyword_path)
    print(len(train_label), len(test_label))
    train_num = 100
    learning_rate = 0.0005
    batch_size = 128
    train(train_text, train_label, test_text, test_label, train_num, learning_rate, batch_size)


if __name__ == '__main__':
    main()



