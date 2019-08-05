from nlp.ner.nerDataProcess import *
import tensorflow as tf
import numpy as np
from bert import modeling
from bert import tokenization
from module.tfversion import baseNet, modelModule
from tensorflow.contrib import crf

class GammaNerNet(baseNet.BaseNet):
    def __init__(self, layers=None):
        super(GammaNerNet, self).__init__(layers)

    def net(self, inputs):
        # output = inputs["input"]
        bert_config = inputs["bert_config"]
        input_ids = inputs["input_ids"]
        input_mask = inputs["input_mask"]
        segment_ids = inputs["segment_ids"]
        bert_scope = inputs["bert_scope"]
        regularizer = inputs["regularizer"]
        output_size = inputs["output_size"]
        y = inputs["y"]
        squence_lengths = inputs["squence_lengths"]
        model = modeling.BertModel(config=bert_config, is_training=True, input_ids=input_ids, input_mask=input_mask,
                                   token_type_ids=segment_ids, use_one_hot_embeddings=False, scope=bert_scope)
        output = model.get_sequence_output()
        output = tf.keras.layers.Masking(0)(output)
        output = tf.keras.layers.LSTM(units=1024, kernel_initializer="he_normal", recurrent_initializer="he_normal",
                                      kernel_regularizer=regularizer, recurrent_regularizer=regularizer,
                                      activity_regularizer=regularizer, return_sequences=True)(output)
        output = tf.keras.layers.Dense(units=output_size, activation="relu", kernel_initializer="he_normal",
                                       kernel_regularizer=regularizer, activity_regularizer=regularizer)(output)
        crf_loss, trans_params = crf.crf_log_likelihood(output, tag_indices=y, sequence_lengths=squence_lengths)
        loss = tf.reduce_mean(-crf_loss)
        decode_tags, _ = crf.crf_decode(output, trans_params, squence_lengths)
        bert_vars = tf.global_variables(bert_scope)
        outputs = {"output": decode_tags, "bert_vars": bert_vars, "loss": loss}
        return outputs

def nermodel(bert_model_base_dir, unique_label: list, model_save_path):
    input_ids = tf.placeholder(tf.int32,shape=[None, None], name="input_ids")
    input_mask = tf.placeholder(tf.int32, shape=[None, None], name="input_mask")
    segment_ids = tf.placeholder(tf.int32, shape=[None, None], name="segment_ids")
    y = tf.placeholder(tf.int32, shape=[None, None], name="y")
    squence_lengths = tf.placeholder(tf.int32, shape=[None], name="squence_lengths")
    lr = tf.placeholder("float")
    bert_config = modeling.BertConfig.from_json_file(bert_model_base_dir+"/bert_config.json")
    bert_scope = "bert"
    regularizer = None
    inputs = {"input_ids": input_ids,
              "input_mask": input_mask,
              "segment_ids": segment_ids,
              "bert_config": bert_config,
              "bert_scope": bert_scope,
              "regularizer": regularizer,
              "output_size": len(unique_label),
              "y": y,
              "squence_lengths": squence_lengths
    }
    net = GammaNerNet()
    outputs = net(inputs)
    output = outputs["output"]
    bert_vars = outputs["bert_vars"]
    loss = outputs["loss"]

    # loss = tf.reduce_mean(tf.keras.losses.sparse_categorical_crossentropy(y, output))
    true_pad_label_num = tf.reduce_sum(tf.cast(tf.equal(y, tf.constant(0)), "float"))

    true_O_label_num = tf.reduce_sum(tf.cast(tf.equal(y, tf.constant(unique_label.index("O"))), "float"))
    true_FIX_label_num = tf.reduce_sum(tf.cast(tf.equal(y,tf.constant(unique_label.index("[FIX]"))), "float"))

    pad_label_correct = tf.cast(tf.equal(y, tf.constant(0)), tf.int32) + \
                        tf.cast(tf.equal(output, tf.constant(0)), tf.int32)
    pad_label_correct = tf.reduce_sum(tf.cast(tf.equal(pad_label_correct, tf.constant(2)), "float"))

    O_label_correct = tf.cast(tf.equal(y, tf.constant(unique_label.index("O"))), tf.int32) + \
                        tf.cast(tf.equal(output, tf.constant(unique_label.index("O"))), tf.int32)
    O_label_correct = tf.reduce_sum(tf.cast(tf.equal(O_label_correct, tf.constant(2)), "float"))

    FIX_label_correct = tf.cast(tf.equal(y, tf.constant(unique_label.index("[FIX]"))), tf.int32) + \
                        tf.cast(tf.equal(output, tf.constant(unique_label.index("[FIX]"))), tf.int32)
    FIX_label_correct = tf.reduce_sum(tf.cast(tf.equal(FIX_label_correct, tf.constant(2)), "float"))

    accuracy_all = tf.reduce_sum(tf.cast(tf.equal(y, output), "float"))
    entity_true = tf.reduce_sum(tf.cast(tf.equal(y, y), "float")) - true_pad_label_num - true_O_label_num - \
                  true_FIX_label_num
    accuracy = (accuracy_all - pad_label_correct - O_label_correct - FIX_label_correct)/entity_true
    modelinputs = [input_ids, input_mask, segment_ids, squence_lengths]
    metrics = [true_pad_label_num, true_O_label_num, true_FIX_label_num, pad_label_correct, O_label_correct,
               FIX_label_correct, accuracy_all, entity_true, accuracy]
    optimizer = tf.train.AdamOptimizer(lr)
    model = modelModule.ModelModule(modelinputs, output, y, loss, optimizer, net_configs=lr,
                                    model_save_path=model_save_path, metrics=metrics, var_list=tf.global_variables())
    return model, bert_vars

bert_model_base_dir = "../../bertapi/base_model/cased_L-12_H-768_A-12"
dataPath = "../data/ner.json"
model_save_path = "../model/gammaNerNet/model.ckpt"
textlist, labellist, unique_label = read_data(dataPath, pad_word="[PAD]", use_bert=True)
model, bert_vars = nermodel(bert_model_base_dir, unique_label, model_save_path)


def train(textlist, labellist, bert_model_base_dir, train_num, batch_size):
    tokenizer = tokenization.FullTokenizer(bert_model_base_dir+"/vocab.txt", do_lower_case=False)
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()
    saver1 = tf.train.Saver(bert_vars)
    config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
    with tf.Session(config=config) as sess:
        sess.run(init)
        saver1.restore(sess, bert_model_base_dir+"/bert_model.ckpt")
        pre_metrics = 0
        step = 0
        print("[true_pad_label_num, true_O_label_num, true_FIX_label_num, pad_label_correct, O_label_correct, ",
              "FIX_label_correct, accuracy_all, entity_true, accuracy]")
        for i in range(train_num):
            position = 0
            while position < len(textlist):
                textlist, labellist, batch_sample, batch_sample_label, position = next_batch(textlist, position,
                                                                                             batch_size,
                                                                                             labellist=labellist)
                _, batch_input_ids, batch_input_mask, batch_segment_ids,\
                batch_label,batch_sequence_lengths = convert_batch_data(batch_sample, batch_sample_label, tokenizer,
                                                                        unique_label=unique_label, pad_word="[PAD]")
                inputs = [batch_input_ids, batch_input_mask, batch_segment_ids, batch_sequence_lengths]
                # print(batch_label.shape,batch_label.dtype)
                result = model.batch_fit(sess, inputs, batch_label, 0.005-0.005*i/(train_num*2), batch_size=batch_size)
                if step % 25 == 0:
                    print("i=", i, "result=", result)
                step += 1
            print("i=", i, "result=", result)
            if result["tr_metrics"][-1] > pre_metrics:
                pre_metrics = result["tr_metrics"][-1]
                saver.save(sess, model_save_path)


def main():
    train(textlist, labellist, bert_model_base_dir, train_num=100, batch_size=128)


if __name__ == '__main__':
    main()

