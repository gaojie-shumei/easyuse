from nlp.ner.nerDataProcess import *
import tensorflow as tf
import numpy as np
from bert import modeling
from bert import tokenization
from module.tfversion import baseNet, modelModule
from tensorflow.contrib import crf
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  #只显示error


class GammaNer(baseNet.BaseNet):
    def net(self, inputs):
        with tf.variable_scope("", reuse=tf.AUTO_REUSE):
            bert_config = inputs["bert_config"]
            input_ids = inputs["input_ids"]
            input_mask = inputs["input_mask"]
            segment_ids = inputs["segment_ids"]
            bert_scope = inputs["bert_scope"]
            regularizer = inputs["regularizer"]
            output_size = inputs["output_size"]
            y = inputs["y"]
            squence_lengths = inputs["squence_lengths"]
            model = modeling.BertModel(config=bert_config, is_training=True, input_ids=input_ids,
                                       input_mask=input_mask,
                                       token_type_ids=segment_ids, use_one_hot_embeddings=False,
                                       scope=bert_scope)
            output = model.get_sequence_output()
            output = tf.keras.layers.Dense(units=output_size, activation="relu", kernel_initializer="he_normal",
                                           kernel_regularizer=regularizer,
                                           activity_regularizer=regularizer)(output)
            crf_loss, trans_params = crf.crf_log_likelihood(output, tag_indices=y, sequence_lengths=squence_lengths)
            loss = tf.reduce_mean(-crf_loss)
            decode_tags, _ = crf.crf_decode(output, trans_params, squence_lengths)
            outputs = {"output": decode_tags, "loss": loss}
        return outputs


with tf.device("/cpu:0"):
    init_checkpoint = "../../bertapi/base_model/cased_L-12_H-768_A-12"
    dataPath = "../data/ner.json"
    model_save_path = "../model/gammaNerNet/model.ckpt"
    textlist, labellist, unique_label = read_data(dataPath, pad_word="[PAD]", use_bert=True)
    input_ids = tf.placeholder(tf.int32,shape=[None, None], name="input_ids")
    input_mask = tf.placeholder(tf.int32, shape=[None, None], name="input_mask")
    segment_ids = tf.placeholder(tf.int32, shape=[None, None], name="segment_ids")
    y = tf.placeholder(tf.int32, shape=[None, None], name="y")
    squence_lengths = tf.placeholder(tf.int32, shape=[None], name="squence_lengths")
    lr = tf.placeholder("float")
    bert_config = modeling.BertConfig.from_json_file(init_checkpoint+"/bert_config.json")
    bert_scope = "bert"
    regularizer = None
    gpu_num = 0
    output_size = len(unique_label)


def ner_model():
    with tf.device("/cpu:0"):
        net = GammaNer()
        if gpu_num != 0:
            _input_ids = tf.split(input_ids, gpu_num)
            _input_mask = tf.split(input_mask, gpu_num)
            _segment_ids = tf.split(segment_ids, gpu_num)
            _y = tf.split(y, gpu_num)
            _squence_lengths = tf.split(squence_lengths, gpu_num)
            output = []
            loss = 0
            for i in range(gpu_num):
                with tf.device("/gpu:%d"%i):
                    inputs = {
                        "bert_config":bert_config,
                        "input_ids":_input_ids[i],
                        "input_mask":_input_mask[i],
                        "segment_ids":_segment_ids[i],
                        "bert_scope":bert_scope,
                        "regularizer":regularizer,
                        "output_size":output_size,
                        "y":_y[i],
                        "squence_lengths":_squence_lengths[i]
                    }
                    _outputs= net(inputs)
                    output.append(_outputs["output"])
                    loss += _outputs["loss"]
            output = tf.concat(output,axis=0)
            loss /= gpu_num
        else:
            inputs = {
                "bert_config": bert_config,
                "input_ids": input_ids,
                "input_mask": input_mask,
                "segment_ids": segment_ids,
                "bert_scope": bert_scope,
                "regularizer": regularizer,
                "output_size": output_size,
                "y": y,
                "squence_lengths": squence_lengths
            }
            outputs = net(inputs)
            output = outputs["output"]
            loss = outputs["loss"]
        assignment_map, _ = modeling.get_assignment_map_from_checkpoint(tf.global_variables(),
                                                                        init_checkpoint+"/bert_model.ckpt")
        tf.train.init_from_checkpoint(init_checkpoint+"/bert_model.ckpt", assignment_map)
    return output, loss

with tf.device("/cpu:0"):
    output, loss = ner_model()
    true_pad_label_num = tf.reduce_sum(tf.cast(tf.equal(y, tf.constant(0)), "float"))

    true_O_label_num = tf.reduce_sum(tf.cast(tf.equal(y, tf.constant(unique_label.index("O"))), "float"))
    true_FIX_label_num = tf.reduce_sum(tf.cast(tf.equal(y, tf.constant(unique_label.index("[FIX]"))), "float"))

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
    accuracy = (accuracy_all - pad_label_correct - O_label_correct - FIX_label_correct) / entity_true
    modelinputs = [input_ids, input_mask, segment_ids, squence_lengths]
    metrics = [true_pad_label_num, true_O_label_num, true_FIX_label_num, pad_label_correct, O_label_correct,
               FIX_label_correct, accuracy_all, entity_true, accuracy]
    optimizer = tf.train.AdamOptimizer(lr)
    with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
        train_ops = optimizer.minimize(loss)
    model = modelModule.ModelModule(modelinputs, output, y, loss, train_ops, lr,
                                    model_save_path,metrics,gpu_num)


def train(textlist, labellist, train_num, learning_rate, batch_size):
    with tf.device("/cpu:0"):
        print(tf.test.is_built_with_cuda())
        print(tf.test.is_gpu_available())
        tokenizer = tokenization.FullTokenizer(init_checkpoint+"/vocab.txt", do_lower_case=False)
        init = tf.global_variables_initializer()
        config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True), allow_soft_placement=True)
        _, tr_input_ids, tr_input_mask, tr_segment_ids, \
        tr_labels, tr_sequence_lengths = convert_batch_data(textlist, labellist, tokenizer,
                                                            unique_label=unique_label, pad_word="[PAD]")
        with tf.Session(config=config) as sess:
            sess.run(init)
            model.fit(sess, train_num, [tr_input_ids, tr_input_mask, tr_segment_ids, tr_sequence_lengths], tr_labels,
                      learning_rate, None, None, None, batch_size, False, True, None, None, None, None)


def main():
    train(textlist, labellist, train_num=100, learning_rate=1e-5, batch_size=64)


if __name__ == '__main__':
    main()
