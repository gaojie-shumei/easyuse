import tensorflow as tf
import numpy as np
from datautil import nlpDataUtil
from nlp.test.read_data_for_conll2003 import *
from keras_contrib import layers as kcl
from keras_contrib import metrics as kcm
from keras_contrib import losses as kcloss

train_path = "../../../nlp//ner/data/conll2003/eng.train"
v_path = "../../../nlp/ner/data/conll2003/eng.testa"
test_path = "../../../nlp/ner/data/conll2003/eng.testb"

data,data_pos_tag,data_chunk_tag,label,position,label_index = read_data(train_path,encoding="utf-8",position=0,
                                                                        read_data_size=None,padding_str="<pad>")

datautil = nlpDataUtil.NLPDataUtil(use_for_bert=False, label_setlist=label_index)
output_size = len(datautil.label_setlist)
datautil.word2vec(sentences=data,size=128,min_count=1,sg=1,name="data.model")

mask = 0
def generate_batch_data_for_keras(data,label,datautil:nlpDataUtil.NLPDataUtil,batch_size):
    position = 0
    while 1:
        if position>=len(data):
            position = 0
        data,label,batch_data,batch_label,position = datautil.next_batch(batch_size,data,label,position)
        pad_data, pad_y_data, actual_lengths = datautil.padding(batch_data,batch_label)
        batch_x,batch_y = datautil.format(pad_data,pad_y_data)
        batch_x = datautil.set_mask(batch_x,actual_lengths,mask)
        batch_y = tf.keras.utils.to_categorical(batch_y,num_classes=output_size)
        # print(batch_y.shape)
        yield (batch_x, batch_y)

def create_model():
    regularizer = None
    input = tf.keras.Input(shape=[None,128])
    # mask = tf.keras.layers.Input(shape=[None])
    output = tf.keras.layers.Masking(mask_value=mask)(input)
    output = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(units=256,kernel_initializer="he_normal",
                                                                recurrent_initializer="he_normal",
                                                                kernel_regularizer=regularizer,
                                                                recurrent_regularizer=regularizer, dropout=0.3,
                                                                return_sequences=True), merge_mode="ave")(output)
    output = tf.keras.layers.LSTM(units=512, kernel_initializer="he_normal", recurrent_initializer="he_normal",
                                  kernel_regularizer=regularizer, recurrent_regularizer=regularizer, dropout=0.3,
                                  return_sequences=True)(output)
    output = tf.keras.layers.Dense(units=output_size, activation="relu",
                                   kernel_initializer='he_normal', kernel_regularizer=regularizer)(output)
    # output = tf.reshape(output,shape=[-1,output_size])
    print(output)
    crf_layer = kcl.crf.CRF(output_size,sparse_target=True,learn_mode="join",test_mode="viterbi")
    output = crf_layer(output)

    model = tf.keras.Model(inputs=input, outputs=output)
    # model.compile(optimizer="adam", loss=kcloss.crf_loss,
    #               metrics=[kcm.crf_accuracy])
    model.summary()
    return model

def main():
    model = create_model()
    # import tensorflow_addons as tfa
    # batch_size = 128
    # steps_per_epoch = len(data)//batch_size + 0 if len(data)%batch_size==0 else 1
    # history = model.fit_generator(generator=generate_batch_data_for_keras(data,label,datautil,batch_size),
    #                               steps_per_epoch=steps_per_epoch,epochs=100)
    import matplotlib.pyplot as plt
    plt.plot(history.history['acc'])
    plt.show()

if __name__=="__main__":
    main()




