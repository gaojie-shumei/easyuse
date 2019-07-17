'''
Created on 2019年7月16日

@author: gaojiexcq
'''
import tensorflow as tf
from tensorflow.keras import layers as tkl
import numpy as np

def model_with_keras_layers():
    '''
    using keras.Sequential()
    '''
    # model = tf.keras.Sequential()
    # model.add(tkl.Dense(64, activation='relu', input_shape=(784,)))
    # model.add(tkl.Dense(32, activation='relu', kernel_initializer=tf.keras.initializers.glorot_normal))
    # model.add(tkl.Dense(32, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)))
    # model.add(tkl.Dense(10, activation='softmax'))

    model = tf.keras.Sequential([
        tkl.Dense(64, activation='relu', kernel_initializer='he_normal', input_shape=(784,)),
        tkl.Dense(64, activation='relu', kernel_initializer='he_normal'),
        tkl.Dense(64, activation='relu', kernel_initializer='he_normal',
                     kernel_regularizer=tf.keras.regularizers.l2(0.01)),
        tkl.Dense(10, activation='softmax')
    ])
    return model

def model_with_keras_function():
    '''
    using keras function() to constract the complex net
    '''
    input_x = tf.keras.Input(shape=(784,))
    hidden1 = tkl.Dense(64, activation='relu', kernel_initializer='he_normal')(input_x)
    hidden2 = tkl.Dense(64, activation='relu', kernel_initializer='he_normal')(hidden1)
    hidden3 = tkl.Dense(64, activation='relu', kernel_initializer='he_normal',
                           kernel_regularizer=tf.keras.regularizers.l2(0.01))(hidden2)
    output = tkl.Dense(10, activation='softmax')(hidden3)

    model = tf.keras.Model(inputs=input_x, outputs=output)
    return model

def read_mnist_data(path):
    f = np.load(path)
    x_train, y_train = f['x_train'], f['y_train']
    x_test, y_test = f['x_test'], f['y_test']
    f.close()
    return (x_train, y_train), (x_test, y_test)
def next_batch(x_train,y_train,position,batch_size,shuffle=True,randomstate=np.random.randint(0,100)):
    temp_x,temp_y = x_train[position:],y_train[position:]
    if shuffle:
        np.random.seed(randomstate)
        np.random.shuffle(temp_x)
        np.random.seed(randomstate)
        np.random.shuffle(temp_y)
    x_train = np.r_[x_train[0:position],temp_x]
    y_train = np.r_[y_train[0:position],temp_y]
    if batch_size<temp_x.shape[0]:
        batch_x = temp_x[0:batch_size]
        batch_y = temp_y[0:batch_size]
    else:
        batch_x = temp_x
        batch_y = temp_y
    position += batch_size
    return x_train,y_train,batch_x,batch_y,position

def generator(x_train,y_train,batch_size):
    position = 0
    while 1:
        if position >= x_train.shape[0]:
            position = 0
        x_train, y_train, batch_x, batch_y, position = next_batch(x_train,y_train,position,batch_size)
        yield (batch_x,batch_y)

def main():
    mnist_path = "../../data/mnist/mnist.npz"
    #get data
    # (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    (x_train, y_train), (x_test, y_test) = read_mnist_data(mnist_path)
    print(x_train.shape,y_train.shape,x_test.shape,y_test.shape)
    x_train = x_train.reshape(x_train.shape[0],-1)
    x_test = x_test.reshape(x_test.shape[0],-1)
    print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)
    print(y_train.dtype,y_test.dtype)
    model = model_with_keras_layers()
    model.compile(optimizer=tf.keras.optimizers.Adam(0.001),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    print(model.summary())
    # history = model.fit(x_train,y_train,128,1000,validation_split=0.2,verbose=1)
    steps_per_epoch = x_train.shape[0]//128 + 0 if x_train.shape[0]%128==0 else 1
    history = model.fit_generator(generator=generator(x_train,y_train,128),steps_per_epoch=steps_per_epoch,epochs=100)
    print(history.__dict__)
    import matplotlib.pyplot as plt
    plt.plot(history.history['acc'])
    # plt.plot(history.history['val_accuracy'])
    # plt.legend(['training', 'validation'], loc='upper left')
    plt.show()
if __name__=="__main__":
    main()