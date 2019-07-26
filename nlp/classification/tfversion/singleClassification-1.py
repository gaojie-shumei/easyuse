import tensorflow as tf
from module.tfversion import modelModule
from module.tfversion import baseNet
import numpy as np


class ClassificationNet(baseNet.BaseNet):
    def __init__(self, layers=None):
        super(ClassificationNet, self).__init__(layers)

    def net(self, inputs):
        outputs = inputs
        # print(self.layers)
        if self.layers is not None:
            if isinstance(self.layers, list):
                for layer in self.layers:
                    outputs = layer(outputs)
            else:
                outputs = self.layers(outputs)
        return outputs


def create_model():
    input = tf.placeholder("float", shape=[None, 784], name="input")
    y = tf.placeholder(tf.int32, shape=[None], name="y")
    layers = []
    layers.append(tf.keras.layers.Dense(units=10, activation=tf.keras.activations.softmax))
    net = ClassificationNet(layers)
    output = net(input)
    loss = tf.reduce_mean(tf.keras.losses.sparse_categorical_crossentropy(y, output))
    accuracy = tf.reduce_mean(tf.cast(tf.equal(y, tf.argmax(output, axis=-1, output_type=tf.int32)), "float"))
    optimizer = tf.train.AdamOptimizer(0.001)
    model = modelModule.ModelModule(input, output, y, loss, optimizer, metrics=accuracy)
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


model = create_model()


def train(x_train, y_train, x_test, y_test, train_num, batch_size):
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        for i in range(train_num):
            position = 0
            while position < x_train.shape[0]:
                x_train, y_train, batch_x, batch_y, position = next_batch(x_train, y_train, position, batch_size)
                result = model.batch_fit(sess, batch_x, batch_y, v_inputs_feed=x_test, v_outputs_feed=y_test)
            print("i=", i, "result=", result)

def test(x_test,y_test):
    with tf.Session() as sess:
        result = model.evaluation(sess, x_test, y_test)
        predict_result = model.predict(sess,x_test)
        predict_result["predict_outputs"] = np.argmax(predict_result["predict_outputs"],axis=-1)
    print("result=", result)
    print(predict_result)
    print("standard outputs=", y_test)

def main():
    path = "../../../data/mnist/mnist.npz"
    (x_train, y_train), (x_test, y_test) = read_mnist_data(path)
    print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)
    x_train = x_train.reshape(x_train.shape[0], -1).astype(np.float32)
    x_test = x_test.reshape(x_test.shape[0], -1).astype(np.float32)
    y_train = y_train.astype(np.int32)
    y_test = y_test.astype(np.int32)
    print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)
    print(y_train.dtype, y_test.dtype)
    print("1"*33, "\ntrain\n")
    train(x_train, y_train, x_test, y_test, train_num=100, batch_size=128)
    print("1" * 33, "\ntest\n")
    test(x_test, y_test)


if __name__ == '__main__':
    main()
