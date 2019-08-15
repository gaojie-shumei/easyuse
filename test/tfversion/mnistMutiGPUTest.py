from module.tfversion import baseNet, modelModule
import tensorflow as tf
import numpy as np
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  #只显示error

class MnistNet(baseNet.BaseNet):
    def net(self, inputs):
        input = inputs["input"]
        output = tf.keras.layers.Dense(units=10, activation="softmax")(input)
        return output

def average_gradients(tower_grads):
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        grads = []
        for g, var in grad_and_vars:
            if g is not None:
                grads.append(tf.expand_dims(g, 0))
            else:
                grads.append(tf.expand_dims(tf.zeros_like(var), 0))

        # grads = [tf.expand_dims(g, 0) for g, _ in grad_and_vars]
        grads = tf.concat(grads, 0)
        grad = tf.reduce_mean(grads, 0)
        grad_and_var = (grad, grad_and_vars[0][1])
        # [(grad0, var0),(grad1, var1),...]
        average_grads.append(grad_and_var)
    return average_grads


def mnist_model(device=None, reuse=tf.AUTO_REUSE, gpu_num=0):
    if device is None:
        device = "/cpu:0"
    with tf.device(device), tf.variable_scope("", reuse=reuse):
        x = tf.placeholder("float", shape=[None, 784], name="x")
        y = tf.placeholder("float", shape=[None], name="y")
        optimizer = tf.train.AdamOptimizer(0.0005)
        net = MnistNet()
        if gpu_num != 0:
            split_num = gpu_num
            while(split_num!=0):
                try:
                    _x = tf.split(x, split_num, axis=0)
                    _y = tf.split(y, split_num, axis=0)
                    break
                except:
                    split_num -= 1
            output, loss, tower_grads = [], [], []
            for i in range(split_num):
                with tf.device("/gpu:%d"%(i)):
                    _output = net({"input": _x[i]})
                    output.append(_output)
                    _loss = tf.reduce_mean(tf.keras.losses.sparse_categorical_crossentropy(_y[i], _output))
                    loss.append(tf.expand_dims(_loss, 0))
                    tower_grads.append(optimizer.compute_gradients(_loss))
            output = tf.concat(output, axis=0)
            loss = tf.reduce_mean(tf.concat(loss, axis=0))
            grad_and_vars = average_gradients(tower_grads)
        else:
            output = net({"input": x})
            loss = tf.reduce_mean(tf.keras.losses.sparse_categorical_crossentropy(y, output))
            grad_and_vars = None
        accuracy = tf.reduce_mean(tf.keras.metrics.sparse_categorical_accuracy(y, output))
        if grad_and_vars is not None:
            train_ops = optimizer.apply_gradients(grad_and_vars)
        else:
            train_ops = optimizer.minimize(loss)
        # print("loss=", loss)
        model = modelModule.ModelModule(x, output, y, loss, train_ops, metrics=accuracy)
        return model


gpu_num = 8
model = mnist_model("/cpu:0", gpu_num=gpu_num)


def train(x_train, y_train):
    with tf.device("/cpu:0"), tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(100):
            position = 0
            while position < x_train.shape[0]:
                x_train, y_train, batch_x, batch_y, position = next_batch(x_train, y_train, position,
                                                                          128 if gpu_num==0 else 128*gpu_num)
                result = model.batch_fit(sess, batch_x, batch_y)
    return

def read_mnist_data(path):
    f = np.load(path)
    x_train, y_train = f['x_train'], f['y_train']
    x_test, y_test = f['x_test'], f['y_test']
    f.close()
    return (x_train, y_train), (x_test, y_test)


def next_batch(x_train, y_train, position, batch_size, shuffle=True, randomstate=np.random.randint(0, 100)):
    temp_x, temp_y = x_train[position:], y_train[position:]
    if shuffle:
        np.random.seed(randomstate)
        np.random.shuffle(temp_x)
        np.random.seed(randomstate)
        np.random.shuffle(temp_y)
    x_train = np.r_[x_train[0:position], temp_x]
    y_train = np.r_[y_train[0:position], temp_y]
    if batch_size < temp_x.shape[0]:
        batch_x = temp_x[0:batch_size]
        batch_y = temp_y[0:batch_size]
    elif batch_size < x_train.shape[0]:
        batch_x = x_train[-batch_size:]
        batch_y = y_train[-batch_size:]
    else:
        res = batch_size
        batch_x, batch_y = None, None
        while res > x_train.shape[0]:
            if batch_x is None:
                batch_x = x_train
                batch_y = y_train
            else:
                batch_x = np.r_[batch_x, x_train]
                batch_y = np.r_[batch_y, y_train]
            res = res - x_train.shape[0]
        if res > 0:
            batch_x = np.r_[batch_x, x_train[0:res]]
            batch_y = np.r_[batch_y, y_train[0:res]]
    position += batch_size
    return x_train, y_train, batch_x, batch_y, position

def main():
    mnist_path = "../../data/mnist/mnist.npz"
    # get data
    # (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    (x_train, y_train), (x_test, y_test) = read_mnist_data(mnist_path)
    print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)
    x_train = x_train.reshape(x_train.shape[0], -1).astype("float")
    x_test = x_test.reshape(x_test.shape[0], -1).astype("float")
    print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)
    print(y_train.dtype, y_test.dtype)
    y_train = y_train.astype(np.int32)
    y_test = y_test.astype(np.int32)
    train(x_train[0:7], y_train[0:7])


if __name__ == "__main__":
    main()