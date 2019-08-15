import tensorflow as tf
import numpy as np
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  #只显示error
with tf.device("/cpu:0"):
    x = tf.placeholder("float", [None, 784])
    y = tf.placeholder("float", [None,10])
def model():
    with tf.device("/gpu:0"):
        out = tf.keras.layers.Dense(128, tf.keras.activations.relu)(x)
    with tf.device("/gpu:1"):
        out = tf.keras.layers.Dense(256,tf.keras.activations.relu)(out)
    with tf.device("/gpu:2"):
        out = tf.keras.layers.Dense(10,tf.keras.activations.softmax)(out)
    return out

with tf.device("/cpu:0"):
    out = model()
    index = tf.argmax(out,axis=-1)
    y_index = tf.argmax(y,axis=-1)
    loss = tf.reduce_mean(tf.keras.losses.categorical_crossentropy(y,out))
    accuracy = tf.reduce_mean(tf.keras.metrics.categorical_accuracy(y,out))
    accuracy1 = tf.reduce_mean(tf.cast(tf.equal(y_index, index), "float"))
    train_ops = tf.train.AdamOptimizer(0.0005).minimize(loss)

def train(x_train, y_train):
    with tf.device("/cpu:0"), tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        step = 0
        for i in range(100):
            position = 0
            while position < x_train.shape[0]:
                x_train, y_train, batch_x, batch_y, position = next_batch(x_train, y_train, position, 128)
                _,tr_loss,tr_acc, tr_acc1, tr_out = sess.run([train_ops, loss, accuracy,accuracy1, index],
                                                             feed_dict={x:batch_x, y:tf.keras.utils.to_categorical(batch_y,10)})
                np_acc = np.mean(tr_out==batch_y)
                if step %100 == 0:
                    print(i, "\t", tr_loss, "\t", tr_acc,"\t",tr_acc1,"\t",np_acc)
                step += 1
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
    print(y_train[0:20])
    print(y_train.astype(np.int64)[0:20])
    y_train = y_train.astype(np.int64)
    y_test = y_test.astype(np.int64)
    train(x_train, y_train)


if __name__ == "__main__":
    main()