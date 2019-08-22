from module.tfversion.dataWrapper import *
from module.tfversion import baseDataProcessor
from module.tfversion import modelModule
import os
from datetime import datetime
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  #只显示error


def read_mnist_data(path):
    f = np.load(path)
    x_train, y_train = f['x_train'], f['y_train']
    x_test, y_test = f['x_test'], f['y_test']
    f.close()
    return (x_train, y_train), (x_test, y_test)


x = tf.placeholder("float",shape=[None, 784], name="x")
is_real_sample = tf.placeholder(tf.int32, shape=[None], name="is_real_sample")
y = tf.placeholder(tf.int32, shape=[None], name="y")


def mnist_model():
    out = tf.layers.dense(x, 128, activation=tf.nn.relu)
    out = tf.layers.dense(out, 10, activation=tf.nn.softmax)
    return out


out = mnist_model()
tf_index = tf.cast(is_real_sample, "float")
loss = tf.losses.sparse_softmax_cross_entropy(y, out, reduction=tf.losses.Reduction.NONE)
loss = tf.reduce_sum(tf.multiply(tf_index, loss)) / tf.reduce_sum(tf_index)
acc = tf.cast(tf.equal(y, tf.argmax(out, axis=-1, output_type=tf.int32)), "float")
acc = tf.reduce_sum(tf.multiply(tf_index, acc)) / tf.reduce_sum(tf_index)


def train(x_train,y_train,x_test,y_test,train_num,learning_rate,batch_size):
    mnist_data_processor = baseDataProcessor.MnistDataProcessor()
    train_samples = mnist_data_processor.creat_samples(x_train,y_train)
    train_features = mnist_data_processor.samples2features(train_samples)
    test_samples = mnist_data_processor.creat_samples(x_test, y_test)
    test_features = mnist_data_processor.samples2features(test_samples)
    # wrapper = TFDataWrapper()
    # _, train_data, train_init = wrapper(train_features, batch_size, is_train=True, drop_remainder=False,
    #                                     num_parallel_calls=0)
    # _, test_data, test_init = wrapper(test_features, batch_size, is_train=False, drop_remainder=False,
    #                                   num_parallel_calls=0)
    wrapper = TFRecordWrapper("train.tf_record",mnist_data_processor.features_typing_fn, False)
    wrapper1 = TFRecordWrapper("test.tf_record", mnist_data_processor.features_typing_fn, False)
    wrapper.write(train_features, batch_size, len(train_features))
    wrapper1.write(test_features, batch_size, len(test_features))
    _, train_data, train_init = wrapper.read(True, batch_size)
    _, test_data, test_init = wrapper1.read(False, batch_size)
    global_step = tf.train.get_or_create_global_step()
    with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
        train_ops = tf.train.AdamOptimizer(learning_rate).minimize(loss, global_step=global_step)
    init = tf.global_variables_initializer()
    model = modelModule.ModelModule([x, is_real_sample], out, y, loss, train_ops, None,
                                    "../model/mnistWrapper", acc, 0)
    with tf.Session() as sess:
        sess.run(init)
        model.fit(sess, train_num, [train_data["x"], train_data["is_real_sample"]], train_data["y"], None,
                  [test_data["x"], test_data["is_real_sample"]], test_data["y"],None, batch_size, False, True,
                  10, "model", train_init, test_init, True)
        result = model.evaluation(sess, [test_data["x"], test_data["is_real_sample"]], test_data["y"], None,
                                  batch_size, True, False, test_init)
        print(result)
        result = model.predict(sess, [test_data["x"], test_data["is_real_sample"]], None, batch_size, True, test_init)
        result["predict"] = np.argmax(result["predict"], axis=-1)
        print("predict=", result["predict"][0:20])
        print("      y=", y_test[0:20])
        # j = 0
        # for i in range(train_num):
        #     sess.run(train_init)
            # while True:
            #     # start = datetime.now()
            #     try:
            #         batch_x,batch_y = sess.run([train_data["x"],train_data["y"]])
            #     except tf.errors.OutOfRangeError:
            #         break
            #     _,tr_loss,tr_acc = sess.run([train_ops,loss,acc],feed_dict={x:batch_x,y:batch_y})
            #     sess.run(test_init)
            #     v_loss = 0
            #     v_acc = 0
            #     while True:
            #         try:
            #             batch_x,batch_y,is_real_sample = sess.run([test_data["x"],test_data["y"],
            #                                                        test_data["is_real_sample"]])
            #         except tf.errors.OutOfRangeError:
            #             break
            #         temp_loss, temp_acc = sess.run([loss,acc],feed_dict={x:batch_x,y:batch_y})
            #         v_loss += temp_loss
            #         v_acc += temp_acc
            #     v_loss /= (x_test.shape[0]//batch_size)
            #     v_acc /= (x_test.shape[0]//batch_size)
            #     # end = datetime.now()
            #     # print("batch fit time=",(end-start).total_seconds())
            #     if j%25==0:
            #         print(i,j,tr_loss,tr_acc,v_loss,v_acc)
            #     j += 1
            # print(i,j,tr_loss,tr_acc,v_loss,v_acc)
    return model



def main():
    mnist_path = "../../data/mnist/mnist.npz"
    # get data
    # (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    (x_train, y_train), (x_test, y_test) = read_mnist_data(mnist_path)
    print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)
    x_train = x_train.reshape(x_train.shape[0], -1).astype(np.float32)
    x_test = x_test.reshape(x_test.shape[0], -1).astype(np.float32)
    print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)
    print(y_train.dtype, y_test.dtype)
    y_train = y_train.astype(np.int32)
    y_test = y_test.astype(np.int32)
    train(x_train, y_train, x_test, y_test, train_num=100, learning_rate=0.0005, batch_size=128)


if __name__ == '__main__':
    main()
