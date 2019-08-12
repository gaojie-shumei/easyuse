from module.tfversion.dataWrapper import *
from module.tfversion import baseDataProcessor
import os
from datetime import datetime
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  #只显示error

def read_mnist_data(path):
    f = np.load(path)
    x_train, y_train = f['x_train'], f['y_train']
    x_test, y_test = f['x_test'], f['y_test']
    f.close()
    return (x_train, y_train), (x_test, y_test)


x = tf.placeholder("float",shape=[None,784],name="x")
y = tf.placeholder(tf.int32,shape=[None],name="y")


def model():
    out = tf.layers.dense(x,128,activation=tf.nn.softmax)
    out = tf.layers.dense(out,10,activation=tf.nn.softmax)
    return out


out = model()
loss = tf.losses.sparse_softmax_cross_entropy(y,out)
acc = tf.reduce_mean(tf.cast(tf.equal(y,tf.argmax(out,axis=-1,output_type=tf.int32)),"float"))



def train(x_train,y_train,x_test,y_test,train_num,learning_rate,batch_size):
    mnist_data_processor = baseDataProcessor.MnistDataProcessor()
    train_samples = mnist_data_processor.creat_samples(x_train,y_train)
    train_features = mnist_data_processor.samples2features(train_samples)
    test_samples = mnist_data_processor.creat_samples(x_test, y_test)
    test_features = mnist_data_processor.samples2features(test_samples)

    wrapper = TFDataWrapper()
    _, train_data, train_init = wrapper(train_features,batch_size,gpu_num=0,is_train=True,drop_remainder=False)
    _, test_data, test_init = wrapper(test_features, batch_size, gpu_num=0, is_train=False, drop_remainder=False)
    with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
        train_ops = tf.train.AdamOptimizer(learning_rate).minimize(loss)
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run([init,train_init])
        for i in range(train_num):
            for j in range(x_train.shape[0]//batch_size):
                start = datetime.now()
                batch_x,batch_y = sess.run([train_data["x"],train_data["y"]])
                _,tr_loss,tr_acc = sess.run([train_ops,loss,acc],feed_dict={x:batch_x,y:batch_y})
                sess.run(test_init)
                v_loss = 0
                v_acc = 0
                while True:
                    try:
                        batch_x,batch_y = sess.run([test_data["x"],test_data["y"]])
                    except:
                        break
                    temp_loss,temp_acc = sess.run([loss,acc],feed_dict={x:batch_x,y:batch_y})
                    v_loss += temp_loss
                    v_acc += temp_acc
                v_loss /= (x_test.shape[0]//batch_size)
                v_acc /= (x_test.shape[0]//batch_size)
                end = datetime.now()
                print("batch fit time=",(end-start).total_seconds())
                if j%25==0:
                    print(i,j,tr_loss,tr_acc,v_loss,v_acc)
            print(i,j,tr_loss,tr_acc,v_loss,v_acc)



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



if __name__ == '__main__':
    main()