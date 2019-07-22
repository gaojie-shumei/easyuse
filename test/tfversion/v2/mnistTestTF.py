import tensorflow as tf
import numpy as np
def create_model():
    x = tf.keras.Input(shape=(784,), name="x", dtype=tf.float32)
    out = tf.keras.layers.Dense(units=64, activation=tf.nn.relu, kernel_initializer="he_normal")(x)
    out = tf.keras.layers.Dense(units=32, activation=tf.nn.relu, kernel_initializer="he_normal")(out)
    out = tf.keras.layers.Dense(units=10, activation=tf.nn.softmax, kernel_initializer="he_normal")(out)
    model = tf.keras.Model(inputs=x, outputs=out)
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


def train(x_train,y_train,x_test,y_test,train_num,learning_rate,batch_size):
    model = create_model()
    optimizer = tf.keras.optimizers.Adam(learning_rate)
    step = 0
    for i in range(train_num):
        position = 0
        while(position < x_train.shape[0]):
            x_train, y_train, batch_x, batch_y, position = next_batch(x_train,y_train,position,batch_size)
            with tf.GradientTape() as tape:
                out = model(tf.convert_to_tensor(batch_x))
                loss = tf.reduce_mean(tf.losses.sparse_categorical_crossentropy(tf.convert_to_tensor(batch_y),out))
                accuracy = tf.reduce_mean(tf.keras.metrics.sparse_categorical_accuracy(tf.convert_to_tensor(batch_y),out))
                grad = tape.gradient(loss, sources=model.trainable_variables)
                # grad = tf.clip_by_value(grad, clip_value_min=-5, clip_value_max=5,name="grad-clip")
                optimizer.apply_gradients(zip(grad, model.trainable_variables))
            if step %100==0:
                out1 = model(tf.convert_to_tensor(x_test))
                t_loss = tf.reduce_mean(tf.losses.sparse_categorical_crossentropy(tf.convert_to_tensor(y_test),out1))
                t_acc = tf.reduce_mean(tf.keras.metrics.sparse_categorical_accuracy(tf.convert_to_tensor(y_test),out1))

                print("epochs=",i,",step=",step,",loss={:f}".format(loss),",acc={:f}".format(accuracy),
                      ",test_loss={:f}".format(t_loss),",test_acc={:f}".format(t_acc))

            step += 1

def main():
    mnist_path = "../../data/mnist/mnist.npz"
    # get data
    # (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    (x_train, y_train), (x_test, y_test) = read_mnist_data(mnist_path)
    print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)
    x_train = x_train.reshape(x_train.shape[0], -1)
    x_test = x_test.reshape(x_test.shape[0], -1)
    print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)
    print(y_train.dtype, y_test.dtype)

    train(x_train,y_train,x_test,y_test,100,0.0005,128)

if __name__=="__main__":
    main()