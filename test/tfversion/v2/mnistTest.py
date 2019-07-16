'''
Created on 2019年7月16日

@author: gaojiexcq
'''
import tensorflow as tf
import tensorflow.keras.layers as tkl


def model_with_keras_layers():
    '''
    using keras.Sequential()
    '''
    
    return

def model_with_keras_function():
    '''
    using keras function() to constract the complex net
    '''
    return 
    
    
def main():
    #get data
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    print(x_train.shape,y_train.shape,x_test.shape,y_test.shape)
    
if __name__=="__main__":
    main()