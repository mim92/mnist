from keras.datasets import mnist
from keras.utils import to_categorical
import numpy as np


def prepare_mnist_data():
    class_num = 10
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    
    x_train = (255 - x_train.astype('float32')) / 255
    x_test = (255 - x_test.astype('float32')) / 255

    x_train = np.expand_dims(x_train, 3)
    x_test = np.expand_dims(x_test, 3)

    # change shape (sample_size, ) to (sample_size, class_num)
    y_train = to_categorical(y_train, class_num)
    y_test = to_categorical(y_test, class_num)

    return (x_train, y_train), (x_test, y_test)
