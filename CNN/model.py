from keras.models import Model
from keras.layers import Conv2D, MaxPooling2D, Dropout, Input, Dense, Flatten
from keras.utils.vis_utils import plot_model
from keras.callbacks import Callback
import os

class ModelSave(Callback):
    def __init__(self, output_model_path, prefix=''):
        self.output_model_path = output_model_path
        self.prefix = prefix
        super().__init__()

    def on_epoch_end(self, epoch, logs=None):
        os.makedirs(self.output_model_path, exist_ok=True)
        weight_file_name = '{0}_epoch_{1:02d}_model.hdf5'.format(self.prefix, epoch + 1)
        weight_file_path = os.path.join(self.output_model_path, weight_file_name)
        self.model.save(weight_file_path)

def prepare_simple_CNN_model(input_shape=(28, 28, 1), class_num=10):
    input = Input(input_shape)
    kernel_size = (3, 3)
    max_pool_size = (2, 2)
    
    cnn = Conv2D(64, kernel_size, padding='same', activation='relu')(input)
    cnn = Dropout(0.1)(cnn)
    cnn = Conv2D(64, kernel_size, padding='same', activation='relu')(cnn)
    cnn = Dropout(0.1)(cnn)
    cnn = Conv2D(64, kernel_size, padding='same', activation='relu')(cnn)
    cnn = Dropout(0.1)(cnn)
    cnn = MaxPooling2D(pool_size=max_pool_size, strides=(2, 2))(cnn)
    
    cnn = Conv2D(64, kernel_size, padding='same', activation='relu')(cnn)
    cnn = Dropout(0.1)(cnn)
    cnn = Conv2D(64, kernel_size, padding='same', activation='relu')(cnn)
    cnn = Dropout(0.1)(cnn)
    cnn = Conv2D(64, kernel_size, padding='same', activation='relu')(cnn)
    cnn = MaxPooling2D(pool_size=max_pool_size, strides=(2, 2))(cnn)

    fc = Flatten()(cnn)
    fc = Dense(1024, activation='relu')(fc)
    softmax = Dense(class_num, activation='softmax')(fc)
    model = Model(input=input, output=softmax)
    
    return model


if __name__ == '__main__':
    model = prepare_simple_CNN_model()
    plot_model(model, to_file='model_CNN.png', show_shapes=True, rankdir='TB')
