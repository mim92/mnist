from keras.models import Model
from keras.layers import Conv2D, MaxPooling2D, Dropout, Input, Dense, Flatten, UpSampling2D
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

def prepare_model(input_shape=(28, 28, 1), class_num=10):
    input = Input(input_shape)
    kernel_size = (3, 3)
    max_pool_size = (2, 2)
    upsampling_size = (2, 2)

    enc_cnn = Conv2D(64, kernel_size, padding='same', activation='relu')(input)
    enc_cnn = Dropout(0.1)(enc_cnn)
    enc_cnn = Conv2D(64, kernel_size, padding='same', activation='relu')(enc_cnn)
    enc_cnn = Dropout(0.1)(enc_cnn)
    enc_cnn = Conv2D(64, kernel_size, padding='same', activation='relu')(enc_cnn)
    enc_cnn = Dropout(0.1)(enc_cnn)
    enc_cnn = MaxPooling2D(pool_size=max_pool_size, strides=(2, 2))(enc_cnn)
    
    enc_cnn = Conv2D(64, kernel_size, padding='same', activation='relu')(enc_cnn)
    enc_cnn = Dropout(0.1)(enc_cnn)
    enc_cnn = Conv2D(64, kernel_size, padding='same', activation='relu')(enc_cnn)
    enc_cnn = Dropout(0.1)(enc_cnn)
    enc_cnn = Conv2D(64, kernel_size, padding='same', activation='relu')(enc_cnn)
    enc_cnn = MaxPooling2D(pool_size=max_pool_size, strides=(2, 2))(enc_cnn)

    fc = Flatten()(enc_cnn)
    fc = Dense(1024, activation='relu')(fc)
    softmax = Dense(class_num, activation='softmax', name='classification')(fc)

    dec_cnn = UpSampling2D(upsampling_size)(enc_cnn)
    dec_cnn = Conv2D(64, kernel_size, padding='same', activation='relu')(dec_cnn)
    dec_cnn = UpSampling2D(upsampling_size)(dec_cnn)
    dec_cnn = Conv2D(1, kernel_size, padding='same', activation='sigmoid', name='autoencoder')(dec_cnn)

    outputs = [softmax, dec_cnn]

    model = Model(input=input, output=outputs)
    return model


if __name__ == '__main__':
    model = prepare_model()
    plot_model(model, to_file='model.png', show_shapes=True, rankdir='TB')
