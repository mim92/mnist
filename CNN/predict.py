from keras.models import load_model
from CNN_Autoencoder.data_gen import prepare_mnist_data
import numpy as np


def predict_accuracy(x_batch, y_batch, model):
    preds = model.predict(x_batch, verbose=0)
    batch_size = len(x_batch)
    count = 0
    for (y, pred) in zip(y_batch, preds):
        if np.argmax(y) == np.argmax(pred):
            count += 1
    print('acc ', (count/batch_size))


if __name__ == '__main__':
    model_path = 'model/MNIST_test_epoch_02_model.hdf5'
    model = load_model(model_path)

    (_, _), (x_test, y_test) = prepare_mnist_data()

    predict_accuracy(x_test, y_test, model)


