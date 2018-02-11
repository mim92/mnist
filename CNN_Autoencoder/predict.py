from keras.models import load_model
from CNN_Autoencoder.data_gen import prepare_mnist_data
import numpy as np
import os
import cv2


def predict_accuracy(x_batch, y_batch, model, output_dir):
    preds = model.predict(x_batch, verbose=0)
    batch_size = len(x_batch)
    count = 0
    for (y, pred) in zip(y_batch[0], preds[0]):
        if np.argmax(y) == np.argmax(pred):
            count += 1
    print('acc ', (count/batch_size))

    os.makedirs(output_dir, exist_ok=True)
    for (i, (y, pred)) in enumerate(zip(y_batch[1], preds[1])):
        org_img = np.array((255 - y) * 255, dtype=np.uint8)
        pred_img = np.array((255 - pred) * 255, dtype=np.uint8)

        output_img = cv2.hconcat([org_img, pred_img])
        output_name = output_dir + str(i).zfill(2) + '.png'
        cv2.imwrite(output_name, output_img)


if __name__ == '__main__':
    model_path = 'model/MNIST_test_epoch_20_model.hdf5'
    model = load_model(model_path)
    output_dir = 'output_image/'

    (_, _), (x_test, y_test) = prepare_mnist_data()

    predict_accuracy(x_test, y_test, model, output_dir)


