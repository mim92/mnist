from keras.optimizers import SGD
from keras.losses import categorical_crossentropy, mean_squared_error
from CNN_Autoencoder.model import prepare_model, ModelSave
from CNN_Autoencoder.data_gen import prepare_mnist_data
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='train for detection')
    parser.add_argument('--load_model_path', type=str, default='')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--output_dir_path', type=str, default='model/')

    args = parser.parse_args()

    model = prepare_model()
    (x_train, y_train), (x_test, y_test) = prepare_mnist_data()
    callback = ModelSave(args.output_dir_path, 'MNIST_test')

    model.compile(loss={'classification': categorical_crossentropy, 'autoencoder': mean_squared_error},
                  loss_weights={'classification': 0.9, 'autoencoder': 0.1},
                  optimizer=SGD(lr=0.01, momentum=0.9, nesterov=True))

    model.fit(x_train, y_train, epochs=args.epochs, batch_size=args.batch_size, callbacks=[callback])
