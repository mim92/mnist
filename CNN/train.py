from keras.optimizers import SGD
from keras.losses import categorical_crossentropy
from model import prepare_simple_CNN_model, ModelSave
from CNN_Autoencoder.data_gen import prepare_mnist_data
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='train for detection')
    parser.add_argument('--load_model_path', type=str, default='')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--output_dir_path', type=str, default='model/')

    args = parser.parse_args()

    model = prepare_simple_CNN_model()
    (x_train, y_train), (x_test, y_test) = prepare_mnist_data()
    callback = ModelSave(args.output_dir_path, 'MNIST_test')

    model.compile(loss=categorical_crossentropy,
                  optimizer=SGD(lr=0.01, momentum=0.9, nesterov=True))

    model.fit(x_train, y_train, epochs=args.epochs, batch_size=args.batch_size, callbacks=[callback])
