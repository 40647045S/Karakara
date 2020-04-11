import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

from karakara import config
config.GPU = True
import karakara.backend as K
# K.set_floatx('float64')
# K.set_epsilon(1e-4)

from karakara.models import Sequential
from karakara.layers import Dense, Dropout, Input
from karakara.activations import Sigmoid, LeakyReLU, Softmax
from karakara.optimizers import SGD, Momentum, Adam, RMSprop

from utils import plot_history, make_mnist_data

input_shape = (784, )
n_classes = 10
epochs = 2
batch_size = 128


def make_model():
    model = Sequential()
    model.add(Input(shape=input_shape))
    model.add(Dense(1024))
    model.add(LeakyReLU(0.2))
    model.add(Dense(512))
    model.add(LeakyReLU(0.2))
    model.add(Dense(256))
    model.add(LeakyReLU(0.2))
    model.add(Dense(10))
    model.add(Softmax())

    return model


def main():
    (X_train, y_train), (X_valid, y_valid), (X_test,
                                             y_test) = make_mnist_data()
    print(X_train.shape)
    print(X_test.shape)

    from itertools import product
    from collections import OrderedDict
    import matplotlib.pyplot as plt

    opts = [Momentum, SGD, RMSprop, Adam]
    lrs = [0.01, 0.001, 0.0001]
    for lr in lrs:
        historys = OrderedDict()
        for opt in opts:
            model = make_model()
            model.compile(opt(), 'categorical_crossentropy', 'accuracy')

            history = model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs,
                                validation_data=(X_valid, y_valid))

            historys[opt.__name__] = history

        plt.figure(figsize=(20, 10))
        plt.subplot(1, 2, 1)
        for history in historys.values():
            plt.plot(history['acc'])
        plt.title('Model accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(historys.keys(), loc='upper left')
        plt.show()

        plt.subplot(1, 2, 2)
        for history in historys.values():
            plt.plot(history['loss'])
        plt.title('Model loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(historys.keys(), loc='upper left')
        plt.show()

        plt.savefig(f'History_{lr}.jpg')
        plt.close()

    test_loss, test_acc = model.evaluate(X_test, y_test)
    print()
    print(f'Test loss: {test_loss}, test acc: {test_acc}')


if __name__ == '__main__':
    main()
