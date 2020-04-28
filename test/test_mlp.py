import os
import sys
sys.path.insert(0, os.path.dirname(
    os.path.dirname(os.path.realpath(__file__))))

from karakara import config
config.GPU = True
import karakara.backend as K
# K.set_floatx('float64')
# K.set_epsilon(1e-4)

from karakara.models import Sequential
from karakara.layers import Dense, Dropout, Input
from karakara.activations import Sigmoid, LeakyReLU, Softmax, ReLU
from karakara.optimizers import SGD, Momentum, Adam, RMSprop

from utils import plot_history, make_mnist_data

input_shape = (784, )
n_classes = 10
epochs = 2
batch_size = 128


def make_model():
    model = Sequential()
    model.add(Input(shape=input_shape))
    model.add(Dense(4096))
    model.add(ReLU())
    model.add(Dense(4096))
    model.add(ReLU())
    model.add(Dense(4096))
    model.add(ReLU())
    model.add(Dense(4096))
    model.add(ReLU())
    model.add(Dense(4096))
    model.add(ReLU())
    model.add(Dense(10))
    model.add(Softmax())

    model.summary()
    model.compile(Adam(), 'categorical_crossentropy', 'accuracy')

    return model


def main():
    (X_train, y_train), (X_valid, y_valid), (X_test,
                                             y_test) = make_mnist_data()
    print(X_train.shape)
    print(X_test.shape)

    model = make_model()

    history = model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs,
                        validation_data=(X_valid, y_valid))

    test_loss, test_acc = model.evaluate(X_test, y_test)
    print()
    print(f'Test loss: {test_loss}, test acc: {test_acc}')


if __name__ == '__main__':
    main()
