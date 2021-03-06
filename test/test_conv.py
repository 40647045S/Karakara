import os
import sys
sys.path.insert(0, os.path.dirname(
    os.path.dirname(os.path.realpath(__file__))))

from karakara import config
config.GPU = True
import karakara.backend as K
# K.set_floatx('float16')
# K.set_epsilon(1e-4)

from karakara.models import Sequential
from karakara.layers import Dense, Dropout, Input
from karakara.layers import Flatten, Conv2D, MaxPooling2D, BatchNormalization_v2
from karakara.activations import Sigmoid, ReLU, LeakyReLU, Softmax
from karakara.optimizers import SGD, Momentum, Adam

from karakara.utils.conv_utils import im2col, col2im
from karakara.backend import np

from utils import make_mnist_data, plot_history, make_fasion_mnist_data, make_cifar10_data

input_shape = (3, 32, 32)
n_classes = 10
epochs = 2
batch_size = 32


def make_model():
    model = Sequential()
    model.add(Input(shape=input_shape))

    model.add(Conv2D(32, kernel_size=(3, 3), padding='same'))
    model.add(ReLU())
    model.add(Conv2D(32, kernel_size=(3, 3), padding='same'))
    model.add(ReLU())
    model.add(Conv2D(32, kernel_size=(3, 3), padding='same'))
    model.add(ReLU())
    model.add(Conv2D(32, kernel_size=(3, 3), padding='same'))
    model.add(ReLU())
    model.add(Conv2D(32, kernel_size=(3, 3), padding='same'))
    model.add(ReLU())

    model.add(Flatten())
    model.add(Dense(2500, kernel_initializer='He'))
    model.add(ReLU())
    model.add(Dense(1500, kernel_initializer='He'))
    model.add(ReLU())
    model.add(Dense(10, kernel_initializer='He'))
    model.add(Softmax())

    model.summary()
    model.compile(Adam(), 'categorical_crossentropy', 'accuracy')

    return model


def main():
    (X_train, y_train), (X_valid, y_valid), (X_test,
                                             y_test) = make_cifar10_data(0.2)
    print(f'X_train: {X_train.shape}')
    print(f'X_valid: {X_valid.shape}')
    print(f'X_test  : {X_test.shape}')

    model = make_model()

    history = model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs,
                        validation_data=(X_valid, y_valid))

    test_loss, test_acc = model.evaluate(X_test, y_test)
    print()
    print(f'Test loss: {test_loss}, test acc: {test_acc}')


if __name__ == '__main__':
    main()
