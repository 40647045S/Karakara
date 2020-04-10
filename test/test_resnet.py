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
from karakara.layers import Input, Dense, Dropout, Add, Separate, Same
from karakara.layers import Flatten, Conv2D, MaxPooling2D, BatchNormalization_v2
from karakara.activations import Sigmoid, ReLU, LeakyReLU, Softmax
from karakara.optimizers import SGD, Momentom, Adam
from karakara.regulizers import l2

from utils import plot_history, make_cifar10_data

input_shape = (3, 32, 32)
n_classes = 10
epochs = 50
batch_size = 64


def cnn_seq(num_filters=16, kernel_size=3, strides=1,
            activation=ReLU, batch_normalization=True, conv_first=True):
    seq = Sequential()
    seq.add(Conv2D(num_filters, kernel_size=kernel_size, stride=strides, padding='same'))
    if batch_normalization:
        seq.add(BatchNormalization_v2())
    seq.add(activation())
    seq.add(Conv2D(num_filters, kernel_size=kernel_size, stride=strides, padding='same'))
    if batch_normalization:
        seq.add(BatchNormalization_v2())
    seq.add(activation())

    return seq


def add_residual_block(model, num_filters=16, kernel_size=3, strides=1,
                       activation=ReLU, batch_normalization=True, conv_first=True):

    model.add(Separate())
    model.add([Same(), cnn_seq(num_filters, kernel_size, strides, activation, batch_normalization, conv_first)])
    model.add(Add())

    return model


def make_model():
    model = Sequential()
    model.add(Input(shape=input_shape))

    model.add(Conv2D(32, kernel_size=3, stride=1, padding='same'))
    model.add(ReLU())
    model.add(MaxPooling2D(2, 2, stride=2))

    add_residual_block(model, num_filters=32, batch_normalization=False)
    add_residual_block(model, num_filters=32, batch_normalization=False)

    model.add(Conv2D(64, kernel_size=3, stride=1, padding='same'))
    model.add(ReLU())
    model.add(MaxPooling2D(2, 2, stride=2))

    add_residual_block(model, 64, batch_normalization=False)
    add_residual_block(model, 64, batch_normalization=False)

    # model.add(Conv2D(64, kernel_size=3, stride=1, padding='same'))
    # model.add(ReLU())
    model.add(MaxPooling2D(2, 2, stride=2))

    model.add(Flatten())
    model.add(Dense(512, kernel_initializer='He'))
    model.add(ReLU())
    model.add(Dense(10, kernel_initializer='He'))
    model.add(Softmax())

    model.summary()
    model.compile(Adam(), 'categorical_crossentropy', 'accuracy')

    return model


def main():
    (X_train, y_train), (X_valid, y_valid), (X_test,
                                             y_test) = make_cifar10_data()
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
