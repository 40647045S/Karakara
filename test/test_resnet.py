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
from karakara.layers import Dense, Dropout, Add, Separate, Same
from karakara.layers import Flatten, Conv2D, MaxPooling2D
from karakara.activations import Sigmoid, ReLU, LeakyReLU, Softmax
from karakara.optimizers import SGD, Momentom, Adam

from karakara.utils.conv_utils import im2col, col2im
from karakara.backend import np

from utils import make_mnist_data, plot_history, make_fasion_mnist_data, make_cifar10_data

input_shape = (3, 32, 32)
n_classes = 10
epochs = 2
batch_size = 64


def make_model():
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3), stride=1,
                     pad='same', input_shape=input_shape))
    model.add(ReLU())
    model.add(Dropout(0.25))
    model.add(MaxPooling2D(2, 2, stride=2))

    model.add(Conv2D(64, kernel_size=(3, 3), stride=1, pad='same'))
    model.add(ReLU())
    model.add(Dropout(0.25))
    model.add(MaxPooling2D(2, 2, stride=2))

    model.add(Flatten())
    model.add(Dropout(0.25))
    model.add(Dense(1024, kernel_initializer='He'))
    model.add(ReLU())
    model.add(Dropout(0.25))
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