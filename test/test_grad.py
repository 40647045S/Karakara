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
from karakara.layers import Dense, Dropout, Conv2D, Flatten
from karakara.activations import Sigmoid, LeakyReLU, Softmax
from karakara.optimizers import SGD, Momentum, Adam, RMSprop

from utils import plot_history, make_mnist_data

from karakara.backend import np, setup_data
np.random.seed(16)

input_shape = (1, 28, 28)
n_classes = 10
epochs = 5
batch_size = 128


def make_model():
    model = Sequential()
    model.add(Conv2D(10, 3, input_shape=input_shape))
    model.add(Flatten())
    model.add(Dense(1024))
    model.add(LeakyReLU(0.2))
    model.add(Dense(512))
    model.add(LeakyReLU(0.2))
    model.add(Dense(256))
    model.add(LeakyReLU(0.2))
    model.add(Dense(128))
    model.add(LeakyReLU(0.2))
    model.add(Dense(10))
    model.add(Softmax())

    # model.summary()
    model.compile(Adam(), 'categorical_crossentropy', 'accuracy')

    return model


def main():
    (X_train, y_train), (X_valid, y_valid), (X_test,
                                             y_test) = make_mnist_data(reshape=False)
    X_train = X_train.reshape(-1, 1, 28, 28)
    print(X_train.shape)
    print(X_test.shape)

    model = make_model()

    print(np.mean(X_train[0]))
    print(y_train[0])

    X = setup_data(np.array([X_train[0].copy() for _ in range(10)]))
    y = setup_data(np.array([y_train[0].copy() for _ in range(10)]))

    for i in range(1, 200):
        X = setup_data(np.array([X_train[0].copy() for _ in range(i)]))
        y = setup_data(np.array([y_train[0].copy() for _ in range(i)]))
        model.forward(X, y, training=True)
        model.cal_gradient()
        print(np.max(model.layers[0][0].kernel.gradient))


if __name__ == '__main__':
    main()
