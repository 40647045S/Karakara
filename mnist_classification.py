from karakara import config
# config.GPU = True
import karakara.backend as K
# K.set_floatx('float16')
# K.set_epsilon(1e-4)

from karakara.models import Sequential
from karakara.layers import Dense, Dropout
from karakara.activations import Sigmoid, LeakyReLU, Softmax
from karakara.optimizers import SGD, Momentom, Adam

from utils import make_mnist_data, plot_history, make_fasion_mnist_data, make_cifar10_data

input_shape = 3072
n_classes = 10
epochs = 50
batch_size = 128


def make_model():
    model = Sequential()
    model.add(Dense(1024, input_shape=input_shape))
    model.add(LeakyReLU(0.2))
    model.add(Dropout(0.3))
    model.add(Dense(512))
    model.add(LeakyReLU(0.2))
    model.add(Dropout(0.3))
    model.add(Dense(256))
    model.add(LeakyReLU(0.2))
    model.add(Dropout(0.3))
    model.add(Dense(128))
    model.add(LeakyReLU(0.2))
    model.add(Dropout(0.3))
    model.add(Dense(10))
    model.add(Softmax())

    model.summary()
    model.compile(Momentom(), 'categorical_crossentropy', 'accuracy')

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
