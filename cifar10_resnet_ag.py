import torchvision

from karakara import config
config.GPU = True
import karakara.backend as K
# K.set_floatx('float16')
# K.set_epsilon(1e-4)

from karakara.models import Sequential
from karakara.layers import Dense, Dropout, BatchNormalization_v2
from karakara.layers import Input, Add, Separate, Same, Flatten
from karakara.layers import Conv2D, MaxPooling2D, AveragePooling2DAll
from karakara.activations import Sigmoid, ReLU, LeakyReLU, Softmax
from karakara.optimizers import SGD, Momentom, Adam
from karakara.regulizers import l2

from utils import plot_history

input_shape = (3, 32, 32)
n_classes = 10
epochs = 100
batch_size = 32
l2_lambda = 0


def cnn_seq(num_filters=16, kernel_size=3, strides=1,
            activation=ReLU, batch_normalization=True, conv_first=True):
    seq = Sequential()
    seq.add(Conv2D(num_filters, kernel_size=kernel_size, strides=strides, padding='same', kernel_regularizer=l2(l2_lambda)))
    if batch_normalization:
        seq.add(BatchNormalization_v2())
    seq.add(activation())
    seq.add(Conv2D(num_filters, kernel_size=kernel_size, strides=1, padding='same', kernel_regularizer=l2(l2_lambda)))
    if batch_normalization:
        seq.add(BatchNormalization_v2())

    return seq


def add_residual_block(model, num_filters=16, kernel_size=3, strides=1,
                       activation=ReLU, cnn_shortcut=False, batch_normalization=True, conv_first=True):

    if cnn_shortcut:
        shortcut = Conv2D(num_filters, kernel_size=1, strides=strides,
                          padding='same', kernel_regularizer=l2(l2_lambda))
    else:
        shortcut = Same()

    cnn_road = cnn_seq(num_filters, kernel_size, strides, activation, batch_normalization, conv_first)

    model.add(Separate())
    model.add([shortcut, cnn_road])
    model.add(Add())
    model.add(ReLU())

    return model


def make_model():
    model = Sequential()
    model.add(Input(shape=input_shape))

    model.add(Conv2D(16, kernel_size=3, strides=1, padding='same', kernel_regularizer=l2(l2_lambda)))
    model.add(BatchNormalization_v2())
    model.add(ReLU())

    add_residual_block(model, num_filters=16)
    add_residual_block(model, num_filters=16)
    add_residual_block(model, num_filters=16)

    add_residual_block(model, num_filters=32, strides=2, cnn_shortcut=True)
    add_residual_block(model, num_filters=32)
    add_residual_block(model, num_filters=32)

    add_residual_block(model, num_filters=64, strides=2, cnn_shortcut=True)
    add_residual_block(model, num_filters=64)
    add_residual_block(model, num_filters=64)
    model.add(AveragePooling2DAll())

    model.add(Flatten())
    model.add(Dense(10, kernel_initializer='He'))
    model.add(Softmax())

    model.summary()
    model.compile(Adam(lr=0.001, decay=1e-5), 'categorical_crossentropy', 'accuracy')

    return model


def main():
    transform_train = torchvision.transforms.Compose([
        torchvision.transforms.RandomCrop(32, padding=4),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.RandomRotation(15),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    training_data = torchvision.datasets.CIFAR10('./data/cifar10', train=True, transform=transform_train, download=True)
    testing_data = torchvision.datasets.CIFAR10('./data/cifar10', train=False, transform=transform_test, download=True)

    model = make_model()

    history = model.fit_torchvision(training_data, batch_size=batch_size, epochs=epochs,
                                    validation_data=testing_data)

    # test_loss, test_acc = model.evaluate(X_test, y_test)
    # print()
    # print(f'Test loss: {test_loss}, test acc: {test_acc}')

    plot_history(history, 'cifar10_resnet.jpg')


if __name__ == '__main__':
    main()
