import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torchvision
import numpy as np

from karakara import config
config.GPU = True
import karakara.backend as K
# K.set_floatx('float16')
# K.set_epsilon(1e-4)

from karakara.models import Sequential
from karakara.layers import Dense, Dropout, BatchNormalization_v2
from karakara.layers import Input, Add, Separate, Same, Flatten, Activation
from karakara.layers import Conv2D, MaxPooling2D, AveragePooling2DAll
from karakara.activations import Sigmoid, ReLU, LeakyReLU, Softmax
from karakara.optimizers import SGD, Momentum, Adam
from karakara.regulizers import l2
from karakara.callbacks import ReduceLROnPlateau

from utils import plot_history

input_shape = (3, 32, 32)
n_classes = 10
epochs = 200
batch_size = 32
l2_lambda = 1e-4
depth = 56


def add_resnet_layer(model, num_filters=16, kernel_size=3, strides=1,
                     activation='relu', batch_normalization=True, conv_first=True):

    conv = Conv2D(num_filters, kernel_size, strides=strides, padding='same', kernel_regularizer=l2(l2_lambda))

    if conv_first:
        model.add(conv)
        if batch_normalization:
            model.add(BatchNormalization_v2())
        if activation is not None:
            model.add(Activation(activation))

    else:
        if batch_normalization:
            model.add(BatchNormalization_v2())
        if activation is not None:
            model.add(Activation(activation))
        model.add(conv)

    return model


def resnet_v2(input_shape, depth, num_classes):

    if (depth - 2) % 9 != 0:
        raise ValueError('depth should be 9n+2 (eg 56 or 110 in [b])')

    num_filters_in = 16
    num_res_blocks = int((depth - 2) / 9)

    model = Sequential()
    model.add(Input(shape=input_shape))

    add_resnet_layer(model, num_filters=num_filters_in, conv_first=True)

    for stage in range(3):
        for res_block in range(num_res_blocks):
            activation = 'relu'
            batch_normalization = True
            strides = 1
            if stage == 0:
                num_filters_out = num_filters_in * 4
                if res_block == 0:
                    activation = None
                    batch_normalization = False
            else:
                num_filters_out = num_filters_in * 2
                if res_block == 0:
                    strides = 2

            main_route = Sequential()
            add_resnet_layer(main_route, num_filters=num_filters_in, kernel_size=1, strides=strides,
                             activation=activation, batch_normalization=batch_normalization, conv_first=False)
            add_resnet_layer(main_route, num_filters=num_filters_in, conv_first=False)
            add_resnet_layer(main_route, num_filters=num_filters_out, kernel_size=1, conv_first=False)

            if res_block == 0:
                short_cut = Sequential()
                add_resnet_layer(short_cut, num_filters=num_filters_out, kernel_size=1,
                                 strides=strides, activation=None, batch_normalization=False)
            else:
                short_cut = Same()

            model.add(Separate())
            model.add([short_cut, main_route])
            model.add(Add())

        num_filters_in = num_filters_out

    model.add(BatchNormalization_v2())
    model.add(Activation('relu'))
    model.add(AveragePooling2DAll())
    model.add(Flatten())
    model.add(Dense(10, kernel_initializer='He'))
    model.add(Softmax())

    return model


def main():
    transform_train = torchvision.transforms.Compose([
        torchvision.transforms.RandomCrop(32, padding=4, padding_mode='edge'),
        torchvision.transforms.RandomHorizontalFlip(),
        # torchvision.transforms.RandomRotation(15),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    training_data = torchvision.datasets.CIFAR10('./data/cifar10', train=True, transform=transform_train, download=True)
    testing_data = torchvision.datasets.CIFAR10('./data/cifar10', train=False, transform=transform_test, download=True)

    model = resnet_v2(input_shape, depth, n_classes)
    model.summary()
    model.compile(Adam(lr=0.001), 'categorical_crossentropy', 'accuracy')

    def lr_schduler(model):
        lr = 1e-3
        if model.n_epoch > 180:
            lr *= 0.5e-3
        elif model.n_epoch > 160:
            lr *= 1e-3
        elif model.n_epoch > 120:
            lr *= 1e-2
        elif model.n_epoch > 80:
            lr *= 1e-1
        model.optimizer.lr = lr
        print('Learning rate: ', lr)
        return lr

    lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1),
                                   cooldown=0,
                                   patience=6,
                                   min_lr=0.5e-6)

    history = model.fit_torchvision(training_data, batch_size=batch_size, epochs=epochs,
                                    validation_data=testing_data, callbacks=[lr_schduler])

    plot_history(history, 'cifar10_resnet56_v2.jpg')

    model.save('resnet56_v2.h8')


if __name__ == '__main__':
    main()
