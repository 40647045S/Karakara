import numpy as np

import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import mnist, fashion_mnist, cifar10
from sklearn.model_selection import train_test_split

keras.backend.set_image_data_format('channels_first')

n_classes = 10


def make_mnist_data(valid_ratio=0.2, reshape=True):
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    if reshape:
        X_train = X_train.reshape(60000, 784)
        X_test = X_test.reshape(10000, 784)
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train /= 255
    X_test /= 255

    y_train = to_categorical(y_train, n_classes)
    y_test = to_categorical(y_test, n_classes)

    if valid_ratio > 0:
        X_train, X_valid, y_train, y_valid = train_test_split(
            X_train, y_train, test_size=valid_ratio, random_state=42)
    else:
        X_valid, y_valid = np.array([]), np.array([])

    print(X_train.shape[0], 'train samples')
    print(X_valid.shape[0], 'vaild samples')
    print(X_test.shape[0], 'test samples')

    return (X_train, y_train), (X_valid, y_valid), (X_test, y_test)


def make_fasion_mnist_data(valid_ratio=0.2):
    (X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()

    X_train = X_train
    X_test = X_test
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train /= 255
    X_test /= 255

    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)

    if valid_ratio > 0:
        X_train, X_valid, y_train, y_valid = train_test_split(
            X_train, y_train, test_size=valid_ratio, random_state=42)
    else:
        X_valid, y_valid = np.array([]), np.array([])

    print(X_train.shape[0], 'train samples')
    print(X_valid.shape[0], 'vaild samples')
    print(X_test.shape[0], 'test samples')

    return (X_train, y_train), (X_valid, y_valid), (X_test, y_test)


def make_cifar10_data(valid_ratio=0.2, image_data_format='channels_first'):
    keras.backend.set_image_data_format(image_data_format)

    (X_train, y_train), (X_test, y_test) = cifar10.load_data()

    X_train = X_train
    X_test = X_test
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train /= 255
    X_test /= 255

    y_train = to_categorical(y_train, n_classes)
    y_test = to_categorical(y_test, n_classes)

    if valid_ratio > 0:
        X_train, X_valid, y_train, y_valid = train_test_split(
            X_train, y_train, test_size=valid_ratio, random_state=42)
    else:
        X_valid, y_valid = np.array([]), np.array([])

    print(X_train.shape[0], 'train samples')
    print(X_valid.shape[0], 'vaild samples')
    print(X_test.shape[0], 'test samples')

    return (X_train, y_train), (X_valid, y_valid), (X_test, y_test)


def plot_history(history, filename):

    # plt.figure(figsize=2)
    # for i in range(generatedImages.shape[0]):
    #     plt.subplot(dim[0], dim[1], i + 1)
    #     plt.imshow(generatedImages[i], interpolation='nearest', cmap='gray_r')
    #     plt.axis('off')
    # plt.tight_layout()
    # plt.savefig(f'gan_images/gan_fasion_mnist_epoch_{epoch}_karakara.png')
    # plt.close()

    plt.figure(figsize=(20, 10))

    plt.subplot(1, 2, 1)
    plt.plot(history['acc'])
    plt.plot(history['val_acc'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()

    plt.subplot(1, 2, 2)
    plt.plot(history['loss'])
    plt.plot(history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()

    plt.savefig(filename)
    plt.close()
