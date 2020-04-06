import numpy as np

import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist, fashion_mnist, cifar10
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

n_classes = 10


def make_mnist_data(valid_ratio=0.2):
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

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


def make_cifar10_data(valid_ratio=0.2):
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()

    X_train = X_train.reshape(50000, 3072)
    X_test = X_test.reshape(10000, 3072)
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


def plot_history(history):
    plt.plot(history['metric'])
    plt.plot(history['val_metric'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()

    plt.plot(history['loss'])
    plt.plot(history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()
