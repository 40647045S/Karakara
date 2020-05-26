import numpy as np

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


def to_categorical(data, n_classes):
    return np.eye(n_classes)[data.reshape(-1)]


def load_data(path):
    data = np.load(path)
    return (data['X_train'], data['y_train']), (data['X_test'], data['y_test'])


def make_mnist_data(valid_ratio=0.2, reshape=True):
    (X_train, y_train), (X_test, y_test) = load_data('datasets/mnist.npz')

    if reshape:
        X_train = X_train.reshape(60000, 784)
        X_test = X_test.reshape(10000, 784)

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


def make_fasion_mnist_data(valid_ratio=0.2):
    (X_train, y_train), (X_test, y_test) = load_data('datasets/fashion_mnist.npz')

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


def make_cifar10_data(valid_ratio=0.2):

    (X_train, y_train), (X_test, y_test) = load_data('datasets/cifar10.npz')

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


def plot_history(history, filename):

    plt.figure(figsize=(16, 8))

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
