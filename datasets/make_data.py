import numpy as np
from tensorflow import keras
keras.backend.set_image_data_format('channels_first')
from tensorflow.keras.datasets import mnist, fashion_mnist, cifar10

datasets = {
    'mnist': mnist,
    'fashion_mnist': fashion_mnist,
    'cifar10': cifar10,
}


def main():
    for name, dataset in datasets.items():
        (X_train, y_train), (X_test, y_test) = dataset.load_data()
        np.savez(f'{name}.npz', X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)


if __name__ == '__main__':
    main()
