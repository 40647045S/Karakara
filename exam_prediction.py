import numpy as np
from itertools import product

from karakara.models import Sequential
from karakara.layers import Dense
from karakara.activations import Sigmoid, Tanh, ReLU, LeakyReLU
from karakara.optimizers import SGD
from karakara.metrics import MSE, LogLoss, Accuracy

n_trains = 10
input_shape = 8
epochs = 1000
batch_size = 10


def make_exam_data(filename):
    with open(filename) as file:
        lines = file.readlines()
    data = np.array([[float(x) for x in line.split()] for line in lines])
    data /= 100

    X = data[:, :-1]
    y = data[:, -1].reshape(-1, 1)

    return X, y


def make_model(hidden_size, activation, lr=0.1):
    model = Sequential()
    model.add(Dense(hidden_size, input_shape=8))
    model.add(activation())
    model.add(Dense(1))

    model.compile(loss='MSE', optimizer=SGD(lr=lr), metric='mse')

    return model


def main():

    X, y = make_exam_data('Data.txt')
    X_train, X_valid = X[:10], X[10:]
    y_train, y_valid = y[:10], y[10:]

    hidden_sizes = [1, 2, 3]
    activations = [Sigmoid, Tanh, ReLU, LeakyReLU]
    lrs = [0.1, 0.01, 0.001]

    history = {}
    for hidden_size, activation, lr in product(hidden_sizes, activations, lrs):
        temp = [[], []]
        for _ in range(n_trains):
            model = make_model(hidden_size, activation, lr=lr)

            model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs,
                      validation_data=(X_valid, y_valid), verbose=False)

            train_loss = model.evaluate(X_train, y_train, batch_size=10)[0]
            valid_loss = model.evaluate(X_valid, y_valid, batch_size=4)[0]

            temp[0].append(train_loss)
            temp[1].append(valid_loss)

        history[(hidden_size, activation.__name__, lr)] = np.mean(temp, axis=1) * 1000

    for h in history.items():
        print(h)


if __name__ == '__main__':
    main()
