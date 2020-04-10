from math import sqrt
from numpy import prod
from ..backend import np
from ..engine.base_layer import Layer


class BatchNormalizationAll(Layer):
    def __init__(self, momentum=0.99, running_mean=None, running_var=None, **kwargs):
        super().__init__()
        self.gamma = None  # 1
        self.beta = None  # 0
        self.momentum = momentum
        self.momentum_decay = 1 - momentum
        self.input_shape = None

        self.running_mean = running_mean
        self.running_var = running_var

        self.batch_size = None
        self.xc = None
        self.std = None

    def build(self, input_shape, **kwargs):
        self.output_shape = input_shape

        D = (prod(input_shape), )

        if self.beta is None:
            self.beta = self.add_weight(D, mean=0, initializer='constant', trainable=True)
            self.gamma = self.add_weight(D, mean=1, initializer='constant', trainable=True)

        if self.running_mean is None:
            self.running_mean = self.add_weight(D, mean=0, initializer='constant', trainable=False)
            self.running_var = self.add_weight(D, mean=1, initializer='constant', trainable=False)

    def compute_output_shape(self):
        return self.output_shape

    def call(self, x, training=True, **kwargs):
        self.input_shape = x.shape
        if x.ndim != 2:
            N, C, H, W = x.shape
            x = x.reshape(N, -1)

        out = self.__forward(x, training)
        out = out.reshape(*self.input_shape)

        x = out[0].get()

        return out

    def __forward(self, x, train_flg):

        if train_flg:
            mu = x.mean(axis=0)
            xc = x - mu
            var = np.mean(xc**2, axis=0)
            std = np.sqrt(var + 10e-7)
            xn = xc / std

            self.batch_size = x.shape[0]
            self.xc = xc
            self.xn = xn
            self.std = std
            self.running_mean.weight = self.running_mean.weight - (self.running_mean.weight - mu) * self.momentum_decay
            self.running_var.weight = self.running_var.weight - (self.running_var.weight - var) * self.momentum_decay
        else:
            xc = x - self.running_mean.weight
            xn = xc / ((np.sqrt(self.running_var.weight + 10e-7)))

        out = self.gamma.weight * xn + self.beta.weight
        return out

    def backward(self, dout):
        if dout.ndim != 2:
            N, C, H, W = dout.shape
            dout = dout.reshape(N, -1)

        dx = self.__backward(dout)

        dx = dx.reshape(*self.input_shape)
        return dx

    def __backward(self, dout):
        dbeta = dout.sum(axis=0)
        dgamma = np.sum(self.xn * dout, axis=0)

        dxn = self.gamma.weight * dout
        dxc = dxn / self.std + \
            self.xc / self.batch_size * - np.sum((dxn * self.xc), axis=0) / (self.std * self.std * self.std)
        dx = dxc - np.sum(dxc, axis=0) / self.batch_size

        self.gamma.gradient = dgamma
        self.beta.gradient = dbeta

        return dx
