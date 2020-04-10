from ..backend import np
from ..engine.base_layer import Layer


class BatchNormalization(Layer):
    def __init__(self, axis=1, momentum=0.99, epison=0.0, **kwargs):
        super().__init__()
        self.axis = 1
        self.gamma = None  # 1
        self.beta = None  # 0
        self.momentum = momentum
        self.momentum_decay = 1 - momentum
        self.input_shape = None
        self.epison = epison

        self.running_mean = None
        self.running_var = None

        self.batch_size = None
        self.xc = None
        self.std = None

    def build(self, input_shape, **kwargs):
        self.output_shape = input_shape

        t = ((0, ) + input_shape)[self.axis]
        D = [1] * (len(input_shape) + 1)
        D[self.axis] = t

        if self.beta is None:
            self.beta = self.add_weight(D, mean=0, initializer='constant', trainable=True)
            self.gamma = self.add_weight(D, mean=1, initializer='constant', trainable=True)

        if self.running_mean is None:
            self.running_mean = self.add_weight(D, mean=0, initializer='constant', trainable=False)
            self.running_var = self.add_weight(D, mean=1, initializer='constant', trainable=False)

    def compute_output_shape(self):
        return self.output_shape

    def call(self, inputs, training=True, **kwargs):
        self.input_shape = inputs.shape

        out = self.__forward(inputs, training)

        return out.reshape(*self.input_shape)

    def __forward(self, inputs, train_flg):

        x = inputs
        if train_flg:
            axises = list(range(x.ndim))
            axises.remove(self.axis)
            axises = tuple(axises)
            mu = x.mean(axis=axises, keepdims=True)
            xc = x - mu
            var = np.var(x, axis=axises, keepdims=True)
            std = np.sqrt(var + self.epison)

            xn = (inputs - mu) / std

            self.batch_size = x.shape[0]
            self.xc = xc
            self.xn = xn
            self.std = std
            self.running_mean.weight = self.running_mean.weight - (self.running_mean.weight - mu) * self.momentum_decay
            self.running_var.weight = self.running_var.weight - (self.running_var.weight - var) * self.momentum_decay
        else:
            xc = x - self.running_mean.weight
            xn = xc / ((np.sqrt(self.running_var.weight + self.epison)))

        out = self.gamma.weight * xn + self.beta.weight
        return out

    def backward(self, dout):

        dx = self.__backward(dout)

        dx = dx.reshape(*self.input_shape)
        return dx

    def __backward(self, dout):
        axises = list(range(dout.ndim))
        axises.remove(self.axis)
        dbeta = dout.sum(axis=axises, keepdims=True)
        dgamma = np.sum(self.xn * dout, axis=axises, keepdims=True)

        dxn = self.gamma.weight * dout
        dxc = dxn / self.std + \
            self.xc / self.batch_size * - np.sum((dxn * self.xc), axis=0) / (self.std * self.std * self.std)
        dx = dxc - np.sum(dxc, axis=0) / self.batch_size

        self.gamma.gradient = dgamma
        self.beta.gradient = dbeta

        return dx


class BatchNormalization_v2(Layer):
    def __init__(self, axis=1, momentum=0.99, epison=0.0, **kwargs):
        super().__init__()
        self.axis = axis
        self.gamma = None  # 1
        self.beta = None  # 0
        self.momentum = momentum
        self.momentum_decay = 1 - momentum
        self.input_shape = None
        self.epison = epison

        self.running_mean = None
        self.running_var = None

        self.batch_size = None
        self.xc = None
        self.std = None

    def build(self, input_shape, **kwargs):
        self.output_shape = input_shape

        D = (input_shape[self.axis - 1], )

        if self.beta is None:
            self.beta = self.add_weight(D, mean=0, initializer='constant', trainable=True)
            self.gamma = self.add_weight(D, mean=1, initializer='constant', trainable=True)

        if self.running_mean is None:
            self.running_mean = self.add_weight(D, mean=0, initializer='constant', trainable=False)
            self.running_var = self.add_weight(D, mean=1, initializer='constant', trainable=False)

    def compute_output_shape(self):
        return self.output_shape

    @profile
    def call(self, inputs, training=True, **kwargs):
        self.origin_input_shape = inputs.shape

        N = inputs.shape[self.axis]
        inputs = np.moveaxis(inputs, self.axis, -1)
        self.moved_input_shape = inputs.shape

        inputs = inputs.reshape(-1, N)

        out = self.__forward(inputs, training)
        out = out.reshape(*self.moved_input_shape)
        out = np.moveaxis(out, -1, self.axis)

        return out

    def __forward(self, x, train_flg):

        if train_flg:
            mu = x.mean(axis=0)
            xc = x - mu
            var = np.var(x, axis=0)
            std = np.sqrt(var + self.epison)
            xn = xc / std

            self.batch_size = x.shape[0]
            self.xc = xc
            self.xn = xn
            self.std = std
            self.running_mean.weight = self.running_mean.weight - (self.running_mean.weight - mu) * self.momentum_decay
            self.running_var.weight = self.running_var.weight - (self.running_var.weight - var) * self.momentum_decay
        else:
            xc = x - self.running_mean.weight
            xn = xc / ((np.sqrt(self.running_var.weight + self.epison)))

        out = self.gamma.weight * xn + self.beta.weight
        return out

    def backward(self, dout):

        N = dout.shape[self.axis]
        dout = np.moveaxis(dout, self.axis, -1)
        dout = dout.reshape(-1, N)

        dx = self.__backward(dout)

        dx = dx.reshape(*self.moved_input_shape)
        dx = np.moveaxis(dx, -1, self.axis)

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
