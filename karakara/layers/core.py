from math import sqrt
from numpy import prod
from ..backend import np
from ..engine.base_layer import Layer
from ..utils.math_utils import cal_init_std


class Input(Layer):
    def __init__(self, shape, **kwargs):
        super().__init__(**kwargs)
        self.input_shape = shape

    def build(self, input_shape, **kwargs):
        return prod(self.input_shape) / 10

    def compute_output_shape(self):
        return self.input_shape

    def call(self, inputs, **kwargs):
        return inputs

    def backward(self, dout):
        return dout


class Dense(Layer):

    def __init__(self, units, input_shape=None, kernel_initializer='Xavier', **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.input_shape = None
        if input_shape:
            self.input_shape = input_shape if isinstance(
                input_shape, (tuple, list)) else (input_shape, )

        self.kernel_initializer = kernel_initializer

    def build(self, input_shape, pre_node_nums, **kwargs):
        if not self.built:
            if not input_shape:
                input_shape = self.input_shape

            if self.input_shape:
                assert self.input_shape == input_shape

            pre_node_nums = prod(input_shape)
            weight_std = cal_init_std(self.kernel_initializer, pre_node_nums)

            # assert input_shape
            self.kernel = self.add_weight(
                shape=(input_shape[-1], self.units), std=weight_std)
            self.bias = self.add_weight(
                shape=(self.units, ), mean=0, initializer='constant')

            self.output_shape = (input_shape[:-1]) + (self.units, )
            self.built = True

            return self.units

    def compute_output_shape(self):
        return self.output_shape

    @profile
    def call(self, inputs, **kwargs):
        self.x = inputs
        output = np.dot(inputs, self.kernel.weight) + self.bias.weight
        return output

    @profile
    def backward(self, dout):
        dx = np.dot(dout, self.kernel.weight.T)
        self.kernel.gradient = np.dot(self.x.T, dout)
        self.bias.gradient = np.sum(dout, axis=0)

        return dx


class Dropout(Layer):
    def __init__(self, dropout_ratio=0.5):
        super().__init__()
        self.dropout_ratio = dropout_ratio
        self.mask = None

    def build(self, input_shape, **kwargs):
        self.output_shape = input_shape

    def compute_output_shape(self):
        return self.output_shape

    def call(self, inputs, training=True, **kwargs):
        if training:
            self.mask = np.random.rand(*inputs.shape) > self.dropout_ratio
            return inputs * self.mask
        else:
            return inputs * (1.0 - self.dropout_ratio)

    def backward(self, dout):
        return dout * self.mask


class BatchNormalization(Layer):
    def __init__(self, axis=1, momentum=0.99, running_mean=None, running_var=None, **kwargs):
        super().__init__()
        self.axis = 1
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

    @profile
    def call(self, inputs, training=True, **kwargs):
        self.input_shape = inputs.shape
        # if x.ndim != 2:
        #     N, C, H, W = x.shape
        #     x = inputs.mean(axis=self.axis)
        # x = x.reshape(N, -1)

        out = self.__forward(inputs, training)

        return out.reshape(*self.input_shape)

    @profile
    def __forward(self, inputs, train_flg):

        x = inputs
        if train_flg:
            axises = list(range(x.ndim))
            axises.remove(self.axis)
            axises = tuple(axises)
            mu = x.mean(axis=axises, keepdims=True)
            xc = x - mu
            var = np.var(x, axis=axises, keepdims=True)
            std = np.sqrt(var + 10e-7)

            xn = (inputs - mu) / std

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

    @profile
    def backward(self, dout):
        # if dout.ndim != 2:
        #     N, C, H, W = dout.shape
        #     dout = dout.reshape(N, -1)

        dx = self.__backward(dout)

        dx = dx.reshape(*self.input_shape)
        return dx

    @profile
    def __backward(self, dout):
        axises = list(range(dout.ndim))
        axises.remove(self.axis)
        dbeta = dout.sum(axis=axises, keepdims=True)
        dgamma = np.sum(self.xn * dout, axis=axises, keepdims=True)

        # dxn = self.gamma.weight * dout
        # dxc = dxn / self.std
        # dstd = -np.sum((dxn * self.xc) / (self.std * self.std), axis=0)
        # dvar = 0.5 * dstd / self.std
        # dxc += (2.0 / self.batch_size) * self.xc * dvar
        # dmu = np.sum(dxc, axis=0)
        # dx = dxc - dmu / self.batch_size

        dxn = self.gamma.weight * dout
        dxc = dxn / self.std + \
            self.xc / self.batch_size * - np.sum((dxn * self.xc), axis=0) / (self.std * self.std * self.std)
        dx = dxc - np.sum(dxc, axis=0) / self.batch_size

        self.gamma.gradient = dgamma
        self.beta.gradient = dbeta

        return dx


class Same(Layer):
    def __init__(self):
        super().__init__()

    def build(self, input_shape, **kwargs):
        self.output_shape = input_shape

    def compute_output_shape(self):
        return self.output_shape

    def call(self, inputs, **kwargs):
        return inputs

    def backward(self, dout):
        return dout


class Separate(Layer):
    def __init__(self):
        super().__init__()

    def build(self, input_shape, **kwargs):
        self.output_shape = input_shape

    def compute_output_shape(self):
        return self.output_shape

    def call(self, inputs, **kwargs):
        return inputs

    def backward(self, dout):
        return np.sum(np.array(dout), axis=0)


class Add(Layer):
    def __init__(self):
        super().__init__()
        self.n_input = None

    def build(self, input_shape, **kwargs):
        for shape in input_shape:
            if not shape == input_shape[0]:
                raise ValueError(
                    f'All input should have same shape but get {input_shape}')
        self.n_input = len(input_shape)
        self.output_shape = input_shape[0]

    def compute_output_shape(self):
        return self.output_shape

    def call(self, inputs, **kwargs):
        return np.sum(np.array(inputs), axis=0)

    def backward(self, dout):
        return dout


class Flatten(Layer):
    def __init__(self):
        super().__init__()
        self.input_shape = None

    def build(self, input_shape, **kwargs):
        self.output_shape = (prod(input_shape), )

    def compute_output_shape(self):
        return self.output_shape

    def call(self, inputs, **kwargs):
        batch_size = inputs.shape[0]
        self.input_shape = inputs.shape
        return inputs.reshape(batch_size, -1)

    def backward(self, dout):
        return dout.reshape(self.input_shape)
