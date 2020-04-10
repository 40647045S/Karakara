from ..backend import np
from ..engine.base_layer import Layer
from ..utils.norm_utils import update_mean_var, bn_backward


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
        self.x_centered = None
        self.std = None

    def build(self, input_shape, **kwargs):
        self.output_shape = input_shape

        t = ((0, ) + input_shape)[self.axis]
        D = [1] * (len(input_shape) + 1)
        D[self.axis] = t

        if self.beta is None:
            self.beta = self.add_weight(D, mean=0, initializer='constant', trainable=False)
            self.gamma = self.add_weight(D, mean=1, initializer='constant', trainable=False)

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
            var = np.mean(np.square(xc), axis=axises, keepdims=True)
            std = np.sqrt(var + self.epison)

            xn = (inputs - mu) / std

            self.batch_size = x.shape[0]
            self.x_centered = xc
            self.x_normed = xn
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
        dgamma = np.sum(self.x_normed * dout, axis=axises, keepdims=True)

        # dx_normed = self.gamma.weight * dout

        dx_normed = self.gamma.weight * dout
        # dx = 1 / self.batch_size / self.std * (self.batch_size * dx_normed -
        #                                        dx_normed.sum(axis=0) - self.x_normed * (dx_normed * self.x_normed).sum(axis=0))

        dxc = dx_normed / self.std + \
            self.x_centered / self.batch_size * - \
            np.sum((dx_normed * self.x_centered), axis=0) / (self.std * self.std * self.std)
        dx = dxc - np.sum(dxc, axis=0) / self.batch_size

        # if not np.all(np.abs(dx - dx2) < 0.01):
        #     print('dx1:', dx[0, 0])
        #     print('dx2:', dx2[0, 0])
        #     assert False

        self.gamma.gradient = dgamma
        self.beta.gradient = dbeta

        return dx


class BatchNormalization_v2(Layer):
    def __init__(self, axis=1, momentum=0.99, epison=1e-7, **kwargs):
        super().__init__()
        self.axis = axis
        self.gamma = None  # 1
        self.beta = None  # 0
        self.momentum = momentum
        self.momentum_decay = 1 - momentum
        self.input_shape = None
        self.epison = epison
        self.D = None

        self.running_mean = None
        self.running_var = None

    def build(self, input_shape, **kwargs):
        self.output_shape = input_shape

        self.D = input_shape[self.axis - 1]

        if self.beta is None:
            self.beta = self.add_weight((self.D, 1), mean=0, initializer='constant', trainable=True)
            self.gamma = self.add_weight((self.D, 1), mean=1, initializer='constant', trainable=True)

        if self.running_mean is None:
            self.running_mean = self.add_weight((self.D, 1), mean=0, initializer='constant', trainable=False)
            self.running_var = self.add_weight((self.D, 1), mean=1, initializer='constant', trainable=False)

    def compute_output_shape(self):
        return self.output_shape

    def call(self, inputs, training=True, **kwargs):
        self.origin_input_shape = inputs.shape

        inputs = np.moveaxis(inputs, self.axis, 0)
        self.moved_input_shape = inputs.shape
        inputs = inputs.reshape(self.D, -1)

        out = self.__forward(inputs, training)

        out = out.reshape(*self.moved_input_shape)
        out = np.moveaxis(out, 0, self.axis)
        return out

    def __forward(self, x, train_flg):

        if train_flg:
            mu = x.mean(axis=1, keepdims=True)
            xc = x - mu
            var = np.mean(np.square(xc), axis=1, keepdims=True)
            inv_std = np.reciprocal(np.sqrt(var + self.epison))
            xn = xc * inv_std

            # todo
            # unbiased estimation

            self.cahes = (x.shape[1], xc, xn, inv_std)
            update_mean_var(mu, var, self.momentum_decay, 0, self.running_mean.weight, self.running_var.weight)
        else:
            xc = x - self.running_mean.weight
            xn = xc / ((np.sqrt(self.running_var.weight + self.epison)))

        out = self.gamma.weight * xn + self.beta.weight
        return out

    def backward(self, dout):

        dout = np.moveaxis(dout, self.axis, 0)
        dout = dout.reshape(self.D, -1)

        dx = self.__backward(dout)

        dx = dx.reshape(*self.moved_input_shape)
        dx = np.moveaxis(dx, 0, self.axis)

        return dx

    def __backward(self, dout):

        N, _, xn, inv_std = self.cahes

        np.sum(dout, axis=1, keepdims=True, out=self.beta.gradient)
        np.sum(xn * dout, axis=1, keepdims=True, out=self.gamma.gradient)

        # dxn = self.gamma.weight * dout
        # dxc = (dxn - xn / N * np.sum((dxn * xn), axis=1, keepdims=True)) * inv_std
        # dx = dxc - np.sum(dxc, axis=1, keepdims=True) / N

        return bn_backward(self.gamma.weight, dout, xn, N, inv_std)
