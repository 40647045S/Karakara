from math import sqrt
from ..backend import np
from ..engine.base_layer import Layer


class Dense(Layer):

    def __init__(self, units, input_shape=None, kernel_initializer='Xavier', **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.input_shape = None
        if input_shape:
            self.input_shape = input_shape if isinstance(
                input_shape, (tuple, list)) else (input_shape, )

        if kernel_initializer == 'Xavier':
            self.weight_std = sqrt(1 / units)
        elif kernel_initializer == 'He':
            self.weight_std = sqrt(2 / units)
        elif isinstance(kernel_initializer, (int, float)):
            self.weight_std = kernel_initializer
        else:
            raise ValueError(f'{kernel_initializer} 是在哈樓？')

    def build(self, input_shape):
        if not self.built:
            if not input_shape:
                input_shape = self.input_shape

            if self.input_shape:
                assert self.input_shape == input_shape

            # assert input_shape

            self.kernel = self.add_weight(
                shape=(input_shape[-1], self.units), std=self.weight_std)
            self.bias = self.add_weight(shape=(self.units, ), std=0)

            self.output_shape = (input_shape[:-1]) + (self.units, )
            self.built = True

    def compute_output_shape(self):
        return self.output_shape

    def call(self, inputs, **kwargs):
        self.x = inputs
        output = np.dot(inputs, self.kernel.weight) + self.bias.weight
        return output

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

    def build(self, input_shape):
        self.output_shape = input_shape

    def compute_output_shape(self):
        return self.output_shape

    def call(self, x, training=True, **kwargs):
        if training:
            self.mask = np.random.rand(*x.shape) > self.dropout_ratio
            return x * self.mask
        else:
            return x * (1.0 - self.dropout_ratio)

    def backward(self, dout):
        return dout * self.mask
