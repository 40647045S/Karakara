from math import sqrt
from numpy import prod
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
            print(input_shape)
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

    def call(self, inputs, training=True, **kwargs):
        if training:
            self.mask = np.random.rand(*inputs.shape) > self.dropout_ratio
            return inputs * self.mask
        else:
            return inputs * (1.0 - self.dropout_ratio)

    def backward(self, dout):
        return dout * self.mask


class Same(Layer):
    def __init__(self):
        super().__init__()

    def build(self, input_shape):
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

    def build(self, input_shape):
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

    def build(self, input_shape):
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

    def build(self, input_shape):
        self.output_shape = (prod(input_shape), )

    def compute_output_shape(self):
        return self.output_shape

    def call(self, inputs, **kwargs):
        batch_size = inputs.shape[0]
        self.input_shape = inputs.shape
        return inputs.reshape(batch_size, -1)

    def backward(self, dout):
        return dout.reshape(self.input_shape)
