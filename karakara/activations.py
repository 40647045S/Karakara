from .backend import np, epsilon
from .engine.base_layer import Layer
from .utils.math_utils import softmax


class BaseActivation(Layer):
    def __init__(self):
        super().__init__(trainable=False)
        self.output_shape = None

    def build(self, input_shape, **kwargs):
        self.output_shape = input_shape

    def compute_output_shape(self):
        return self.output_shape


class Sigmoid(BaseActivation):
    def __init__(self):
        super().__init__()
        self.out = None

    def call(self, x, **kwargs):
        out = 1 / (1 + np.exp(-x))
        self.out = out
        return out

    def backward(self, dout):
        dx = dout * (1.0 - self.out) * self.out

        return dx


class Tanh(BaseActivation):
    def __init__(self):
        super().__init__()
        self.out = None

    def call(self, x, **kwargs):
        out = np.tanh(x)
        self.out = out
        return out

    def backward(self, dout):
        dx = dout * (1.0 - np.power(self.out, 2))

        return dx


class ReLU(BaseActivation):

    def __init__(self):
        super().__init__()
        self.mask = None

    def call(self, inputs, **kwargs):
        self.mask = (inputs <= 0)
        output = inputs.copy()
        output[self.mask] = 0

        return output

    def backward(self, dout):
        dout[self.mask] = 0
        dx = dout

        return dx


class LeakyReLU(BaseActivation):

    def __init__(self, alpha=0.1):
        super().__init__()
        self.mask = None
        self.alpha = alpha

    def call(self, inputs, **kwargs):
        self.mask = (inputs <= 0)
        output = inputs.copy()
        output[self.mask] *= self.alpha

        return output

    def backward(self, dout):
        dout[self.mask] *= self.alpha
        dx = dout

        return dx


class Softmax(BaseActivation):

    def __init__(self, epsilon=epsilon()):
        super().__init__()
        self.out = None
        self.epsilon = epsilon
        self.output_shape = None

    def compute_output_shape(self):
        return self.output_shape

    def call(self, x, **kwargs):
        self.out = softmax(x)
        return self.out

    def backward(self, dout):
        fraction = np.maximum(self.out - self.out ** 2, self.epsilon)
        dx = fraction * dout
        return dx


activation_table = {'sigmoid': Sigmoid, 'tanh': Tanh,
                    'relu': ReLU, 'leakyrelu': LeakyReLU, 'softmax': Softmax}


def Activation(name, **kwargs):
    if not name in activation_table:
        raise ValueError(f'Unknow activation: {name}')
    activation = activation_table[name]
    return activation(**kwargs)
