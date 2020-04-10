from ..backend import np
from ..engine.base_layer import Layer


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
        return sum(inputs)

    def backward(self, dout):
        return dout
