from .backend import np, epsilon
from .engine.base_layer import Layer
from .utils.math_utils import softmax, categorical_crossentropy_error


class BaseLossLayer(Layer):
    def __init__(self):
        super().__init__(trainable=False)

    def build(self, input_shape, **kwargs):
        self.output_shape = 1

    def compute_output_shape(self):
        return 1


class CategoricalCrossEntropy(BaseLossLayer):

    def __init__(self, epsilon=epsilon()):
        super().__init__()
        self.pred = None
        self.label = None
        self.loss = None
        self.epsilon = epsilon
        self.output_shape = None

    def call(self, inputs, labels, **kwargs):
        self.pred = inputs
        self.label = labels
        self.loss = categorical_crossentropy_error(self.pred, self.label)

        return self.loss

    def backward(self, dout=1):
        batch_size = self.label.shape[0]

        if self.label.ndim == 1:
            label = np.zeros_like(self.pred)
            label[np.arange(batch_size), self.label.astype('int')] = 1
        else:
            label = self.label

        dx = -1 * label / np.maximum(self.pred, self.epsilon)
        dx = dx / batch_size

        return dx


class MeanSquareError(BaseLossLayer):

    def __init__(self, epsilon=epsilon(), reduction='mean'):
        super().__init__()
        self.pred = None
        self.label = None
        self.loss = None
        self.output_shape = None
        self.reduction = reduction

    def call(self, inputs, labels, **kwargs):
        if inputs.ndim == 1:
            inputs = inputs.reshape(-1, 1)
        self.pred = inputs
        if labels.ndim == 1:
            labels = labels.reshape(-1, 1)
        self.label = labels
        batch_size = self.label.shape[0]
        self.loss = np.square(self.pred - self.label) / 2

        loss = np.sum(self.loss)
        if self.reduction == 'mean':
            loss /= batch_size

        return loss

    def backward(self, dout=1):
        batch_size = self.label.shape[0]
        dx = (self.pred - self.label)

        if self.reduction == 'mean':
            dx /= batch_size

        return dx


class BinaryCrossEntropy(BaseLossLayer):

    def __init__(self, epsilon=epsilon(), reduction='mean'):
        super().__init__()
        self.pred = None
        self.epsilon = epsilon
        self.label = None
        self.loss = None
        self.output_shape = None
        self.reduction = reduction

    def call(self, inputs, labels, **kwargs):
        if inputs.ndim == 1:
            inputs = inputs.reshape(-1, 1)
        self.pred = inputs
        if labels.ndim == 1:
            labels = labels.reshape(-1, 1)
        self.label = labels
        batch_size = self.label.shape[0]
        self.loss = -(self.label * np.log(self.pred) +
                      (1 - self.label) * np.log(1 - self.pred + self.epsilon))

        loss = np.sum(self.loss)
        if self.reduction == 'mean':
            loss /= batch_size

        return loss

    def backward(self, dout=1):
        batch_size = self.label.shape[0]
        dx = - (np.divide(self.label, self.pred + self.epsilon) -
                np.divide(1 - self.label, 1 - self.pred + self.epsilon))

        if self.reduction == 'mean':
            dx = dx / batch_size

        return dx


class SoftmaxWithLoss(BaseLossLayer):
    def __init__(self):
        super().__init__()
        self.loss = None
        self.y = None
        self.t = None

    def forward(self, x, t, **kwargs):
        self.t = t
        self.y = softmax(x)
        self.loss = categorical_crossentropy_error(self.y, self.t)

        return self.loss

    def backward(self, dout=1):
        batch_size = self.t.shape[0]
        if self.t.size == self.y.size:
            dx = (self.y - self.t) / batch_size
        else:
            dx = self.y.copy()
            dx[np.arange(batch_size), self.t] -= 1
            dx = dx / batch_size

        return dx


loss_table = {
    'categorical_crossentropy': CategoricalCrossEntropy,
    'mse': MeanSquareError,
    'binary_crossentropy': BinaryCrossEntropy,
}


def get(identifier):
    if isinstance(identifier, str):
        return loss_table[identifier.lower()]()
    elif isinstance(identifier, BaseLossLayer):
        return identifier
    else:
        raise ValueError(f'Unknow loss {identifier}')
