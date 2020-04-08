from ..backend import np, setup_data, to_gpu
from ..utils.generic_utils import naming_system


class Weight:

    def __init__(self, weight, trainable=True, name=None):
        self.weight = weight
        self.gradient = np.zeros_like(weight)
        self.trainable = trainable
        self.name = name

    def get_params_count(self):
        return self.weight.size

    def to_gpu(self, device=0):
        self.weight = to_gpu(self.weight, device=device)
        self.gradient = to_gpu(self.gradient, device=device)


class Layer(object):

    def __init__(self, name=None, trainable=True):
        self.name = name if name else naming_system.autoname(self)
        self._trainable = trainable
        self.trainable_weights = []
        self.non_trainable_weights = []
        self.built = False
        self.device = 0

    @property
    def trainable(self):
        return self._trainable

    @trainable.setter
    def trainable(self, new_trainable):
        self._trainable = new_trainable
        for trainable_weight in self.trainable_weights:
            trainable_weight.trainable = new_trainable

    def add_weight(self, shape, mean=0, std=0, initializer='normal', trainable=True):
        if initializer == 'normal':
            value = setup_data(std * np.random.randn(*shape) + mean)
        elif initializer == 'uniform':
            value = setup_data(std * np.random.rand(*shape) + mean)
        elif initializer == 'constant':
            value = setup_data(np.full(shape, mean))
        else:
            raise ValueError(f'Unknow initializer: {initializer}')

        weight = Weight(value, name=self.name)

        if trainable:
            self.trainable_weights.append(weight)
        else:
            self.non_trainable_weights.append(weight)

        return weight

    def get_trainable_weights(self):
        return self.trainable_weights

    def get_non_trainable_weights(self):
        return self.non_trainable_weights

    def get_params_count(self):
        trainable_count = sum([w.get_params_count()
                               for w in self.trainable_weights])
        non_trainable_count = sum([w.get_params_count()
                                   for w in self.non_trainable_weights])

        return trainable_count, non_trainable_count

    def to_gpu(self, device=0):
        self.device = device
        for weight in self.trainable_weights:
            weight.to_gpu(device)
        for weight in self.non_trainable_weights:
            weight.to_gpu(device)

    def build(self, input_shape, **kwargs):
        raise NotImplementedError

    def compute_output_shape(self):
        raise NotImplementedError

    def __call__(self, x):
        return self.call(x)

    def call(self, x, **kwargs):
        raise NotImplementedError

    def backward(self, dout):
        raise NotImplementedError
