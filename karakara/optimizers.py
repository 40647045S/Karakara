from .backend import np, epsilon, setup_data
from .utils.optimizer_utils import adam_update, momentum_update


class Optimizer:

    def get_trainable(self, weights):
        return (w for w in weights if w.trainable)

    def update(self, weights):
        raise NotImplementedError


class SGD(Optimizer):

    def __init__(self, lr=0.01):
        self.lr = lr

    def update(self, weights):
        for weight in self.get_trainable(weights):
            weight.weight -= self.lr * weight.regularized_grad()


class Momentum(Optimizer):

    def __init__(self, lr=0.01, momentum=0.9):
        self.lr = lr
        self.momentum = momentum
        self.v = {}

    def update(self, weights):
        for weight in self.get_trainable(weights):
            if not weight in self.v:
                self.v[weight] = np.zeros_like(weight.weight)

            momentum_update(weight.regularized_grad(), self.momentum, self.lr, weight.weight, self.v[weight])
            # self.v[weight] = self.momentum * self.v[weight] - self.lr * weight.regularized_grad()
            # weight.weight += self.v[weight]


class RMSprop(Optimizer):
    def __init__(self, lr=0.001, rho=0.9, epsilon=epsilon(), decay=0.0):
        self.lr = lr
        self.current_lr = lr
        self.epsilon = epsilon
        self.rho = rho
        self.decay = decay
        self.h = {}
        self.iter = 0

    def update(self, weights):
        self.iter += 1
        self.current_lr *= (1 - self.decay)

        for weight in self.get_trainable(weights):
            if not weight in self.h:
                self.h[weight] = np.zeros_like(weight.weight)

            grad = weight.regularized_grad()
            param_h = self.h[weight]
            param_h = self.rho * param_h + (1 - self.rho) * np.square(grad)

            weight.weight -= self.current_lr * grad / (np.sqrt(param_h) + self.epsilon)


class Adam(Optimizer):

    def __init__(self, lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=epsilon(), decay=0.):
        self.origin_lr = lr
        self.lr = lr
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.cbeta_1 = 1 - beta_1
        self.cbeta_2 = 1 - beta_2
        self.decay = (1 - decay)
        self.iter = 0
        self.m = {}
        self.v = {}
        self.epsilon = epsilon

    def update(self, weights):

        self.iter += 1
        # print(lr_t)
        lr_t = (self.lr * (self.decay ** self.iter)) * \
            np.sqrt(1.0 - self.beta_2 ** self.iter) / (1.0 - self.beta_1**self.iter)

        for weight in self.get_trainable(weights):
            if not weight in self.v:
                self.v[weight] = np.zeros_like(weight.weight)
                self.m[weight] = np.zeros_like(weight.weight)
            # param_v = self.v[weight]
            # param_m = self.m[weight]
            # grad = weight.regularized_grad()

            # print(weight.name, grad.dtype)
            adam_update(weight.regularized_grad(), self.cbeta_1, self.cbeta_2, self.epsilon,
                        lr_t, weight.weight, self.m[weight], self.v[weight])


optimizer_table = {
    'sgd': SGD,
    'momentom': Momentum,
    'adam': Adam,
    'rmsprop': RMSprop,
}


def get(identifier):
    if isinstance(identifier, str):
        return optimizer_table[identifier.lower()]()
    elif isinstance(identifier, Optimizer):
        return identifier
    else:
        raise ValueError(f'Unknow optimizer {identifier}')
