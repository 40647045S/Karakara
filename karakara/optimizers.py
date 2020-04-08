from .backend import np, epsilon


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
            weight.weight -= self.lr * weight.gradient


class Momentom(Optimizer):

    def __init__(self, lr=0.01, momentum=0.9):
        self.lr = lr
        self.momentum = momentum
        self.v = {}

    def update(self, weights):
        for weight in self.get_trainable(weights):
            if not weight in self.v:
                self.v[weight] = np.zeros_like(weight.weight)
            # param_v = self.v[weight]
            self.v[weight] = self.momentum * \
                self.v[weight] - self.lr * weight.gradient
            weight.weight += self.v[weight]


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

            param_h = self.h[weight]
            param_h = self.rho * param_h + (1 - self.rho) * weight.gradient**2

            weight.weight -= self.current_lr * weight.gradient / (np.sqrt(param_h) + self.epsilon)


class Adam(Optimizer):

    def __init__(self, lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=epsilon()):
        self.lr = lr
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.iter = 0
        self.m = {}
        self.v = {}
        self.epsilon = epsilon

    def update(self, weights):

        self.iter += 1
        lr_t = self.lr * np.sqrt(1.0 - self.beta_2 **
                                 self.iter) / (1.0 - self.beta_1**self.iter)

        for weight in self.get_trainable(weights):
            if not weight in self.v:
                self.v[weight] = np.zeros_like(weight.weight)
                self.m[weight] = np.zeros_like(weight.weight)
            param_v = self.v[weight]
            param_m = self.m[weight]

            param_m += (1 - self.beta_1) * (weight.gradient - param_m)
            param_v += (1 - self.beta_2) * (weight.gradient**2 - param_v)

            weight.weight -= lr_t * param_m / (np.sqrt(param_v) + self.epsilon)


optimizer_table = {
    'sgd': SGD,
    'momentom': Momentom,
    'adam': Adam,
}


def get(identifier):
    if isinstance(identifier, str):
        return optimizer_table[identifier.lower()]()
    elif isinstance(identifier, Optimizer):
        return identifier
    else:
        raise ValueError(f'Unknow optimizer {identifier}')
