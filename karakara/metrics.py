from .backend import np


class Metric:
    nickname = ''

    def call(self, y_true, y_pred):
        return NotImplementedError

    def __call__(self, y_true, y_pred):
        return self.call(y_true, y_pred)


class Accuracy(Metric):
    nickname = 'acc'

    def __init__(self):
        super().__init__()

    def call(self, y_true, y_pred):
        y_pred = np.argmax(y_pred, axis=1)
        if y_true.ndim != 1:
            y_true = np.argmax(y_true, axis=1)

        accuracy = np.sum(y_pred == y_true) / float(y_pred.shape[0])
        return accuracy


class MSE(Metric):
    nickname = 'mse'

    def __init__(self):
        super().__init__()

    def call(self, y_true, y_pred):
        return np.mean((y_pred - y_true) ** 2 / 2)


class LogLoss(Metric):
    nickname = 'log_loss'

    def __init__(self):
        super().__init__()

    def call(self, y_true, y_pred):
        batch_size = y_true.shape[0]
        loss = -(y_true * np.log(y_pred) +
                 (1 - y_true) * np.log(1 - y_pred))

        return np.sum(loss) / batch_size


metric_tabel = {
    'mse': MSE,
    'accuracy': Accuracy,
    'log_loss': LogLoss
}


def get(identifier):
    if isinstance(identifier, str):
        return metric_tabel[identifier.lower()]()
    elif isinstance(identifier, Metric):
        return identifier
    else:
        raise ValueError(f'Unknow metric {identifier}')
