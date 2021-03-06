from ..backend import np, epsilon


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def softmax(x):
    x = x - np.max(x, axis=-1, keepdims=True)
    x = np.exp(x)
    return x / np.sum(x, axis=-1, keepdims=True)


def categorical_crossentropy_error(y, t):
    t = t.astype('int')

    if y.ndim == 1:
        y = y.reshape(1, y.size)
        t = t.reshape(1, t.size)

    if t.size == y.size:
        t = t.argmax(axis=1)

    batch_size = y.shape[0]
    return -np.sum(np.log(y[np.arange(batch_size), t] + epsilon())) / batch_size


def cal_init_std(initializer, pre_node_nums):
    if initializer == 'Xavier':
        return np.sqrt(1.0 / pre_node_nums)
    elif initializer == 'He':
        return np.sqrt(2.0 / pre_node_nums)
    elif isinstance(initializer, (int, float)):
        return initializer
    else:
        raise ValueError(f'initializer: {initializer} 是在哈樓？')
