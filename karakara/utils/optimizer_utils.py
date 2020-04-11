import numpy
from ..backend import np
from ..config import GPU

if GPU:
    adam_kernel = np.ElementwiseKernel(
        'P grad, T cbeta1, T cbeta2, '
        'T epsilon, T lr',
        'P param, P m, P v',
        '''
            m += cbeta1 * (grad - m);
            v += cbeta2 * (grad * grad - v);
            param -= lr * m / (sqrt(v) + epsilon)
        ''',
        'adam')


def SGD_update():
    pass


def Momentom_update():
    pass


def RMSprop_update():
    pass


def adam_update(grad, cbeta_1, cbeta_2, epsilon,
                lr_t, weight, param_m, param_v):

    if isinstance(grad, numpy.ndarray):
        param_m += cbeta_1 * (grad - param_m)
        param_v += + cbeta_2 * (np.square(grad) - param_v)
        weight -= - lr_t * param_m / (np.sqrt(param_v) + epsilon)
    else:
        adam_kernel(grad, cbeta_1, cbeta_2, epsilon,
                    lr_t, weight, param_m, param_v)
