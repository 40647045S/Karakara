from karakara.config import GPU
from .common import floatx

if GPU:
    import cupy as np
    np.cuda.set_allocator(np.cuda.MemoryPool().malloc)

    print('\033[92m' + '-' * 60 + '\033[0m')
    print(' ' * 23 + '\033[92mGPU Mode (cupy)\033[0m')
    print('\033[92m' + '-' * 60 + '\033[0m\n')
else:
    import numpy as np


def set_random_seed(seed):
    np.random.seed(seed)


def setup_data(x, device=0):
    if GPU:
        with np.cuda.Device(device):
            return np.asarray(x, dtype=floatx())
    else:
        return x.astype(floatx())


def restore_data(x):
    if GPU:
        return np.asnumpy(x)
    else:
        return x


def to_gpu(x, device=0):
    with np.cuda.Device(device):
        return np.asarray(x)


def to_cpu(x):
    return np.asnumpy(x)
