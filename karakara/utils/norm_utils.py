import numpy
from ..backend import np, setup_data


def update_mean_var(mean, var, decay, adjust, running_mean, running_var):
    if isinstance(mean, numpy.ndarray):
        running_mean -= (running_mean - mean) * decay
        running_var -= (running_var - var) * decay
    else:
        np.ElementwiseKernel(
            'T mean, T var, U decay, U adjust',
            'U r_mean, U r_var',
            '''
            r_mean = r_mean - (r_mean - mean) * decay;
            r_var = r_var - (r_var - var) * decay;
            ''',
            'update_mean_var'
        )(mean, var, decay, adjust,
          running_mean, running_var)


def bn_backward(gamma, dout, xn, N, inv_std):
    dxn = gamma * dout
    if isinstance(gamma, numpy.ndarray):
        dxc = (dxn - xn / N * np.sum((dxn * xn), axis=1, keepdims=True)) * inv_std
    else:
        N = setup_data(N)
        dxc = np.zeros_like(dout)
        sss = np.sum((dxn * xn), axis=1, keepdims=True)
        np.ElementwiseKernel(
            'P dxn, T xn, P N, P sss, T inv_std',
            'P dxc',
            '''
                dxc = (dxn - xn / N * sss) * inv_std
                ''',
            'bn_backward'
        )(dxn, xn, N, sss, inv_std, dxc)

    return dxc - np.sum(dxc, axis=1, keepdims=True) / N
