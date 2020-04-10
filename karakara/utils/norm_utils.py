import numpy
from ..backend import np


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
