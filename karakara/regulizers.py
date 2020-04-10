from .backend import np


class Regularizer(object):

    def __call__(self, x):
        return 0.


class L1L2(Regularizer):

    def __init__(self, l1=0., l2=0.):
        self.l1 = l1
        self.l2 = l2

    def __call__(self, x):
        regularization = 0.
        if self.l1:
            regularization += self.l1 * np.sum(np.abs(x))
        if self.l2:
            regularization += self.l2 * np.sum(np.square(x))
        return regularization

    def cal_grad(self, x):
        regularization = np.zeros_like(x)
        if self.l1:
            regularization += self.l1
        if self.l2:
            regularization += self.l2 * x
        return regularization

    def get_config(self):
        return {'l1': float(self.l1),
                'l2': float(self.l2)}


def l1(l=0.01):
    return L1L2(l1=l)


def l2(l=0.01):
    return L1L2(l2=l)


def l1_l2(l1=0.01, l2=0.01):
    return L1L2(l1=l1, l2=l2)
