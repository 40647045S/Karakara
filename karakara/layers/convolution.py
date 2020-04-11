# Rer: https://github.com/oreilly-japan/deep-learning-from-scratch/

from numpy import prod

from ..backend import np, setup_data
from ..engine.base_layer import Layer
from ..utils.conv_utils import im2col, col2im
from ..utils.math_utils import cal_init_std


class Conv2D(Layer):
    def __init__(self, filters, kernel_size, strides=1, padding='valid', kernel_regularizer=None, input_shape=None, **kwargs):
        super().__init__(**kwargs)

        self.filters = filters
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)

        self.kernel_h = kernel_size[0]
        self.kernel_w = kernel_size[1]
        self.stride = strides

        if padding == 'valid':
            self.pad = 0
        elif padding == 'same':
            self.pad = kernel_size[0] // 2

        self.kernel_regularizer = kernel_regularizer

        self.input_shape = input_shape

        self.x = None
        self.col = None
        self.col_W = None

    def build(self, input_shape, **kwargs):
        if not self.built:
            if not input_shape:
                input_shape = self.input_shape

            if self.input_shape:
                assert self.input_shape == input_shape

            C, H, W = input_shape
            self.channel = C

            pre_node_nums = C * self.kernel_h * self.kernel_w
            weight_std = cal_init_std('He', pre_node_nums)

            self.kernel = self.add_weight(
                shape=(self.filters, self.channel, self.kernel_h, self.kernel_w), std=weight_std, regularizer=self.kernel_regularizer)
            self.bias = self.add_weight(
                shape=(self.filters, ), mean=0, initializer='constant')

            out_h = 1 + int((H + 2 * self.pad - self.kernel_h) / self.stride)
            out_w = 1 + int((W + 2 * self.pad - self.kernel_w) / self.stride)

            self.output_shape = (self.filters, out_h, out_w)
            self.built = True

    def compute_output_shape(self):
        return self.output_shape

    def call(self, inputs, **kwargs):
        N, C, H, W = inputs.shape
        out_h = 1 + int((H + 2 * self.pad - self.kernel_h) / self.stride)
        out_w = 1 + int((W + 2 * self.pad - self.kernel_w) / self.stride)

        col = im2col(inputs, self.kernel_h,
                     self.kernel_w, self.stride, self.stride, self.pad, self.pad)
        col_W = self.kernel.weight.reshape(self.filters, -1).T

        out = np.dot(col, col_W) + self.bias.weight
        out = out.reshape(N, out_h, out_w, -1).transpose(0, 3, 1, 2)

        self.x = inputs
        self.col = col
        self.col_W = col_W

        return out

    def backward(self, dout):
        _, _, h, w = self.x.shape

        dout = dout.transpose(0, 2, 3, 1).reshape(-1, self.filters)

        np.sum(dout, axis=0, out=self.bias.gradient)
        self.kernel.gradient = np.dot(self.col.T, dout)
        self.kernel.gradient = self.kernel.gradient.transpose(
            1, 0).reshape(self.kernel.weight.shape)

        dcol = np.dot(dout, self.col_W.T)
        dx = col2im(dcol, self.x.shape, self.kernel_h,
                    self.kernel_w, self.stride, self.stride, self.pad, self.pad)

        return dx


class MaxPooling2D(Layer):

    def __init__(self, pool_h, pool_w, stride=None, pad='valid', **kwargs):
        super().__init__(**kwargs)
        self.pool_h = pool_h
        self.pool_w = pool_w

        if stride:
            self.stride = stride
        else:
            self.stride = pool_h

        if pad == 'valid':
            self.pad = 0
        elif pad == 'same':
            self.pad = pool_h // 2

        self.x = None
        self.arg_max = None

    def build(self, input_shape, **kwargs):
        C, H, W = input_shape
        out_h = int(1 + (H - self.pool_h) / self.stride)
        out_w = int(1 + (W - self.pool_w) / self.stride)
        self.output_shape = (C, out_h, out_w)

    def compute_output_shape(self):
        return self.output_shape

    def call(self, inputs, **kwargs):
        N, C, H, W = inputs.shape
        out_h = int(1 + (H - self.pool_h) / self.stride)
        out_w = int(1 + (W - self.pool_w) / self.stride)

        col = im2col(inputs, self.pool_h, self.pool_w,
                     self.stride, self.stride, self.pad, self.pad)
        col = col.reshape(-1, self.pool_h * self.pool_w)

        arg_max = np.argmax(col, axis=1)
        out = np.max(col, axis=1)
        out = out.reshape(N, out_h, out_w, C).transpose(0, 3, 1, 2)

        self.x = inputs
        self.arg_max = arg_max

        return out

    def backward(self, dout):
        _, _, h, w = self.x.shape

        dout = dout.transpose(0, 2, 3, 1)

        pool_size = self.pool_h * self.pool_w
        dmax = setup_data(np.zeros((dout.size, pool_size)))
        dmax[np.arange(self.arg_max.size),
             self.arg_max.flatten()] = dout.flatten()
        dmax = dmax.reshape(dout.shape + (pool_size,))
        dcol = dmax.reshape(dmax.shape[0] * dmax.shape[1] * dmax.shape[2], -1)
        dx = col2im(dcol, self.x.shape, self.pool_h,
                    self.pool_w, self.stride, self.stride, self.pad, self.pad)

        return dx


class AveragePooling2DAll(Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self, input_shape, **kwargs):
        C, _, _ = input_shape
        self.output_shape = (C, 1, 1)

    def compute_output_shape(self):
        return self.output_shape

    def call(self, inputs, **kwargs):
        self.input_shape = inputs.shape
        _, _, W, H = inputs.shape
        self.mean_size = W * H
        out = np.mean(inputs, axis=(2, 3), keepdims=True)

        return out

    def backward(self, dout):
        dx = np.broadcast_to(dout, self.input_shape)
        dx /= self.mean_size

        return dx
