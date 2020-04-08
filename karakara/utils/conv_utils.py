# Rer: https://github.com/oreilly-japan/deep-learning-from-scratch/
# Ref: https://github.com/chainer/chainer/

import numpy
from ..backend import np


def get_conv_outsize(size, k, s, p, cover_all=False, d=1):
    dk = k + (k - 1) * (d - 1)
    if cover_all:
        return (size + p * 2 - dk + s - 1) // s + 1
    else:
        return (size + p * 2 - dk) // s + 1


def col2im(x, *args, **kwargs):
    if isinstance(x, numpy.ndarray):
        return col2im_cpu(x, *args, **kwargs)
    return col2im_gpu(x, *args, **kwargs)


def im2col(x, *args, **kwargs):
    if isinstance(x, numpy.ndarray):
        return im2col_cpu(x, *args, **kwargs)
    return im2col_gpu(x, *args, **kwargs)


def im2col_old(input_data, filter_h, filter_w, stride=1, pad=0):

    N, C, H, W = input_data.shape
    out_h = (H + 2 * pad - filter_h) // stride + 1
    out_w = (W + 2 * pad - filter_w) // stride + 1

    img = np.pad(input_data, [(0, 0), (0, 0),
                              (pad, pad), (pad, pad)], 'constant')
    col = np.zeros((N, C, filter_h, filter_w, out_h, out_w))

    for y in range(filter_h):
        y_max = y + stride * out_h
        for x in range(filter_w):
            x_max = x + stride * out_w
            col[:, :, y, x, :, :] = img[:, :, y:y_max:stride, x:x_max:stride]

    col = col.transpose(0, 4, 5, 1, 2, 3).reshape(N * out_h * out_w, -1)
    return col


def col2im_old(col, image_shape, filter_h, filter_w, stride=1, pad=0):
    N, C, H, W = image_shape
    out_h = (H + 2 * pad - filter_h) // stride + 1
    out_w = (W + 2 * pad - filter_w) // stride + 1
    col = col.reshape(N, out_h, out_w, C, filter_h,
                      filter_w).transpose(0, 3, 4, 5, 1, 2)

    img = np.zeros((N, C, H + 2 * pad + stride - 1, W + 2 * pad + stride - 1))
    for y in range(filter_h):
        y_max = y + stride * out_h
        for x in range(filter_w):
            x_max = x + stride * out_w
            img[:, :, y:y_max:stride, x:x_max:stride] += col[:, :, y, x, :, :]

    return img[:, :, pad:H + pad, pad:W + pad]


def im2col_cpu(img, kh, kw, sy, sx, ph, pw, pval=0, cover_all=False, dy=1, dx=1,
               out_h=None, out_w=None):
    n, c, h, w = img.shape
    if out_h is None:
        out_h = get_conv_outsize(h, kh, sy, ph, cover_all, dy)
    assert out_h > 0, 'Height in the output should be positive.'
    if out_w is None:
        out_w = get_conv_outsize(w, kw, sx, pw, cover_all, dx)
    assert out_w > 0, 'Width in the output should be positive.'

    img = numpy.pad(img,
                    ((0, 0), (0, 0), (ph, ph + sy - 1), (pw, pw + sx - 1)),
                    mode='constant', constant_values=(pval,))
    col = numpy.ndarray((n, c, kh, kw, out_h, out_w), dtype=img.dtype)

    for j in range(kh):
        jdy = j * dy
        j_lim = jdy + sy * out_h
        for i in range(kw):
            idx = i * dx
            i_lim = idx + sx * out_w
            col[:, :, j, i, :, :] = img[:, :, jdy:j_lim:sy, idx:i_lim:sx]

    col = col.transpose(0, 4, 5, 1, 2, 3).reshape(n * out_h * out_w, -1)
    return col


def im2col_gpu(img, kh, kw, sy, sx, ph, pw, cover_all=False, dy=1, dx=1,
               out_h=None, out_w=None):
    n, c, h, w = img.shape
    if out_h is None:
        out_h = get_conv_outsize(h, kh, sy, ph, cover_all, dy)
    assert out_h > 0, 'Height in the output should be positive.'
    if out_w is None:
        out_w = get_conv_outsize(w, kw, sx, pw, cover_all, dx)
    assert out_w > 0, 'Width in the output should be positive.'

    col = np.empty((n, c, kh, kw, out_h, out_w), dtype=img.dtype)
    np.ElementwiseKernel(
        'raw T img, int32 h, int32 w, int32 out_h, int32 out_w,'
        'int32 kh, int32 kw, int32 sy, int32 sx, int32 ph, int32 pw,'
        'int32 dy, int32 dx',
        'T col',
        '''
           int c0 = i / (kh * kw * out_h * out_w);
           int ky = i / (kw * out_h * out_w) % kh;
           int kx = i / (out_h * out_w) % kw;
           int out_y = i / out_w % out_h;
           int out_x = i % out_w;
           int in_y = ky * dy + out_y * sy - ph;
           int in_x = kx * dx + out_x * sx - pw;
           if (in_y >= 0 && in_y < h && in_x >= 0 && in_x < w) {
             col = img[in_x + w * (in_y + h * c0)];
           } else {
             col = 0;
           }
        ''',
        'im2col')(img.reduced_view(),
                  h, w, out_h, out_w, kh, kw, sy, sx, ph, pw, dy, dx, col)

    col = col.transpose(0, 4, 5, 1, 2, 3).reshape(n * out_h * out_w, -1)
    return col


def col2im_cpu(col, image_shape, kh, kw, sy, sx, ph, pw, dy=1, dx=1):
    n, c, h, w = image_shape
    out_h = (h + 2 * ph - kh) // sx + 1
    out_w = (w + 2 * pw - kw) // sy + 1
    col = col.reshape(n, out_h, out_w, c, kh, kw).transpose(0, 3, 4, 5, 1, 2)
    img = numpy.zeros((n, c, h + 2 * ph + sy - 1, w + 2 * pw + sx - 1),
                      dtype=col.dtype)
    for j in range(kh):
        jdy = j * dy
        j_lim = jdy + sy * out_h
        for i in range(kw):
            idx = i * dx
            i_lim = idx + sx * out_w
            img[:, :, jdy:j_lim:sy, idx:i_lim:sx] += col[:, :, j, i]
    return img[:, :, ph:h + ph, pw:w + pw]


def col2im_gpu(col, image_shape, kh, kw, sy, sx, ph, pw, dy=1, dx=1):
    n, c, h, w = image_shape
    out_h = (h + 2 * ph - kh) // sx + 1
    out_w = (w + 2 * pw - kw) // sy + 1
    col = col.reshape(n, out_h, out_w, c, kh, kw).transpose(0, 3, 4, 5, 1, 2)
    img = np.empty((n, c, h, w), dtype=col.dtype)
    np.ElementwiseKernel(
        'raw T col, int32 h, int32 w, int32 out_h, int32 out_w,'
        'int32 kh, int32 kw, int32 sy, int32 sx, int32 ph, int32 pw,'
        'int32 dx, int32 dy',
        'T img',
        '''
           int c0 = i / (h * w);
           int y  = i / w % h;
           int x  = i % w;
           T val = 0;
           for (int ky = 0; ky < kh; ++ky) {
             int out_y = (y + ph - ky * dy);
             if (0 > out_y || out_y >= out_h * sy) continue;
             if (out_y % sy != 0) continue;
             out_y /= sy;
             for (int kx = 0; kx < kw; ++kx) {
               int out_x = (x + pw - kx * dx);
               if (0 > out_x || out_x >= out_w * sx) continue;
               if (out_x % sx != 0) continue;
               out_x /= sx;
               int k = out_y + out_h * (kx + kw * (ky + kh * c0));
               val = val + col[out_x + out_w * k];
             }
           }
           img = val;
        ''',
        'col2im')(col.reduced_view(),
                  h, w, out_h, out_w, kh, kw, sy, sx, ph, pw, dx, dy, img)
    return img
