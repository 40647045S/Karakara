import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import time
import math
import cupy as cp
import common


def main():
    a = cp.random.randn(2000, 2000).astype('float32')
    b = cp.random.randn(2000, 1).astype('float32')
    c = cp.random.randn(2000, 2000).astype('float32')

    start_time = time.time()

    # for i in range(1, 50):

    c = cp.array([[1, 2, 3, 4], [5, 6, 7, 8]]).reshape(2, 4, 1, 1)
    d = cp.broadcast_to(c, (2, 4, 2, 2))
    print(c)
    print(d)
    print(d.shape)

    end_time = time.time()
    print(f'Use time: {end_time - start_time}')


if __name__ == '__main__':
    main()
