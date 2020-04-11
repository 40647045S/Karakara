import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import time
import math
import cupy as cp


def main():
    a = cp.random.randn(2000, 2000).astype('float32')
    b = cp.random.randn(2000, 2000).astype('float32')
    a = cp.arange(9).reshape(3, 3)

    start_time = time.time()

    # for i in range(1, 50):

    a[cp.array([0, 1, 2]), cp.array([0, 1, 2])] = -1
    print(a)

    end_time = time.time()
    print(f'Use time: {end_time - start_time}')


if __name__ == '__main__':
    main()
