import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import time
import math
import cupy as cp


def main():
    a = cp.random.randn(2000, 2000).astype('int64')
    b = cp.random.randn(2000, 2000).astype('float32')

    start_time = time.time()

    # for i in range(1, 50):

    c = a / b
    print(c.dtype)

    end_time = time.time()
    print(f'Use time: {end_time - start_time}')


if __name__ == '__main__':
    main()
