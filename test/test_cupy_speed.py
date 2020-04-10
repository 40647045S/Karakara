import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import time
import math
import cupy as cp


def main():
    a = cp.random.randn(2000, 2000).astype('float32')
    b = cp.random.randn(2000, 1).astype('float32')
    c = cp.random.randn(2000, 2000).astype('float32')

    start_time = time.time()

    for i in range(1, 1250 * 50):

        print(i, 0.001 * (1 - 3e-5)**i)

    end_time = time.time()
    print(f'Use time: {end_time - start_time}')


if __name__ == '__main__':
    main()
