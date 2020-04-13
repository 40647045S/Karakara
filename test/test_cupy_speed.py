import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import time
import math
import cupy as cp


def main():
    a = cp.random.randn(100, 100).astype('float32')
    b = cp.random.randn(2000, 2000).astype('float32')

    a = cp.arange(10)

    start_time = time.time()

    # for i in range(1, 50):

    print(a[5:100])

    end_time = time.time()
    print(f'Use time: {end_time - start_time}')


if __name__ == '__main__':
    main()
