import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import time
import cupy as cp


def main():
    a = cp.random.randn(2000, 2000).astype('float32')
    b = cp.random.randn(2000, 1).astype('float32')
    c = cp.random.randn(2000, 2000).astype('float32')

    start_time = time.time()

    for _ in range(20000):
        cp.ElementwiseKernel('P a, P b',
                             'P c',
                             """
        c = a + b;
        """,
                             'qq')(a, b, c)
        # c = a + b

    end_time = time.time()
    print(f'Use time: {end_time - start_time}')


if __name__ == '__main__':
    main()
