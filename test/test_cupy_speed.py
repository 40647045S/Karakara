import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import time
import cupy as cp


def main():
    a = cp.random.randn(2000000, 1).astype('float32')
    b = cp.array(0.02).astype('float16')
    c = 0.02
    # b = cp.random.randn(200, 100000).astype('float32')

    start_time = time.time()

    for _ in range(20000):
        x = a ** 5

    end_time = time.time()
    print(f'Use time: {end_time - start_time}')


if __name__ == '__main__':
    main()
