import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import time
import cupy as cp


def main():
    a = cp.random.randn(128, 32, 32, 32).astype('float32')
    # b = cp.random.randn(200, 100000).astype('float32')

    start_time = time.time()

    for _ in range(10000):
        x = cp.mean(a, axis=(0, 2, 3))

    end_time = time.time()
    print(f'Use time: {end_time - start_time}')


if __name__ == '__main__':
    main()
