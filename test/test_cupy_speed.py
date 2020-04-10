import time
import cupy as cp


def main():
    # a = cp.random.randn(100000, 200).astype('float32')
    # b = cp.random.randn(200, 100000).astype('float32')

    a = cp.random.randn(100000).astype('float32')
    b = cp.random.randn(200, 100000).astype('float32')

    start_time = time.time()

    for _ in range(10000):

        a + b

    end_time = time.time()
    print(f'Use time: {end_time - start_time}')


if __name__ == '__main__':
    main()
