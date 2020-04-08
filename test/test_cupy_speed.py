import time
import cupy as cp


def main():
    a = cp.random.randn(200, 64, 32, 32).astype('float32')
    b = cp.arange(30)

    start_time = time.time()

    for _ in range(10000):
        c = a[b]

    end_time = time.time()
    print(f'Use time: {end_time - start_time}')


if __name__ == '__main__':
    main()
