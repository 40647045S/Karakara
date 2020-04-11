import torchvision

def main():
    mnist = torchvision.datasets.MNIST('./data/mnist', download=True)
    print(mnist)

