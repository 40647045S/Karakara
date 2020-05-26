import pickle
import collections
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from karakara import config
config.GPU = True
import karakara.backend as K
K.set_random_seed(1988)
# K.set_floatx('float16')
# K.set_epsilon(1e-4)

from karakara.models import Sequential
from karakara.layers import Dense, Dropout
from karakara.activations import Sigmoid, ReLU, LeakyReLU, Tanh
from karakara.optimizers import SGD, Momentum, Adam

from utils import make_cifar10_data

np.random.seed(1988)

image_shape = 3072
randomDim = 100
epochs = 500
batchSize = 128
plot_freq = 5


class GAN_Set:
    def __init__(self, generator, discriminator, gan_model, optimizer):
        self.generator = generator
        self.discriminator = discriminator
        self.gan_model = gan_model
        self.optimizer = optimizer


def make_generator(optimizer):
    generator = Sequential()
    generator.add(Dense(256, input_shape=randomDim))
    generator.add(LeakyReLU(0.2))
    generator.add(Dense(512))
    generator.add(LeakyReLU(0.2))
    generator.add(Dense(1024))
    generator.add(LeakyReLU(0.2))
    generator.add(Dense(2048))
    generator.add(LeakyReLU(0.2))
    generator.add(Dense(image_shape))
    generator.add(Sigmoid())

    return generator


def make_discriminator(optimizer):
    discriminator = Sequential()
    discriminator.add(Dense(2048, input_shape=image_shape))
    discriminator.add(LeakyReLU(0.2))
    discriminator.add(Dropout(0.3))
    discriminator.add(Dense(1024))
    discriminator.add(LeakyReLU(0.2))
    discriminator.add(Dropout(0.3))
    discriminator.add(Dense(512))
    discriminator.add(LeakyReLU(0.2))
    discriminator.add(Dropout(0.3))
    discriminator.add(Dense(256))
    discriminator.add(LeakyReLU(0.2))
    discriminator.add(Dropout(0.3))
    discriminator.add(Dense(1))
    discriminator.add(Sigmoid())
    discriminator.compile(loss='binary_crossentropy',
                          optimizer=optimizer, metric=None)

    return discriminator


def make_gan_model(generator, discriminator, optimizer):
    discriminator.trainable = False
    gan_model = Sequential()
    gan_model.add(generator)
    gan_model.add(discriminator)
    gan_model.compile(loss='binary_crossentropy',
                      optimizer=optimizer, metric=None)

    return gan_model


def save_model(file, generator, discriminator, gan_model, optimizer):
    g_set = GAN_Set(generator, discriminator, gan_model, optimizer)
    pickle.dump(g_set, file)


def plotGeneratedImages(epoch, generator, examples=100, dim=(10, 10), figsize=(10, 10)):
    noise = np.random.normal(0, 1, size=[examples, randomDim])
    generatedImages = generator.predict(noise)
    generatedImages = generatedImages.reshape(examples, 32, 32, 3)

    plt.figure(figsize=figsize)
    for i in range(generatedImages.shape[0]):
        plt.subplot(dim[0], dim[1], i + 1)
        plt.imshow(generatedImages[i], interpolation='nearest')
        plt.axis('off')
    plt.tight_layout()
    plt.savefig(f'gan_images/dgan_cifar10_epoch_{epoch}_karakara.png')
    plt.close()


def train(data, generator, discriminator, gan_model, epochs=1, batchSize=128, plot_freq=5):
    batchCount = int(data.shape[0] / batchSize)
    print(f'Epochs: {epochs}')
    print(f'Batch size: {batchSize}')
    print(f'Batches per epoch: {batchCount}')

    for now_epoch in range(1, epochs + 1):
        print(f'Epoch {now_epoch}/{epochs}')

        pbar = tqdm(range(batchCount),
                    ncols=120, ascii=True, unit='batches')
        for n_batch in pbar:

            noise = np.random.normal(0, 1, size=[batchSize, randomDim])
            imageBatch = data[np.random.randint(
                0, data.shape[0], size=batchSize)]

            generatedImages = generator.predict(noise)
            X = np.concatenate([imageBatch, generatedImages])

            yDis = np.zeros(2 * batchSize)
            yDis[:batchSize] = 0.9

            discriminator.trainable = True
            dloss, _ = discriminator.train_on_batch(X, yDis)

            noise = np.random.normal(0, 1, size=[batchSize, randomDim])
            yGen = np.ones(batchSize)
            discriminator.trainable = False
            gloss, _ = gan_model.train_on_batch(noise, yGen)

            if n_batch % 10 == 0:
                pbar.set_postfix(dloss=f'{dloss:.4f}',
                                 gloss=f'{gloss:.4f}')
                pbar.set_description(f"{n_batch}/{batchCount}")

        if now_epoch % plot_freq == 0:
            plotGeneratedImages(now_epoch, generator)


def main():
    (X_train, _), (_, _), (X_test, _) = make_cifar10_data(valid_ratio=0, image_data_format='channels_last')
    X_train = np.concatenate([X_train, X_test])
    X_train = X_train.reshape(X_train.shape[0], -1)

    adam = Adam(lr=0.0002, beta_1=0.5)
    generator, discriminator = make_generator(adam), make_discriminator(adam)
    gan_model = make_gan_model(generator, discriminator, adam)

    train(X_train, generator, discriminator,
          gan_model, epochs=epochs, batchSize=batchSize, plot_freq=plot_freq)


if __name__ == '__main__':
    main()
