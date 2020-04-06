import threading
from copy import deepcopy

from ..backend import to_gpu, np
from ..engine.sequential import Sequential


def cal_grad(model, X, y, device):
    with np.cuda.Device(device):
        model.evaluate(X, y, training=True)
        model.cal_gradient()


class MultiGPUModel(Sequential):
    def __init__(self, model, gpus=2):
        self.gpus = gpus
        self.submodel = deepcopy(model).to_gpu(1)

    def train_on_batch(self, batch_X, batch_y):

        batch_size = batch_X.shape[0] // 2
        batch_X_1, batch_X_2 = batch_X[:batch_size], batch_X[batch_size:]
        batch_y_1, batch_y_2 = batch_y[:batch_size], batch_y[batch_size:]

        submodel = deepcopy(self)
        submodel.to_gpu(device=1)

        t1 = threading.Thread(target=cal_grad, args=(
            self, batch_X_1, batch_y_1, 0))

        t2 = threading.Thread(target=cal_grad, args=(
            submodel, batch_X_2, batch_y_2, 1))

        t1.start()
        t2.start()

        t1.join()
        t2.join()

        # batch_loss_1, batch_metric_1 = self.evaluate(
        #     batch_X_1, batch_y_1, training=True)
        # self.cal_gradient()

        # with np.cuda.Device(1):
        #     submodel = deepcopy(self)
        #     submodel.to_gpu(device=1)
        #     batch_loss_2, batch_metric_2 = submodel.evaluate(
        #         batch_X_2, batch_y_2, training=True)
        #     submodel.cal_gradient()
        # print(batch_loss_2, batch_metric_2)

        # print(f'loss: {batch_loss_1}, {batch_loss_2}')
        # print(f'metric: {batch_metric_1}, {batch_metric_2}')
        # print(batch_loss_1, batch_loss_2)
        # batch_loss = (batch_loss_1 + batch_loss_2) / 2
        # batch_metric = (batch_metric_1 + batch_metric_2) / 2

        batch_loss, batch_metric = 3, 3

        for w1, w2 in zip(self.trainable_weights, submodel.trainable_weights):
            w1.gradient = (w1.gradient + to_gpu(w2.gradient, 0)) / 2

        self.update()

        return batch_loss, batch_metric


def multi_gpu_model(model, gpus):
    model.submodel = deepcopy(model)
    model.submodel.device = 1
    model.submodel.to_gpu(device=1)
    model.__class__ = MultiGPUModel

    return model
