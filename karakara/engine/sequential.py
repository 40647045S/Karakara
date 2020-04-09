from tqdm import tqdm
from statistics import mean
from itertools import chain

from .base_layer import Layer
from ..backend import np, setup_data, restore_data
from ..utils.generic_utils import has_arg
from .. import optimizers
from .. import losses
from .. import metrics


class Sequential(Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.layers = []
        self.input_shape = None
        self.output_shape = None
        self.lossLayer = None
        self.optimizer = None
        self.metric = None
        self.pre_node_nums = None

    def build(self, input_shape, **kwargs):
        self.built = True

    def compute_output_shape(self):
        return self.output_shape

    def add(self, layers):
        if not isinstance(layers, (list, tuple)):
            layers = [layers]

        self.layers.append(layers)
        output_shapes = []

        for layer in layers:
            pre_node_nums = layer.build(
                self.output_shape, pre_node_nums=self.pre_node_nums)
            if pre_node_nums:
                self.pre_node_nums = pre_node_nums
            self.trainable_weights.extend(layer.get_trainable_weights())
            self.non_trainable_weights.extend(
                layer.get_non_trainable_weights())
            output_shapes.append(layer.compute_output_shape())

        self.output_shape = output_shapes
        if len(output_shapes) == 1:
            self.output_shape = self.output_shape[0]

    def summary(self):
        print("_________________________________________________________________")
        print("Layer (type)                 Output Shape              Param #   ")
        print("=================================================================")
        total_ct_params = 0
        total_nt_params = 0
        for layer in chain(*self.layers):
            layer_name = layer.name
            layer_type = type(layer).__name__
            layer_shape = layer.compute_output_shape()
            layer_ct_params, layer_nt_params = layer.get_params_count()
            layer_shape_str = f"{(None, ) + layer_shape}"
            layer_name_str = f"{layer_name} ({layer_type})"
            print(
                f"{layer_name_str:29.29s}{layer_shape_str:26.26s}{str(layer_ct_params+layer_nt_params):10.10s}")
            total_ct_params += layer_ct_params
            total_nt_params += layer_nt_params
        print("=================================================================")
        print(f"Total params: {total_ct_params + total_nt_params}")
        print(f"Trainable params: {total_ct_params}")
        print(f"Non-trainable params: {total_nt_params}")
        print("_________________________________________________________________")
        print()

    def setup_equipment(self, thring, ccc):
        if type(thring) == str:
            return ccc(thring)
        else:
            return thring

    def compile(self, optimizer, loss, metric=None):
        self.optimizer = optimizers.get(optimizer)
        self.lossLayer = losses.get(loss)
        self.metric = metrics.get(metric) if metric else None

    def setup_data(self, data):
        return setup_data(data, device=self.device)

    def call(self, inputs, training=False):
        x = inputs
        for layers in self.layers:
            x_temp = []
            for layer in layers:
                x_temp.append(layer.call(x, training=training))
            x = x_temp
            if len(x) == 1:
                x = x[0]

        return x

    def predict(self, inputs, training=False):
        inputs = self.setup_data(inputs)
        output = self.call(inputs, training)
        output = restore_data(output)

        return output

    def forward(self, X, y, training=True):
        X, y = self.setup_data(X), self.setup_data(y)
        x = self.call(X, training)

        loss = self.lossLayer.call(x, y)
        loss = float(loss)

        metric = None
        if self.metric:
            metric = self.metric.call(x, y)
            metric = float(metric)

        return loss, metric

    def evaluate(self, X, y, batch_size=32, training=False):
        num_step = X.shape[0] // batch_size
        X, y = self.setup_data(X), self.setup_data(y)

        losses, metrics = [], []
        for i in range(num_step):
            X_step, y_step = X[batch_size * i: batch_size *
                               (i + 1)], y[batch_size * i: batch_size * (i + 1)]
            loss, metric = self.forward(X_step, y_step, training=training)
            losses.append(loss)
            metrics.append(metric)

        return mean(losses), mean(metrics)

    def backward(self, dout):
        for layers in reversed(self.layers):
            dout_temp = []
            for layer in layers:
                dout_temp.append(layer.backward(dout))
            dout = dout_temp
            if len(dout) == 1:
                dout = dout[0]
        return dout

    def cal_gradient(self):
        dout = 1
        dout = self.lossLayer.backward(dout)

        self.backward(dout)

    def update(self):
        self.optimizer.update(self.trainable_weights)

    def train_on_batch(self, batch_X, batch_y):
        batch_loss, batch_metric = self.forward(
            batch_X, batch_y, training=True)
        self.cal_gradient()
        self.update()

        return batch_loss, batch_metric

    def set_up_history(self):
        self.history = {'loss': [], 'val_loss': []}
        if self.metric:
            self.history[self.metric.nickname] = []
            self.history['val_' + self.metric.nickname] = []

    def fit(self, X, y, batch_size, epochs, validation_data, verbose=1):
        self.set_up_history()
        num_of_samples = X.shape[0]

        X_valid, y_valid = validation_data
        num_of_validate = X_valid.shape[0]

        X, y = self.setup_data(X), self.setup_data(y)
        X_valid, y_valid = self.setup_data(X_valid), self.setup_data(y_valid)

        print(
            f'Train on {num_of_samples} samples, validate on {num_of_validate} samples.')

        for n_epoch in range(epochs):
            if verbose:
                print(f'Epoch {n_epoch+1}/{epochs}')
            order = np.random.permutation(num_of_samples)

            batch_losses, batch_metrics = [], []
            pbar = range(0, num_of_samples, batch_size)
            if verbose:
                pbar = tqdm(pbar, ncols=120, ascii=True, unit='batches')
            for n_batch in pbar:
                data_slice = order[n_batch:n_batch + batch_size]
                X_train, y_train = X[data_slice], y[data_slice]
                batch_loss, batch_metric = self.train_on_batch(
                    X_train, y_train)
                batch_losses.append(batch_loss)
                batch_metrics.append(batch_metric)

                if verbose and n_batch % 10 == 0:
                    pbar.set_description(f"{n_batch}/{num_of_samples}")
                    pbar.set_postfix(loss=f'{batch_loss:.4f}',
                                     metric=f'{batch_metric:.4f}')

            train_loss = mean(batch_losses)
            train_metric = mean(batch_metrics)
            valid_loss, valid_metric = self.evaluate(X_valid, y_valid, batch_size=batch_size)

            self.history['loss'].append(train_loss)
            self.history[self.metric.nickname].append(train_metric)
            self.history['val_loss'].append(valid_loss)
            self.history['val_' + self.metric.nickname].append(valid_metric)
            if verbose:
                print(
                    f'loss: {train_loss:.4f} - metric: {train_metric:.4f} - val_loss: {valid_loss:.4f} - valid_metric: {valid_metric:.4f}')

        return self.history
