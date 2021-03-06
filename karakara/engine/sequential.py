import pickle
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

    def build(self, input_shape=None, **kwargs):
        if not self.built:
            self.output_shape = input_shape
            for layers in self.layers:
                output_shapes = []
                for layer in layers:
                    layer.build(self.output_shape)
                    self.trainable_weights.extend(layer.get_trainable_weights())
                    self.non_trainable_weights.extend(layer.get_non_trainable_weights())
                    output_shapes.append(layer.compute_output_shape())

                self.output_shape = output_shapes
                if len(output_shapes) == 1:
                    self.output_shape = self.output_shape[0]

        self.built = True

    def compute_output_shape(self):
        return self.output_shape

    def add(self, layers):
        if not isinstance(layers, (list, tuple)):
            layers = [layers]

        self.layers.append(layers)

    def summary(self):
        self.build()
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
                f"{layer_name_str:29.28s}{layer_shape_str:26.25s}{str(layer_ct_params+layer_nt_params):10.10s}")
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
        self.build()
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
            metric = self.metric.call(y_pred=x, y_true=y)
            metric = float(metric)

        return loss, metric

    def evaluate(self, X, y, batch_size=32, training=False):
        num_step = X.shape[0] // batch_size + bool(X.shape[0] % batch_size)
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

            avg_batch_loss, avg_batch_metrics = 0, 0
            pbar = range(0, num_of_samples, batch_size)
            if verbose:
                pbar = tqdm(pbar, ncols=100, ascii='.>>>>>>>>>>>>=', unit='bs',
                            bar_format='{desc}[{bar}] - ETA: {remaining_s:3.1f}s - {rate_fmt}{postfix}', leave=False)
            for index, n_batch in enumerate(pbar):
                data_slice = order[n_batch:n_batch + batch_size]
                X_train, y_train = X[data_slice], y[data_slice]
                batch_loss, batch_metric = self.train_on_batch(
                    X_train, y_train)
                avg_batch_loss = (index * avg_batch_loss + batch_loss) / (index + 1)
                avg_batch_metrics = (index * avg_batch_metrics + batch_metric) / (index + 1)

                if verbose and n_batch % 10 == 0:
                    pbar.set_description(f"{n_batch:6d}/{num_of_samples}")
                    pbar.set_postfix(loss=f'{avg_batch_loss:.4f}',
                                     metric=f'{avg_batch_metrics:.4f}')

            if verbose:
                pbar.bar_format = '{desc}[{bar}] - USE: {elapsed_s:3.1f}s - {rate_fmt}{postfix}'
                pbar.display()
                print()

            train_loss = avg_batch_loss
            train_metric = avg_batch_metrics
            valid_loss, valid_metric = self.evaluate(X_valid, y_valid, batch_size=batch_size)

            self.history['loss'].append(train_loss)
            self.history[self.metric.nickname].append(train_metric)
            self.history['val_loss'].append(valid_loss)
            self.history['val_' + self.metric.nickname].append(valid_metric)
            if verbose:
                print(
                    f'loss: {train_loss:.4f} - metric: {train_metric:.4f} - val_loss: {valid_loss:.4f} - valid_metric: {valid_metric:.4f}')

        return self.history

    def fit_torchvision(self, training_data, batch_size, epochs, validation_data, callbacks=None, verbose=1):
        self.set_up_history()
        num_of_samples = len(training_data)
        num_of_validate = len(validation_data)

        print(f'Train on {num_of_samples} samples, validate on {num_of_validate} samples.')

        for n_epoch in range(1, epochs + 1):
            self.n_epoch = n_epoch
            if verbose:
                print(f'Epoch {n_epoch}/{epochs}')

            if callbacks:
                for callback in callbacks:
                    callback(self)

            X, y, X_valid, y_valid = [], [], [], []

            for image, label in training_data:
                X.append(image.numpy())
                y.append(label)

            for image, label in validation_data:
                X_valid.append(image.numpy())
                y_valid.append(label)

            X, y = self.setup_data(X), self.setup_data(y).astype('int')
            X_valid, y_valid = self.setup_data(X_valid), self.setup_data(y_valid).astype('int')

            order = np.random.permutation(num_of_samples)

            avg_batch_loss, avg_batch_metrics = 0, 0
            pbar = range(0, num_of_samples, batch_size)
            if verbose:
                pbar = tqdm(pbar, ncols=100, ascii='.>>>>>>>>>>>>=', unit='bs',
                            bar_format='{desc}[{bar}] - ETA: {remaining_s:3.1f}s - {rate_fmt}{postfix}', leave=False)
            for index, n_batch in enumerate(pbar):
                data_slice = order[n_batch:n_batch + batch_size]
                X_train, y_train = X[data_slice], y[data_slice]
                batch_loss, batch_metric = self.train_on_batch(
                    X_train, y_train)
                avg_batch_loss = (index * avg_batch_loss + batch_loss) / (index + 1)
                avg_batch_metrics = (index * avg_batch_metrics + batch_metric) / (index + 1)

                if verbose and n_batch % 10 == 0:
                    pbar.set_description(f"{n_batch:6d}/{num_of_samples}")
                    pbar.set_postfix(loss=f'{avg_batch_loss:.4f}',
                                     metric=f'{avg_batch_metrics:.4f}')

            pbar.bar_format = '{desc}[{bar}] - USE: {elapsed_s:3.1f}s - {rate_fmt}{postfix}'
            pbar.display()
            print()

            train_loss = avg_batch_loss
            train_metric = avg_batch_metrics
            valid_loss, valid_metric = self.evaluate(X_valid, y_valid, batch_size=batch_size)

            self.history['loss'].append(train_loss)
            self.history[self.metric.nickname].append(train_metric)
            self.history['val_loss'].append(valid_loss)
            self.history['val_' + self.metric.nickname].append(valid_metric)
            if verbose:
                print(
                    f'loss: {train_loss:.4f} - metric: {train_metric:.4f} - val_loss: {valid_loss:.4f} - valid_metric: {valid_metric:.4f}')

        return self.history

    def fit_dataloader(self, training_data, batch_size, epochs, validation_data=None, callbacks=None, verbose=1):
        self.set_up_history()
        num_of_samples = len(training_data)
        if validation_data is not None:
            num_of_validate = len(validation_data)

        if validation_data is not None:
            print(f'Train on {num_of_samples} samples, validate on {num_of_validate} samples.')
        else:
            print(f'Train on {num_of_samples} samples.')

        for n_epoch in range(1, epochs + 1):
            self.n_epoch = n_epoch
            if verbose:
                print(f'Epoch {n_epoch}/{epochs}')

            if callbacks:
                for callback in callbacks:
                    callback(self)

            avg_batch_loss, avg_batch_metrics = 0, 0
            pbar = range(0, num_of_samples)
            if verbose:
                pbar = tqdm(pbar, ncols=100, ascii='.>>>>>>>>>>>>=', unit='bs',
                            bar_format='{desc}[{bar}] - ETA: {remaining_s:3.1f}s - {rate_fmt}{postfix}', leave=False)
            for index, (n_batch, sample) in enumerate(zip(pbar, training_data)):
                X_train, y_train = self.setup_data(sample[0]), self.setup_data(sample[1])
                batch_loss, batch_metric = self.train_on_batch(
                    X_train, y_train)
                avg_batch_loss = (index * avg_batch_loss + batch_loss) / (index + 1)
                if self.metric:
                    avg_batch_metrics = (index * avg_batch_metrics + batch_metric) / (index + 1)

                if verbose:
                    pbar.set_description(f"{n_batch:6d}/{num_of_samples}")
                    pbar.set_postfix(loss=f'{avg_batch_loss:.4f}')
                    if self.metric is not None:
                        pbar.set_postfix(loss=f'{avg_batch_loss:.4f}', metric=f'{avg_batch_metrics:.4f}')

            pbar.bar_format = '{desc}[{bar}] - USE: {elapsed_s:3.1f}s - {rate_fmt}{postfix}'
            pbar.display()
            print()

            train_loss = avg_batch_loss
            train_metric = avg_batch_metrics
            if validation_data is not None:
                valid_loss, valid_metric = self.evaluate(X_valid, y_valid, batch_size=batch_size)

            self.history['loss'].append(train_loss)
            if self.metric is not None:
                self.history[self.metric.nickname].append(train_metric)
            if validation_data is not None:
                self.history['val_loss'].append(valid_loss)
                self.history['val_' + self.metric.nickname].append(valid_metric)
            # if verbose:
            #     print(
            #         f'loss: {train_loss:.4f} - metric: {train_metric:.4f} - val_loss: {valid_loss:.4f} - valid_metric: {valid_metric:.4f}')

        return self.history

    def save(self, filename):
        with open(filename, 'wb') as file:
            pickle.dump(self, file)


def load_model(filename):
    with open(filename, 'rb') as file:
        return pickle.load(file)
