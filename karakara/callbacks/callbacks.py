class ReduceLROnPlateau:

    def __init__(self, monitor='val_loss', factor=0.1, patience=10,
                 verbose=0, mode='auto', min_delta=1e-4, cooldown=0, min_lr=0):
        self.monitor = monitor
        self.factor = factor
        self.patience = patience
        self.verbose = verbose
        self.mode = mode
        self.min_delta = min_delta
        self.cooldown = cooldown
        self.min_lr = min_lr

    def __call__(self, model):
        lr = model.optimizer.lr

        if len(model.history[self.monitor]) < (self.patience + 1):
            return

        if min(model.history[self.monitor][-self.patience:]) >= model.history[self.monitor][-(self.patience + 1)]:
            new_lr = max(self.min_lr, lr * self.factor)

            print(
                f'{self.monitor} not improve from {model.history[self.monitor][-(self.patience + 1)]} for {self.patience} epochs')
            print(f'Reduce learning rate from {lr} to {new_lr}')

            model.optimizer.lr = new_lr

        return lr
