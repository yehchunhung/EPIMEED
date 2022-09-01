import tensorflow as tf

class LinearDecaySchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, peak_lr, total_steps, warmup_steps):
        super().__init__()

        self.peak_lr = peak_lr
        self.total_steps = total_steps
        self.warmup_steps = warmup_steps

    def __call__(self, step):
        arg1 = step / self.warmup_steps
        arg2 = (self.total_steps - step) / (self.total_steps - self.warmup_steps)
        return self.peak_lr * tf.math.minimum(arg1, arg2)

class PlateauSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, peak_lr, warmup_steps):
        super().__init__()

        self.peak_lr = peak_lr
        self.warmup_steps = warmup_steps

    def __call__(self, step):
        ratio = step / self.warmup_steps
        return self.peak_lr * tf.math.minimum(ratio, 1.0)
