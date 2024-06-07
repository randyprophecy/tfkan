import tensorflow as tf
from tfkan.layers.dense import DenseKAN


@tf.keras.saving.register_keras_serializable("tfkan")
class GridExtensionCallback(tf.keras.callbacks.Callback):
    def __init__(
        self,
        extend_steps=200,
        margin=0.01,
        grid_eps=0.01,
        extend_factor=2,
        inputs_index=None,  # Index to specify which input to use if multiple inputs
    ):
        super().__init__()
        self.extend_steps = extend_steps
        self.margin = margin
        self.grid_eps = grid_eps
        self.extend_factor = extend_factor
        self.inputs_index = inputs_index
        self.current_step = 0

    def on_train_batch_begin(self, batch, logs=None):
        self.inputs = None

    def on_train_batch_end(self, batch, logs=None):
        self.current_step += 1
        if self.current_step % self.extend_steps == 0:
            # Access the input data directly
            if self.inputs_index is None:
                self.inputs = self.model.inputs
            else:
                self.inputs = self.model.inputs[self.inputs_index]

            for layer in self.model.layers:
                if isinstance(layer, DenseKAN):
                    new_grid_size = min(
                        layer.grid_size * self.extend_factor, layer.grid_size * 2
                    )
                    if new_grid_size > layer.grid_size:
                        print(f"Extending grid for layer {layer.name} to {new_grid_size}")
                        layer.extend_grid_from_samples(
                            self.inputs, new_grid_size, self.margin, self.grid_eps
                        )

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "extend_steps": self.extend_steps,
                "margin": self.margin,
                "grid_eps": self.grid_eps,
                "extend_factor": self.extend_factor,
                "inputs_index": self.inputs_index,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


@tf.keras.saving.register_keras_serializable("tfkan")
class GridUpdateCallback(tf.keras.callbacks.Callback):
    def __init__(self, update_steps=200, margin=0.01, grid_eps=0.01, inputs_index=None):
        super().__init__()
        self.update_steps = update_steps
        self.margin = margin
        self.grid_eps = grid_eps
        self.current_step = 0
        self.inputs_index = inputs_index

    def on_train_batch_begin(self, batch, logs=None):
        self.inputs = None

    def on_train_batch_end(self, batch, logs=None):
        self.current_step += 1
        if self.current_step % self.update_steps == 0:
            # Access the input data directly
            if self.inputs_index is None:
                self.inputs = self.model.inputs
            else:
                self.inputs = self.model.inputs[self.inputs_index]

            for layer in self.model.layers:
                if isinstance(layer, DenseKAN):
                    print(f"Updating grid for layer {layer.name}")
                    layer.update_grid_from_samples(
                        self.inputs, self.margin, self.grid_eps
                    )

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "update_steps": self.update_steps,
                "margin": self.margin,
                "grid_eps": self.grid_eps,
                "inputs_index": self.inputs_index,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)