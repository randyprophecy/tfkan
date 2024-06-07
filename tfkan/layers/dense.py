import tensorflow as tf
from keras.layers import Layer
from .base import LayerKAN
from ..ops.spline import fit_spline_coef
from ..ops.grid import build_adaptive_grid
from typing import Tuple, List, Any, Union, Callable


@tf.keras.saving.register_keras_serializable("tfkan")
class DenseKAN(Layer, LayerKAN):
    def __init__(
        self,
        units: int,
        use_bias: bool = True,
        grid_size: int = 5,
        spline_order: int = 3,
        grid_range: Union[Tuple[float], List[float]] = (-1.0, 1.0),
        spline_initialize_stddev: float = 0.1,
        basis_activation: Union[str, Callable] = "silu",
        dtype=tf.float32,
        **kwargs,
    ):
        super(DenseKAN, self).__init__(dtype=dtype, **kwargs)
        self.units = units
        self.grid_size = grid_size
        self.spline_order = spline_order
        self.grid_range = grid_range
        self.basis_activation = basis_activation
        self.use_bias = use_bias
        self.spline_initialize_stddev = spline_initialize_stddev

    def build(self, input_shape: Any):
        if isinstance(input_shape, int):
            in_size = input_shape
        else:
            in_size = input_shape[-1]

        self.in_size = in_size
        self.spline_basis_size = self.grid_size + self.spline_order
        bound = self.grid_range[1] - self.grid_range[0]

        self.grid = tf.linspace(
            self.grid_range[0] - self.spline_order * bound / self.grid_size,
            self.grid_range[1] + self.spline_order * bound / self.grid_size,
            self.grid_size + 2 * self.spline_order + 1,
        )
        self.grid = tf.repeat(self.grid[None, :], in_size, axis=0)
        self.grid = tf.Variable(
            initial_value=tf.cast(self.grid, dtype=self.dtype),
            trainable=False,
            dtype=self.dtype,
            name="spline_grid",
        )

        self.spline_kernel = self.add_weight(
            name="spline_kernel",
            shape=(self.in_size, self.spline_basis_size, self.units),
            initializer=tf.keras.initializers.RandomNormal(
                stddev=self.spline_initialize_stddev
            ),
            trainable=True,
            dtype=self.dtype,
        )

        self.scale_factor = self.add_weight(
            name="scale_factor",
            shape=(self.in_size, self.units),
            initializer=tf.keras.initializers.GlorotUniform(),
            trainable=True,
            dtype=self.dtype,
        )

        if isinstance(self.basis_activation, str):
            self.basis_activation = tf.keras.activations.get(self.basis_activation)
        elif not callable(self.basis_activation):
            raise ValueError(
                f"expected basis_activation to be str or callable, found {type(self.basis_activation)}"
            )

        if self.use_bias:
            self.bias = self.add_weight(
                name="bias",
                shape=(self.units,),
                initializer=tf.keras.initializers.Zeros(),
                trainable=True,
                dtype=self.dtype,
            )
        else:
            self.bias = None

        self.built = True

    def call(self, inputs, *args, **kwargs):
        inputs, orig_shape = self._check_and_reshape_inputs(inputs)
        output_shape = tf.concat([orig_shape, [self.units]], axis=0)
        spline_out = self.calc_spline_output(inputs)
        spline_out += tf.expand_dims(self.basis_activation(inputs), axis=-1)
        spline_out *= tf.expand_dims(self.scale_factor, axis=0)
        spline_out = tf.reshape(tf.reduce_sum(spline_out, axis=-2), output_shape)
        if self.use_bias:
            spline_out += self.bias
        return spline_out

    def _check_and_reshape_inputs(self, inputs):
        shape = tf.shape(inputs)
        ndim = len(shape)
        if ndim < 2:
            raise ValueError(
                f"expected min_ndim=2, found ndim={ndim}. Full shape received: {shape}"
            )
        if inputs.shape[-1] != self.in_size:
            raise ValueError(
                f"expected last dimension of inputs to be {self.in_size}, found {shape[-1]}"
            )
        orig_shape = shape[:-1]
        inputs = tf.reshape(inputs, (-1, self.in_size))
        return inputs, orig_shape

    def update_grid_from_samples(
        self, inputs: tf.Tensor, margin: float = 0.01, grid_eps: float = 0.01
    ):
        inputs, _ = self._check_and_reshape_inputs(inputs)
        spline_out = self.calc_spline_output(inputs)
        grid = build_adaptive_grid(
            inputs, self.grid_size, self.spline_order, grid_eps, margin, self.dtype
        )
        updated_kernel = fit_spline_coef(inputs, spline_out, grid, self.spline_order)
        self.grid.assign(grid)
        self.spline_kernel.assign(updated_kernel)

    def extend_grid_from_samples(
        self,
        inputs: tf.Tensor,
        extend_grid_size: int,
        margin: float = 0.01,
        grid_eps: float = 0.01,
        **kwargs,
    ):
        if extend_grid_size < self.grid_size:
            raise ValueError(
                f"expected extend_grid_size > grid_size, found {extend_grid_size} <= {self.grid_size}"
            )
        inputs, _ = self._check_and_reshape_inputs(inputs)
        spline_out = self.calc_spline_output(inputs)
        grid = build_adaptive_grid(
            inputs, extend_grid_size, self.spline_order, grid_eps, margin, self.dtype
        )
        l2_reg, fast = kwargs.pop("l2_reg", 0), kwargs.pop("fast", True)
        updated_kernel = fit_spline_coef(
            inputs, spline_out, grid, self.spline_order, l2_reg, fast
        )
        delattr(self, "grid")
        self.grid = tf.Variable(
            initial_value=tf.cast(grid, dtype=self.dtype),
            trainable=False,
            dtype=self.dtype,
            name="spline_grid",
        )
        self.grid_size = extend_grid_size
        self.spline_basis_size = extend_grid_size + self.spline_order
        delattr(self, "spline_kernel")
        self.spline_kernel = self.add_weight(
            name="spline_kernel",
            shape=(self.in_size, self.spline_basis_size, self.units),
            initializer=tf.keras.initializers.Constant(updated_kernel),
            trainable=True,
            dtype=self.dtype,
        )

    def get_config(self):
        config = super(DenseKAN, self).get_config()
        config.update(
            {
                "units": self.units,
                "grid_size": self.grid_size,
                "spline_order": self.spline_order,
                "grid_range": self.grid_range,
                "basis_activation": (
                    self.basis_activation
                    if isinstance(self.basis_activation, str)
                    else tf.keras.activations.serialize(self.basis_activation)
                ),
                "use_bias": self.use_bias,
                "spline_initialize_stddev": self.spline_initialize_stddev,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        if isinstance(config["basis_activation"], dict):
            config["basis_activation"] = tf.keras.activations.deserialize(
                config["basis_activation"]
            )
        return cls(**config)
