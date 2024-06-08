import tensorflow as tf
from keras.layers import Layer
from .base import LayerKAN
from ..ops.spline import fit_spline_coef
from ..ops.grid import build_adaptive_grid
from typing import Tuple, List, Any, Union, Callable


@tf.keras.saving.register_keras_serializable("tfkan")
class DenseKAN(Layer, LayerKAN):
    """
    DenseKAN Layer: A Keras layer implementing a spline-based dense layer with adaptive grid updates.

    The layer leverages B-spline basis functions to approximate non-linear functions over the input space.
    Grid updates and extensions are performed to adaptively refine the spline representation during training.

    Arguments:
        units (int): Number of output units.
        use_bias (bool): Whether to use a bias term. Default is True.
        grid_size (int): Initial size of the spline grid. Default is 5.
        spline_order (int): Order of the splines used. Default is 3.
        grid_range (Union[Tuple[float], List[float]]): Range of the grid. Default is (-1.0, 1.0).
        spline_initialize_stddev (float): Standard deviation for spline kernel initialization. Default is 0.1.
        basis_activation (Union[str, Callable]): Activation function for the basis functions. Default is "silu".
        dtype (tf.DType): Data type for the layer. Default is tf.float32.
        update_steps (int): Steps interval for updating the grid. Default is 25.
        extend_steps (int): Steps interval for extending the grid. Default is 200.
        l2_reg (float): L2 regularization parameter for spline coefficient fitting. Default is 0.01.
        fast (bool): Whether to use the fast version of the least squares solver. Default is False.
        **kwargs: Additional keyword arguments.

    References:
        - Zhou, Z. P., et al. "Spline-Based Activation Function in Neural Networks."

    Example:
        >>> layer = DenseKAN(units=10, grid_size=5, spline_order=3)
        >>> output = layer(inputs)
    """

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
        update_steps=25,
        extend_steps=200,
        l2_reg=0.01,
        fast=False,
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
        self.update_steps = update_steps
        self.extend_steps = extend_steps
        self.l2_reg = l2_reg
        self.fast = fast
        self.current_step = 0
        self._current_inputs = None

    def build(self, input_shape: Any):
        """
        Build the DenseKAN layer, initializing weights and grid.

        The grid is initialized over the range defined by `grid_range` and extended based on the input shape.
        The spline kernel and scale factors are also initialized.

        Arguments:
            input_shape (Any): Shape of the input tensor.

        Example:
            >>> layer.build((None, 10))
        """
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
        """
        Forward pass of the DenseKAN layer.

        The layer computes the spline-based transformation of the inputs and applies the optional bias.
        It also manages grid updates and extensions at specified intervals.

        Arguments:
            inputs (tf.Tensor): Input tensor.
            *args: Additional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            tf.Tensor: Output tensor after applying spline-based transformation and optional bias.

        Example:
            >>> output = layer.call(inputs)
        """
        self._current_inputs = inputs  # Store inputs for use in grid extension
        inputs, orig_shape = self._check_and_reshape_inputs(inputs)
        output_shape = tf.concat([orig_shape, [self.units]], axis=0)
        spline_out = self.calc_spline_output(inputs)
        spline_out += tf.expand_dims(self.basis_activation(inputs), axis=-1)
        spline_out *= tf.expand_dims(self.scale_factor, axis=0)
        spline_out = tf.reshape(tf.reduce_sum(spline_out, axis=-2), output_shape)
        if self.use_bias:
            spline_out += self.bias

        # Update or extend the grid as needed
        self.current_step += 1
        if self.current_step % self.update_steps == 0:
            self.update_grid_from_samples(inputs)
        if self.current_step % self.extend_steps == 0:
            self.extend_grid_from_samples(inputs, self.grid_size * 2)

        return spline_out

    def _check_and_reshape_inputs(self, inputs):
        """
        Check and reshape the inputs to ensure they are compatible with the layer.

        Arguments:
            inputs (tf.Tensor): Input tensor.

        Returns:
            Tuple[tf.Tensor, tf.Tensor]: Reshaped inputs and original shape.

        Example:
            >>> reshaped_inputs, orig_shape = layer._check_and_reshape_inputs(inputs)
        """
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

    @tf.function
    def update_grid_from_samples(
        self, inputs: tf.Tensor = None, margin: float = 0.01, grid_eps: float = 0.01
    ):
        """
        Update the spline grid using the current input samples.

        The grid update ensures that the spline functions are defined over the range of the input data.

        Arguments:
            inputs (tf.Tensor): Input tensor. If None, uses stored inputs.
            margin (float): Margin for grid update. Default is 0.01.
            grid_eps (float): Grid epsilon for adaptive grid building. Default is 0.01.

        Example:
            >>> layer.update_grid_from_samples(inputs)

        Formula:
            The grid update involves recalculating the grid points based on the current input data:
            grid_new = build_adaptive_grid(inputs, grid_size, spline_order, grid_eps, margin, dtype)
        """
        tf.print("Updating grid")
        if inputs is None:
            inputs = self._current_inputs  # Use stored inputs
        inputs, _ = self._check_and_reshape_inputs(inputs)
        spline_out = self.calc_spline_output(inputs)
        grid = build_adaptive_grid(
            inputs, self.grid_size, self.spline_order, grid_eps, margin, self.dtype
        )
        updated_kernel = fit_spline_coef(
            inputs, spline_out, grid, self.spline_order, self.l2_reg, self.fast
        )
        self.grid.assign(grid)
        self.spline_kernel.assign(updated_kernel)

    @tf.function
    def extend_grid_from_samples(
        self,
        inputs: tf.Tensor = None,
        extend_grid_size: int = None,
        margin: float = 0.01,
        grid_eps: float = 0.01,
        **kwargs,
    ):
        """
        Extend the spline grid using the current input samples.

        The grid extension refines the spline approximation by adding more grid points.

        Arguments:
            inputs (tf.Tensor): Input tensor. If None, uses stored inputs.
            extend_grid_size (int): New grid size after extension. If None, defaults to twice the current grid size.
            margin (float): Margin for grid extension. Default is 0.01.
            grid_eps (float): Grid epsilon for adaptive grid building. Default is 0.01.
            **kwargs: Additional keyword arguments.

        Example:
            >>> layer.extend_grid_from_samples(inputs)

        Formula:
            The grid extension involves adding new grid points and recalculating the spline coefficients:
            grid_new = build_adaptive_grid(inputs, extend_grid_size, spline_order, grid_eps, margin, dtype)
            updated_kernel = fit_spline_coef(inputs, spline_out, grid_new, spline_order, l2_reg, fast)
        """
        tf.print("Extending grid")
        if inputs is None:
            inputs = self._current_inputs  # Use stored inputs
        if extend_grid_size is None:
            extend_grid_size = self.grid_size * 2
        if extend_grid_size < self.grid_size:
            raise ValueError(
                f"expected extend_grid_size > grid_size, found {extend_grid_size} <= {self.grid_size}"
            )
        inputs, _ = self._check_and_reshape_inputs(inputs)
        spline_out = self.calc_spline_output(inputs)
        grid = build_adaptive_grid(
            inputs, extend_grid_size, self.spline_order, grid_eps, margin, self.dtype
        )
        l2_reg, fast = kwargs.pop("l2_reg", self.l2_reg), kwargs.pop("fast", self.fast)
        updated_kernel = fit_spline_coef(
            inputs, spline_out, grid, self.spline_order, l2_reg, fast
        )

        # Resize existing variables
        self.grid.assign(tf.cast(grid, dtype=self.dtype))
        self.spline_kernel.assign(updated_kernel)

        self.grid_size = extend_grid_size

    def get_config(self):
        """
        Returns the configuration of the DenseKAN layer.

        Returns:
            dict: Configuration dictionary.

        Example:
            >>> config = layer.get_config()
        """
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
                "update_steps": self.update_steps,
                "extend_steps": self.extend_steps,
                "l2_reg": self.l2_reg,
                "fast": self.fast,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        """
        Creates a DenseKAN layer from the given configuration.

        Arguments:
            config (dict): Configuration dictionary.

        Returns:
            DenseKAN: A new instance of DenseKAN.

        Example:
            >>> layer = DenseKAN.from_config(config)
        """
        if isinstance(config["basis_activation"], dict):
            config["basis_activation"] = tf.keras.activations.deserialize(
                config["basis_activation"]
            )
        return cls(**config)
