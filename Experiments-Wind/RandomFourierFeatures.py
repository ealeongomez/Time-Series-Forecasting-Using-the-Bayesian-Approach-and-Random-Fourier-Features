

# Devuelve el código que permite inicializarlos
def _get_random_features_initializer(initializer, shape,seed):
    def _get_cauchy_samples(loc, scale, shape):
        np.random.seed(seed)
        probs = np.random.uniform(low=0., high=1., size=shape)
        return loc + scale * np.tan(np.pi * (probs - 0.5))

    if isinstance(initializer,str):
        if initializer == "gaussian":
            return tf.keras.initializers.RandomNormal(stddev=1.0,seed=seed)
        elif initializer == "laplacian":
            return tf.keras.initializers.Constant(
                _get_cauchy_samples(loc=0.0, scale=1.0, shape=shape))
        else:
            raise ValueError(f'Unsupported kernel initializer {initializer}')


import tensorflow as tf
import numpy as np

class Conv1dRFF(tf.keras.layers.Layer):

    def __init__(self, output_dim, kernel_size=3, scale=None, padding='VALID', data_format='NWC', normalization=True, function=True,
                 trainable_scale=False, trainable_W=False, seed=None, kernel='gaussian', **kwargs):
        super(Conv1dRFF, self).__init__(**kwargs)
        self.output_dim = output_dim                   # Output dimension
        self.kernel_size = kernel_size                 # Convolutional operation size
        self.scale = scale                             # Kernel scale
        self.padding = padding                         # Padding type
        self.data_format = data_format                 # Convolutional data format
        self.normalization = normalization             # Normalization flag
        self.function = function                       # Sine or cosine function
        self.trainable_scale = trainable_scale         # Scale trainability
        self.trainable_W = trainable_W                 # Kernel weights trainability
        self.seed = seed                               # Random seed
        self.initializer = kernel                      # Kernel initializer

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'output_dim': self.output_dim,
            'kernel_size': self.kernel_size,
            'scale': self.scale,
            'padding': self.padding,
            'data_format': self.data_format,
            'normalization': self.normalization,
            'function': self.function,
            'trainable_scale': self.trainable_scale,
            'trainable_W': self.trainable_W,
            'seed': self.seed,
            'initializer': self.initializer
        })
        return config

    def build(self, input_shape):
        input_dim = input_shape[-1]

        kernel_initializer = _get_random_features_initializer(self.initializer,
                                                              shape=(self.kernel_size, self.kernel_size,
                                                                     input_dim, self.output_dim),
                                                              seed=self.seed)

        self.kernel = self.add_weight(
            name='kernel',
            shape=(self.kernel_size, input_dim, self.output_dim),
            dtype=tf.float32,
            initializer=kernel_initializer,
            trainable=self.trainable_W
        )

        self.bias = self.add_weight(
            name='bias',
            shape=(self.output_dim,),
            dtype=tf.float32,
            initializer=tf.random_uniform_initializer(minval=0.0, maxval=2 * np.pi, seed=self.seed),
            trainable=self.trainable_W
        )

        # Set `scale` correctly based on the kernel type if not provided
        if not self.scale:
            if self.initializer == 'gaussian':
                self.scale = np.sqrt((input_dim * self.kernel_size ** 2) / 2.0)
            elif self.initializer == 'laplacian':
                self.scale = 1.0
            else:
                raise ValueError(f'Unsupported kernel initializer {self.initializer}')

        # Initialize `kernel_scale` as a constant tensor
        self.kernel_scale = self.add_weight(
            name='kernel_scale',
            shape=(1,),
            dtype=tf.float32,
            initializer=tf.constant_initializer(self.scale),
            trainable=self.trainable_scale
        )

    def call(self, inputs):
        # Ensure that `self.kernel_scale` is a tensor before dividing
        scale = tf.math.divide(1.0, tf.cast(self.kernel_scale, tf.float32))
        kernel = tf.math.multiply(scale, self.kernel)

        inputs = tf.convert_to_tensor(inputs, dtype=tf.float32)
        outputs = tf.nn.conv1d(inputs, kernel, stride=1, padding=self.padding, data_format=self.data_format)
        outputs = tf.nn.bias_add(outputs, self.bias)

        # Apply normalization and function if enabled
        if self.normalization:
            normalization_factor = tf.math.sqrt(2 / self.output_dim)
            outputs = normalization_factor * (tf.cos(outputs) if self.function else tf.sin(outputs))
        else:
            outputs = tf.cos(outputs) if self.function else tf.sin(outputs)

        return outputs
