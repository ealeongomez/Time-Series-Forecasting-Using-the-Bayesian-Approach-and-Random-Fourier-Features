import tensorflow as tf
import numpy as np


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

class Conv1dRFF(tf.keras.layers.Layer):

    # Contructor
    def __init__(self, output_dim, kernel_size=3, scale=None, padding='VALID', data_format='NWC', normalization=True, function=True,
                 trainable_scale=False, trainable_W=False,
                 seed=None, kernel='gaussian',
                 **kwargs):

        super(Conv1dRFF, self).__init__(**kwargs)

        self.output_dim=output_dim                  # Output dimension
        self.kernel_size=kernel_size                # Convolutional operation size
        self.scale=scale                            # Kernel gaussian
        self.padding=padding                        #
        self.data_format=data_format                # Format of operation convolutional
        self.normalization=normalization,           #
        self.function=function                      # sine or cosine
        self.trainable_scale=trainable_scale        #
        self.trainable_W=trainable_W                #
        self.seed=seed                              # Type of kernel
        self.initializer=kernel

    # ----------------------------------------------------------------------
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
            'trainable': self.trainable,
            'trainable_scale':self.trainable_scale,
            'trainable_W':self.trainable_W,
            'seed':self.seed
        })
        return config

    # ----------------------------------------------------------------------
    def build(self, input_shape):

        input_dim = input_shape[-1]
        #kernel_initializer = tf.random_normal_initializer(stddev=1.0)

        kernel_initializer = _get_random_features_initializer(self.initializer,
                                                              shape=(self.kernel_size,self.kernel_size,
                                                              input_dim,
                                                              self.output_dim),
                                                              seed=self.seed)

        # Inicializador de los valores de la operación convolucional (kernel)
        self.kernel = self.add_weight(
            name='kernel',
            shape=(self.kernel_size, input_shape[-1], self.output_dim),
            dtype=tf.float32,
            initializer=kernel_initializer,
            trainable=self.trainable_W
        )
        # Incilización de los pesos del bias
        """
          Pendiente de revisar la configuración del parametro initializer
        """
        self.bias = self.add_weight(
            name='bias',
            shape=(self.output_dim,),
            dtype=tf.float32,
            initializer=tf.random_uniform_initializer(minval=0.0, maxval=2*np.pi, seed=self.seed),
            trainable=self.trainable_W
        )
        # Inicializador de ancho de banda del kernel
        if not self.scale:
            if  self.initializer == 'gaussian':
                self.scale = np.sqrt((input_dim*self.kernel_size**2)/2.0)
                #print(self.scale)
            elif self.initializer == 'laplacian':
                self.scale = 1.0
            else:
                raise ValueError(f'Unsupported kernel initializer {self.initializer}')
        #
        self.kernel_scale = self.add_weight(
            name='kernel_scale',
            shape=(1,),
            dtype=tf.float32,
            initializer=tf.compat.v1.constant_initializer(self.scale),
            trainable=self.trainable_scale,
            constraint=tf.keras.constraints.NonNeg()
        )

    # ----------------------------------------------------------------------
    def call(self, inputs):

        scale = tf.math.divide(1.0, self.kernel_scale)
        kernel = tf.math.multiply(scale, self.kernel)

        inputs = tf.convert_to_tensor(inputs, dtype=tf.float32)
        inputs = tf.cast(inputs, tf.float32)

        outputs = tf.nn.conv1d(inputs, kernel, stride=1, padding=self.padding, data_format=self.data_format)
        outputs = tf.nn.bias_add(outputs, self.bias)

        if self.normalization:
            if self.function:
                outputs = tf.math.multiply(tf.math.sqrt(2/self.output_dim),tf.cos(outputs))
            else:
                outputs = tf.where(tf.equal(tf.math.mod(outputs, 2), 0), tf.math.multiply(tf.math.sqrt(2/self.output_dim), tf.cos(outputs)), tf.math.multiply(tf.math.sqrt(2/self.output_dim), tf.sin(outputs)))
        else:
            if self.function:
                outputs = tf.cos(outputs)
            else:
                outputs = tf.where(tf.equal(tf.math.mod(outputs, 2), 0), tf.cos(outputs), tf.math.sqrt(2/self.output_dim), tf.sin(outputs))

        return outputs
