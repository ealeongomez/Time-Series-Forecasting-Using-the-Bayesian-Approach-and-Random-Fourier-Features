
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

def _get_random_features_initializer(initializer, shape, seed=None):
    # Esta función debe retornar un arreglo numpy con la inicialización deseada.
    # Por ejemplo, si el initializer es 'gaussian':
    if seed is not None:
        np.random.seed(seed)
    if initializer == 'gaussian':
        # Por ejemplo, inicialización gaussiana estándar
        return np.random.randn(*shape)
    elif initializer == 'laplacian':
        # Por ejemplo, inicialización con distribución Laplaciana
        return np.random.laplace(loc=0.0, scale=1.0, size=shape)
    else:
        raise ValueError(f'Unsupported initializer {initializer}')



class Conv1dRFF_PT(nn.Module):
    def __init__(self, output_dim, kernel_size=3, scale=None, padding='VALID', normalization=True,
                 function=True, trainable_scale=False, trainable_W=False, seed=None, kernel='gaussian', **kwargs):
        super(Conv1dRFF_PT, self).__init__()
        self.output_dim = output_dim
        self.kernel_size = kernel_size
        self.scale = scale
        # Reemplazamos 'VALID' por 0 para no padding
        self.padding = 0 if padding == 'VALID' else padding
        self.normalization = normalization
        self.function = function
        self.trainable_scale = trainable_scale
        self.trainable_W = trainable_W
        self.seed = seed
        self.initializer = kernel

        # Parámetros se inicializarán perezosamente la primera vez que se llame a forward
        self.kernel = None
        self.bias = None
        self.kernel_scale = None

    def forward(self, inputs):
        # Si los parámetros no están inicializados, los inicializamos ahora
        if self.kernel is None:
            input_shape = inputs.shape  # (batch_size, in_channels, length)
            input_dim = input_shape[1]

            kernel_initializer = _get_random_features_initializer(
                self.initializer,
                shape=(self.output_dim, input_dim, self.kernel_size),
                seed=self.seed
            )

            # Crear el kernel en el mismo device que 'inputs'
            kernel_initializer = torch.tensor(
                kernel_initializer,
                dtype=torch.float32,
                device=inputs.device
            )

            self.kernel = nn.Parameter(kernel_initializer, requires_grad=self.trainable_W)

            # Crear el bias en el mismo device
            self.bias = nn.Parameter(
                torch.empty(self.output_dim, device=inputs.device).uniform_(0.0, 2 * np.pi),
                requires_grad=self.trainable_W
            )

            # Set scale if not provided
            if self.scale is None:
                if self.initializer == 'gaussian':
                    self.scale = np.sqrt((input_dim * (self.kernel_size ** 2)) / 2.0)
                elif self.initializer == 'laplacian':
                    self.scale = 1.0
                else:
                    raise ValueError(f'Unsupported kernel initializer {self.initializer}')

            # Crear la kernel_scale en el mismo device
            self.kernel_scale = nn.Parameter(
                torch.tensor([self.scale], dtype=torch.float32, device=inputs.device),
                requires_grad=self.trainable_scale
            )

        scale = 1.0 / self.kernel_scale
        kernel = scale * self.kernel

        # Aplicar la convolución 1D
        outputs = F.conv1d(inputs, kernel, bias=self.bias, stride=1, padding=self.padding)

        # Si normalization es True:
        if self.normalization:
            # Crear el factor de escala sqrt(2 / output_dim) en el device correcto
            scale_factor = torch.sqrt(torch.tensor(2.0 / self.output_dim,
                                                   dtype=outputs.dtype,
                                                   device=outputs.device))

            if self.function:
                # outputs = scale_factor * cos(outputs)
                outputs = scale_factor * torch.cos(outputs)
            else:
                # condition = (outputs % 2 == 0)
                condition = (outputs % 2 == 0)
                # outputs = where(condition, scale_factor * cos(outputs), scale_factor * sin(outputs))
                outputs = torch.where(condition,
                                      scale_factor * torch.cos(outputs),
                                      scale_factor * torch.sin(outputs))
        else:
            # Si normalization es False
            if self.function:
                # outputs = cos(outputs)
                outputs = torch.cos(outputs)
            else:
                # condition = (outputs % 2 == 0)
                condition = (outputs % 2 == 0)
                # outputs = where(condition, cos(outputs), sin(outputs))
                outputs = torch.where(condition, torch.cos(outputs), torch.sin(outputs))

        return outputs

