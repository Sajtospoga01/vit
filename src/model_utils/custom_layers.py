import tensorflow as tf
import keras
from keras.layers import Layer, AveragePooling2D, Conv2D, Conv2DTranspose, Concatenate


class PyramidPoolingModule(Layer):
    """
    Custom implementation of a pyramid pooling module
    """

    def __init__(
            self, pool_sizes, kernels, num_channels, data_format="channels_last", **kwargs
    ):
        super(PyramidPoolingModule, self).__init__(**kwargs)
        self.pool_sizes = pool_sizes
        self.kernels = kernels
        self.num_channels = num_channels
        self.data_format = data_format
        self.pooling_layers = []
        self.conv_layers = []
        self.conv_transpose_layers = []
        self.cropping_layers = []

    def build(self, input_shape):
        i = 0
        for pool_size, kernel in zip(self.pool_sizes, self.kernels):
            self.pooling_layers.append(
                AveragePooling2D(
                    pool_size=(pool_size, pool_size),
                    strides=(pool_size, pool_size),
                    padding="valid",
                    data_format=self.data_format,
                )
            )
            self.conv_layers.append(
                Conv2D(
                    input_shape[-1],
                    kernel,
                    padding="same",
                    data_format=self.data_format,
                    dilation_rate=pool_size,
                )
            )

            self.conv_transpose_layers.append(
                Conv2DTranspose(
                    input_shape[-1],
                    kernel,
                    strides=(pool_size, pool_size),
                    padding="valid",
                    data_format=self.data_format,
                )
            )

            crop = (0, 0)
            if i == 1:
                crop = (0, 1)
            else:
                crop = (0, 0)
            print(crop)
            self.cropping_layers.append(
                keras.layers.Cropping2D(
                    cropping=(crop, crop),
                    data_format=self.data_format,
                )
            )
            i += 1

    def call(self, x):
        input_shape = tf.shape(x)
        h, w = input_shape[1], input_shape[2]
        pyramid_features = [x]
        for pooling_layer, conv_layer, conv_transpose_layer, cropping_layer in zip(
                self.pooling_layers,
                self.conv_layers,
                self.conv_transpose_layers,
                self.cropping_layers,
        ):
            x = pooling_layer(x)
            x = conv_layer(x)
            x = conv_transpose_layer(x)
            x = cropping_layer(x)

            pyramid_features.append(x)

        output = Concatenate(axis=-1)(pyramid_features)
        return output

    def get_config(self):
        config = super(PyramidPoolingModule, self).get_config()
        config.update(
            {
                "pool_sizes": self.pool_sizes,
                "kernels": self.kernels,
                "num_channels": self.num_channels,
                "data_format": self.data_format,
            }
        )
        return config
