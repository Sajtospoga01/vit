from src.model_utils.base_model import ModelGenerator
from keras.models import Model

import tensorflow as tf
from keras.layers import (
    Layer,
    Conv2D,
    BatchNormalization,
    Activation,
    AveragePooling2D,
    MaxPooling2D,
    Conv2DTranspose,
    Concatenate,
    Input,
    Dropout,
    ZeroPadding2D,
    Concatenate,
    ReLU,
    ZeroPadding2D,
    UpSampling2D,
)
import keras
from src.model_utils.custom_layers import PyramidPoolingModule


class VGG16_UNET_SAPPM(ModelGenerator):
    """
    Custom class defining a VGG16 Unet forwad pass
    """

    pretrained_url = (
        "https://github.com/fchollet/deep-learning-models/"
        "releases/download/v0.1/"
        "vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5"
    )
    pretrained_url_top = "https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5"
    VGG_Weights_path = tf.keras.utils.get_file(
        pretrained_url.split("/")[-1], pretrained_url
    )

    def __init__(
            self,
            in_shape,
            n_classes,
            image_ordering="channels_last",
            dropouts=[0, 0, 0, 0, 0, 0, 0, 0, 0],
    ):
        """
        Initializes a VGG16 Unet Segmentation Class

        Parameters
        ----------
        input_shape (tuple): Input shape of the image
        n_classes (int): Number of classes to be segmented

        Returns
        -------
        None
        """

        self.n_classes = n_classes
        self.IMAGE_ORDERING = image_ordering

        self.dropouts = dropouts

        # fmt: off
        MERGE_AXIS = -1

        img_input = Input(shape=in_shape)

        x = tf.keras.applications.vgg16.preprocess_input(img_input)
        x = Conv2D(64, (3, 3), padding='same', name='block1_conv1', data_format=self.IMAGE_ORDERING)(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)
        x = Conv2D(64, (3, 3), padding='same', name='block1_conv2', data_format=self.IMAGE_ORDERING)(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)
        f1 = x

        x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_avg_pool', data_format=self.IMAGE_ORDERING)(x)
        x = Dropout(self.dropouts[0])(x)


        # Block 2
        x = Conv2D(128, (3, 3), padding='same', name='block2_conv1', data_format=self.IMAGE_ORDERING)(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)
        x = Conv2D(128, (3, 3), padding='same', name='block2_conv2', data_format=self.IMAGE_ORDERING)(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)
        f2 = x

        x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_avg_pool', data_format=self.IMAGE_ORDERING)(x)
        x = Dropout(self.dropouts[1])(x)


        # Block 3
        x = Conv2D(256, (3, 3), padding='same', name='block3_conv1', data_format=self.IMAGE_ORDERING)(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)
        x = Conv2D(256, (3, 3), padding='same', name='block3_conv2', data_format=self.IMAGE_ORDERING)(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)
        x = Conv2D(256, (3, 3), padding='same', name='block3_conv3', data_format=self.IMAGE_ORDERING)(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)
        f3 = x

        x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_avg_pool', data_format=self.IMAGE_ORDERING)(x)
        x = Dropout(self.dropouts[2])(x)


        # Block 4
        x = Conv2D(512, (3, 3), padding='same', name='block4_conv1', data_format=self.IMAGE_ORDERING)(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)
        x = Conv2D(512, (3, 3), padding='same', name='block4_conv2', data_format=self.IMAGE_ORDERING)(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)
        x = Conv2D(512, (3, 3), padding='same', name='block4_conv3', data_format=self.IMAGE_ORDERING)(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)
        f4 = x

        x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_avg_pool', data_format=self.IMAGE_ORDERING)(x)
        x = Dropout(self.dropouts[3])(x)

        # Block 5 pyramid pooling block

        x = Conv2D(512, (3, 3), padding='same', name='block5_conv1', data_format=self.IMAGE_ORDERING)(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)
        x = Conv2D(512, (3, 3), padding='same', name='block5_conv2', data_format=self.IMAGE_ORDERING)(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)
        x = Conv2D(512, (3, 3), padding='same', name='block5_conv3', data_format=self.IMAGE_ORDERING)(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)
        f5 = x

        f5 = PyramidPoolingModule([1,2,4,8],[(1,1),(3,3),(3,3),(3,3)],num_channels=self.n_classes)(f5)

        o = f5
        o = (ZeroPadding2D((1, 1), data_format=self.IMAGE_ORDERING))(o)
        o = (Conv2D(512, (3, 3), padding='valid', data_format=self.IMAGE_ORDERING))(o)
        o = (BatchNormalization())(o)
        o = (Activation('relu'))(o)
        o = (Conv2D(512, (3, 3), padding='same', data_format=self.IMAGE_ORDERING))(o)
        o = (BatchNormalization())(o)
        o = (Activation('relu'))(o)
        o = (Dropout(self.dropouts[4]))(o)

        o = (Conv2DTranspose(512,(2, 2), strides = (2,2), data_format=self.IMAGE_ORDERING))(o)
        o = (Concatenate(axis=MERGE_AXIS)([o, f4]))
        o = (ZeroPadding2D((1, 1), data_format=self.IMAGE_ORDERING))(o)
        o = (Conv2D(256, (3, 3), padding='valid', data_format=self.IMAGE_ORDERING))(o)
        o = (BatchNormalization())(o)
        o = (Activation('relu'))(o)
        o = (Conv2D(256, (3, 3), padding='same', data_format=self.IMAGE_ORDERING))(o)
        o = (BatchNormalization())(o)
        o = (Activation('relu'))(o)
        o = (Dropout(self.dropouts[5]))(o)

        o = (Conv2DTranspose(256,(2, 2),strides = (2,2), data_format=self.IMAGE_ORDERING))(o)
        o = (Concatenate(axis=MERGE_AXIS)([o, f3]))
        o = (ZeroPadding2D((1, 1), data_format=self.IMAGE_ORDERING))(o)
        o = (Conv2D(128, (3, 3), padding='valid', data_format=self.IMAGE_ORDERING))(o)
        o = (BatchNormalization())(o)
        o = (Activation('relu'))(o)
        o = (Conv2D(128, (3, 3), padding='same', data_format=self.IMAGE_ORDERING))(o)
        o = (BatchNormalization())(o)
        o = (Activation('relu'))(o)
        o = (Dropout(self.dropouts[6]))(o)

        o = (Conv2DTranspose (128,(2, 2),strides = (2,2), data_format=self.IMAGE_ORDERING))(o)
        o = (Concatenate(axis=MERGE_AXIS)([o, f2]))
        o = (ZeroPadding2D((1, 1), data_format=self.IMAGE_ORDERING))(o)
        o = (Conv2D(64, (3, 3), padding='valid', data_format=self.IMAGE_ORDERING))(o)
        o = (BatchNormalization())(o)
        o = (Activation('relu'))(o)
        o = (Conv2D(64, (3, 3), padding='same', data_format=self.IMAGE_ORDERING))(o)
        o = (BatchNormalization())(o)
        o = (Activation('relu'))(o)
        o = (Dropout(self.dropouts[7]))(o)

        o = (Conv2DTranspose (64,(2, 2),strides = (2,2), data_format=self.IMAGE_ORDERING))(o)
        o = (Concatenate(axis=MERGE_AXIS)([o, f1]))
        o = (ZeroPadding2D((1, 1), data_format=self.IMAGE_ORDERING))(o)
        o = (Conv2D(32, (3, 3), padding='valid', data_format=self.IMAGE_ORDERING))(o)
        o = (BatchNormalization())(o)
        o = (Activation('relu'))(o)
        o = (Conv2D(32, (3, 3), padding='same', data_format=self.IMAGE_ORDERING))(o)
        o = (BatchNormalization())(o)
        o = (Activation('relu'))(o)
        o = (Dropout(self.dropouts[8]))(o)

        o = Conv2D(self.n_classes, (1, 1), padding='same',name="logit_layer", data_format=self.IMAGE_ORDERING)(o)

        o = (Activation('softmax'))(o)

        super(VGG16_UNET_SAPPM,self).__init__(inputs=img_input, outputs=o, name="VGG16_UNET_WITH_PYRAMID_POOLING")
        self.in_shape = in_shape
        self.n_classes = n_classes
        self.image_ordering = image_ordering
        self.dropouts = dropouts


    def get_config(self):
        config = super(VGG16_UNET, self).get_config()
        config.update({
            'in_shape': self.in_shape,
            'n_classes': self.n_classes,
            'image_ordering': self.image_ordering,
            'dropouts': self.dropouts,
        })
        return config

    @classmethod
    def from_config(cls, config):
        print(config)
        return cls(**config)

class VGG16_UNET(ModelGenerator):
    """
    Custom class defining a VGG16 Unet forwad pass
    """

    pretrained_url = (
        "https://github.com/fchollet/deep-learning-models/"
        "releases/download/v0.1/"
        "vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5"
    )
    pretrained_url_top = "https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5"
    VGG_Weights_path = tf.keras.utils.get_file(
        pretrained_url.split("/")[-1], pretrained_url
    )

    def __init__(
            self,
            in_shape,
            n_classes,
            image_ordering="channels_last",
            dropouts=[0, 0, 0, 0, 0, 0, 0, 0, 0],
            **kwargs,
    ):
        """
        Initializes a VGG16 Unet Segmentation Class

        Parameters
        ----------
        input_shape (tuple): Input shape of the image
        n_classes (int): Number of classes to be segmented

        Returns
        -------
        None
        """

        self.n_classes = n_classes
        self.IMAGE_ORDERING = image_ordering

        self.dropouts = dropouts

        # fmt: off
        MERGE_AXIS = -1

        img_input = Input(shape=in_shape)

        x = tf.keras.applications.vgg16.preprocess_input(img_input)
        x = Conv2D(64, (3, 3), padding='same', name='block1_conv1', data_format=self.IMAGE_ORDERING)(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)
        x = Conv2D(64, (3, 3), padding='same', name='block1_conv2', data_format=self.IMAGE_ORDERING)(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)
        f1 = x

        x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_avg_pool', data_format=self.IMAGE_ORDERING)(x)
        x = Dropout(self.dropouts[0])(x)


        # Block 2
        x = Conv2D(128, (3, 3), padding='same', name='block2_conv1', data_format=self.IMAGE_ORDERING)(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)
        x = Conv2D(128, (3, 3), padding='same', name='block2_conv2', data_format=self.IMAGE_ORDERING)(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)
        f2 = x

        x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_avg_pool', data_format=self.IMAGE_ORDERING)(x)
        x = Dropout(self.dropouts[1])(x)


        # Block 3
        x = Conv2D(256, (3, 3), padding='same', name='block3_conv1', data_format=self.IMAGE_ORDERING)(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)
        x = Conv2D(256, (3, 3), padding='same', name='block3_conv2', data_format=self.IMAGE_ORDERING)(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)
        x = Conv2D(256, (3, 3), padding='same', name='block3_conv3', data_format=self.IMAGE_ORDERING)(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)
        f3 = x

        x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_avg_pool', data_format=self.IMAGE_ORDERING)(x)
        x = Dropout(self.dropouts[2])(x)


        # Block 4
        x = Conv2D(512, (3, 3), padding='same', name='block4_conv1', data_format=self.IMAGE_ORDERING)(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)
        x = Conv2D(512, (3, 3), padding='same', name='block4_conv2', data_format=self.IMAGE_ORDERING)(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)
        x = Conv2D(512, (3, 3), padding='same', name='block4_conv3', data_format=self.IMAGE_ORDERING)(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)
        f4 = x

        x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_avg_pool', data_format=self.IMAGE_ORDERING)(x)
        x = Dropout(self.dropouts[3])(x)

        # Block 5 pyramid pooling block

        x = Conv2D(512, (3, 3), padding='same', name='block5_conv1', data_format=self.IMAGE_ORDERING)(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)
        x = Conv2D(512, (3, 3), padding='same', name='block5_conv2', data_format=self.IMAGE_ORDERING)(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)
        x = Conv2D(512, (3, 3), padding='same', name='block5_conv3', data_format=self.IMAGE_ORDERING)(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)
        f5 = x

        o = f5
        o = (ZeroPadding2D((1, 1), data_format=self.IMAGE_ORDERING))(o)
        o = (Conv2D(512, (3, 3), padding='valid', data_format=self.IMAGE_ORDERING))(o)
        o = (BatchNormalization())(o)
        o = (Activation('relu'))(o)
        o = (Conv2D(512, (3, 3), padding='same', data_format=self.IMAGE_ORDERING))(o)
        o = (BatchNormalization())(o)
        o = (Activation('relu'))(o)
        o = (Dropout(self.dropouts[4]))(o)

        o = (Conv2DTranspose(512,(2, 2), strides = (2,2), data_format=self.IMAGE_ORDERING))(o)
        o = (Concatenate(axis=MERGE_AXIS)([o, f4]))
        o = (ZeroPadding2D((1, 1), data_format=self.IMAGE_ORDERING))(o)
        o = (Conv2D(256, (3, 3), padding='valid', data_format=self.IMAGE_ORDERING))(o)
        o = (BatchNormalization())(o)
        o = (Activation('relu'))(o)
        o = (Conv2D(256, (3, 3), padding='same', data_format=self.IMAGE_ORDERING))(o)
        o = (BatchNormalization())(o)
        o = (Activation('relu'))(o)
        o = (Dropout(self.dropouts[5]))(o)

        o = (Conv2DTranspose(256,(2, 2),strides = (2,2), data_format=self.IMAGE_ORDERING))(o)
        o = (Concatenate(axis=MERGE_AXIS)([o, f3]))
        o = (ZeroPadding2D((1, 1), data_format=self.IMAGE_ORDERING))(o)
        o = (Conv2D(128, (3, 3), padding='valid', data_format=self.IMAGE_ORDERING))(o)
        o = (BatchNormalization())(o)
        o = (Activation('relu'))(o)
        o = (Conv2D(128, (3, 3), padding='same', data_format=self.IMAGE_ORDERING))(o)
        o = (BatchNormalization())(o)
        o = (Activation('relu'))(o)
        o = (Dropout(self.dropouts[6]))(o)

        o = (Conv2DTranspose (128,(2, 2),strides = (2,2), data_format=self.IMAGE_ORDERING))(o)
        o = (Concatenate(axis=MERGE_AXIS)([o, f2]))
        o = (ZeroPadding2D((1, 1), data_format=self.IMAGE_ORDERING))(o)
        o = (Conv2D(64, (3, 3), padding='valid', data_format=self.IMAGE_ORDERING))(o)
        o = (BatchNormalization())(o)
        o = (Activation('relu'))(o)
        o = (Conv2D(64, (3, 3), padding='same', data_format=self.IMAGE_ORDERING))(o)
        o = (BatchNormalization())(o)
        o = (Activation('relu'))(o)
        o = (Dropout(self.dropouts[7]))(o)

        o = (Conv2DTranspose (64,(2, 2),strides = (2,2), data_format=self.IMAGE_ORDERING))(o)
        o = (Concatenate(axis=MERGE_AXIS)([o, f1]))
        o = (ZeroPadding2D((1, 1), data_format=self.IMAGE_ORDERING))(o)
        o = (Conv2D(32, (3, 3), padding='valid', data_format=self.IMAGE_ORDERING))(o)
        o = (BatchNormalization())(o)
        o = (Activation('relu'))(o)
        o = (Conv2D(32, (3, 3), padding='same', data_format=self.IMAGE_ORDERING))(o)
        o = (BatchNormalization())(o)
        o = (Activation('relu'))(o)
        o = (Dropout(self.dropouts[8]))(o)

        o = Conv2D(self.n_classes, (1, 1), padding='same',name="logit_layer", data_format=self.IMAGE_ORDERING)(o)

        o = (Activation('softmax'))(o)

        if not 'name' in kwargs:
            kwargs['name'] = "VGG16_UNET"

        super(VGG16_UNET,self).__init__(inputs=img_input, outputs=o, **kwargs)
        self.in_shape = in_shape
        self.n_classes = n_classes
        self.image_ordering = image_ordering
        self.dropouts = dropouts


    def get_config(self):
        config = super(VGG16_UNET, self).get_config()
        config.update({
            'in_shape': self.in_shape,
            'n_classes': self.n_classes,
            'image_ordering': self.image_ordering,
            'dropouts': self.dropouts,
        })
        return config

    @classmethod
    def from_config(cls, config):
        print(config)
        return cls(**config)