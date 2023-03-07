import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow import keras


class Conv2dBlock(keras.layers.Layer):
    def __init__(self, filters: int, kernel_size: int, strides: int, padding: str, name: str) -> None:
        super(Conv2dBlock, self).__init__()
        self.conv = keras.layers.Conv2D(
            filters=filters, kernel_size=kernel_size, strides=strides, padding=padding, name=f'{name}_conv'
        )
        self.bn = keras.layers.BatchNormalization()
    
    def call(self, x: tf.Tensor, training: bool = True) -> tf.Tensor:
        x = self.conv(x)
        reshape = False
        if len(x.shape) == 5:
            shape = x.shape
            reshape = True
            x = tf.reshape(x, [shape[0], shape[1], shape[2], -1])
            
        if reshape:
            x = tf.reshape(x, shape=shape)
        x = self.bn(x)
        x = keras.activations.relu(x)
        return x


class Conv2dTransposeBlock(keras.layers.Layer):
    def __init__(self, filters: int, kernel_size: int, strides: int, padding: str, name: str) -> None:
        super(Conv2dTransposeBlock, self).__init__()
        self.conv = keras.layers.Conv2DTranspose(
            filters=filters, kernel_size=kernel_size, strides=strides, padding=padding, name=f'{name}_conv'
        )
        self.bn = keras.layers.BatchNormalization()
    
    def call(self, x: tf.Tensor, training: bool = True) -> tf.Tensor:
        x = self.conv(x)
        reshape = False
        if len(x.shape) == 5:
            shape = x.shape
            reshape = True
            x = tf.reshape(x, [shape[0], shape[1], shape[2], -1])
            
        if reshape:
            x = tf.reshape(x, shape=shape)
        x = self.bn(x)
        x = keras.activations.relu(x)
        return x


class Conv3dBlock(keras.layers.Layer):
    def __init__(self, filters: int, kernel_size: int, strides: int, padding: str, name: str) -> None:
        super(Conv3dBlock, self).__init__()
        self.conv = keras.layers.Conv3D(
            filters=filters, kernel_size=kernel_size, strides=strides, padding=padding, name=f'{name}_conv'
        )
        self.bn = keras.layers.BatchNormalization()
    
    def call(self, x: tf.Tensor, training: bool = True) -> tf.Tensor:
        x = self.conv(x)
        reshape = False
        if len(x.shape) == 5:
            shape = x.shape
            reshape = True
            x = tf.reshape(x, [shape[0], shape[1], shape[2], -1])
        x = self.bn(x)
        if reshape:
            x = tf.reshape(x, shape=shape)
        x = keras.activations.relu(x)
        return x


class Conv3dTransposeBlock(keras.layers.Layer):
    def __init__(self, filters: int, kernel_size: int, strides: int, padding: str, name: str) -> None:
        super(Conv3dTransposeBlock, self).__init__()
        self.conv = keras.layers.Conv3DTranspose(
            filters=filters, kernel_size=kernel_size, strides=strides, padding=padding, name=f'{name}_conv'
        )
        self.bn = keras.layers.BatchNormalization()
    
    def call(self, x: tf.Tensor, training: bool = True) -> tf.Tensor:
        x = self.conv(x)
        reshape = False
        if len(x.shape) == 5:
            shape = x.shape
            reshape = True
            x = tf.reshape(x, [shape[0], shape[1], shape[2], -1])
        x = self.bn(x)
        if reshape:
            x = tf.reshape(x, shape=shape)
        x = keras.activations.relu(x)
        return x
