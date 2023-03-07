import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow import keras
from typing import List
from src.models.layers import Conv2dBlock, Conv2dTransposeBlock, Conv3dTransposeBlock, Conv3dBlock


class ImageEncoder(keras.Model):
    def __init__(
            self, 
            filters: List[int] = [64, 128, 256, 512, 400],     
            kernel_sizes: List[int] = [11, 5, 5, 5, 8], 
            strides: List[int] = [4, 2, 2, 2, 1], 
            padding: List[str] = ['same', 'same', 'same', 'same', 'valid']) -> None:
        super(ImageEncoder, self).__init__()
        self._layers = []
        
        for ix in range(len(filters)):
            self._layers.append(
                Conv2dBlock(
                    filters=filters[ix], kernel_size=kernel_sizes[ix], strides=strides[ix], 
                    padding=padding[ix], name=f'img_enc_conv2d_{ix}'
                )
            )
        
        self.encoder = keras.Sequential(layers=self._layers, name='image_encoder')
    
    def call(self, x: tf.Tensor, training: bool = True) -> tf.Tensor:
        latent = self.encoder(x, training)
        mu, std = latent[..., :200], latent[..., 200:]
        latent = mu + std * tf.random.normal(std.shape)
        return latent, mu, std


class Generator(keras.Model):
    def __init__(
            self, 
            filters: List[int] = [512, 256, 128, 64, 1], 
            kernel_sizes: List[int] = [4, 4, 4, 4, 4], 
            padding: List[str] = ['valid', 'same', 'same', 'same', 'same'],
            strides: List[int] = [1, 2, 2, 2, 2]) -> None:
        super(Generator, self).__init__()
        layers = []
        
        for ix in range(len(filters)):
            layers.append(
                Conv3dTransposeBlock(filters=filters[ix], kernel_size=kernel_sizes[ix], 
                                     strides=strides[ix], padding=padding[ix], name=f'gen_conv3d_{ix}')
            )
        
        self.gen = keras.Sequential(layers=layers, name='generator')
    
    def call(self, x: tf.Tensor, training: bool = True) -> tf.Tensor:
        return self.gen(tf.expand_dims(x, 1), training)


class Discriminator(keras.Model):
    def __init__(
            self, 
            filters: List[int] = [64, 128, 256, 512, 1], 
            kernel_sizes: List[int] = [4, 4, 4, 4, 4], 
            padding: List[str] = ['same', 'same', 'same', 'same', 'valid'],
            strides: List[int] = [2, 2, 2, 2, 1]) -> None:
        super(Discriminator, self).__init__()
        layers = []
        for ix in range(len(filters)):
            layers.append(
                Conv3dBlock(
                    filters=filters[ix], kernel_size=kernel_sizes[ix], strides=strides[ix], 
                    padding=padding[ix], name=f'disc_conv3d_{ix}'
                )
            )
        self.disc = keras.Sequential(layers=layers, name='discriminator')
        
    def call(self, x: tf.Tensor, training: bool = True) -> tf.Tensor:
        x = self.disc(x, training)
        return keras.activations.sigmoid(x)

