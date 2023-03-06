import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import List


class ImageEncoder(nn.Module):
    def __init__(self, in_channels: int = 3, 
                 channels: List[int] = [64, 128, 256, 512, 400],
                 kernel_sizes: List[int] = [11, 5, 5, 5, 8],
                 strides: List[int] = [4, 2, 2, 2, 1]
    ) -> None:
        super(ImageEncoder, self).__init__()
        self.in_channels = in_channels
        layers = []
        
        for ix in range(len(channels)):
            layers.append(
                nn.Sequential(
                    nn.Conv2d(in_channels=in_channels, out_channels=channels[ix], kernel_size=kernel_sizes[ix], stride=strides[ix], padding=1), 
                    nn.BatchNorm2d(channels[ix]),
                    nn.ReLU()
                )
            )
            in_channels = channels[ix]
        
        self.net = nn.Sequential(*layers)
        
    def sample_normal(self, std: torch.Tensor) -> torch.Tensor:
        sampler = torch.distributions.Normal(loc=0, scale=1)
        return sampler.sample(std.shape)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        latent = self.net(x)
        mu, std = latent[:, :200,], latent[:, 200:]
        z = self.sample_normal(std)
        latent = mu + z * std
        return latent, mu, std
    

class Generator(nn.Module):
    def __init__(self, in_channels: int = 200, channels: List[int] = [512, 256, 128, 64, 1], kernel_sizes: List[int] = [4, 4, 4, 4, 4], strides: List[int] = [1, 2, 2, 2, 2], 
                 paddings: List[int] = [0, 1, 1, 1, 1]
    ) -> None:
        super(Generator, self).__init__()
        layers = []
        
        for ix in range(len(channels)):
            layers.append(
                nn.Sequential(
                    nn.ConvTranspose3d(in_channels=in_channels, out_channels=channels[ix], stride=strides[ix], kernel_size=kernel_sizes[ix], padding=paddings[ix]), 
                    nn.BatchNorm3d(channels[ix]), 
                    nn.ReLU()
                )
            )
            in_channels = channels[ix]
        layers.append(nn.Sigmoid())
        self.net = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
    

class Discriminator(nn.Module):
    def __init__(self, in_channels: int = 1, channels: List[int] = [64, 128, 256, 512, 1], kernel_sizes: List[int] = [4, 4, 4, 4, 4], strides: List[int] = [4, 2, 2, 2, 1]) -> None:
        super(Discriminator, self).__init__()
        layers = []
        
        for ix in range(len(channels)):
            layers.append(
                nn.Sequential(
                    nn.Conv3d(in_channels=in_channels, out_channels=channels[ix], kernel_size=kernel_sizes[ix], stride=strides[ix], padding=1), 
                    nn.BatchNorm3d(channels[ix]),
                    nn.LeakyReLU(0.2)
                )
            )
            in_channels = channels[ix]
        
        self.net = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

