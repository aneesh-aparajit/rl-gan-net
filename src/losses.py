import torch
import torch.nn as nn
import torch.nn.functional as F


def encoder_loss(mu: torch.tensor, std: torch.tensor, gen_out: torch.tensor, X: torch.tensor):
    kl_divergence = 0.5 * torch.sum(mu ** 2 + std ** 2 - torch.log(1e-8 + std ** 2) - 1) / torch.prod(X.shape)
    gen_mse       = nn.MSELoss()(gen_out, X)
    return kl_divergence + gen_mse


def generator_loss(reconstruction: torch.tensor, X: torch.tensor, y: torch.tensor):
    gen_loss = F.binary_cross_entropy(input=reconstruction, target=torch.ones_like(reconstruction))
    gen_diff_loss = F.mse_loss(input=reconstruction, target=y)
    return gen_loss + gen_diff_loss


def discriminator_loss(reconstruction: torch.tensor, X: torch.tensor, y: torch.tensor):
    # disc_real = F.binary_cross_entropy(input=y, target=)
    pass
