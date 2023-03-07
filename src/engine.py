import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler

from losses import encoder_loss, generator_loss, discriminator_loss
from tqdm import tqdm
import wandb
from src.config import CFG as cfg


def train_one_epoch(
    enc: nn.Module, gen: nn.Module, disc: nn.Module,
    enc_opt: optim, gen_opt: optim, disc_opt: optim,
    enc_scheduler: lr_scheduler, gen_scheduler: lr_scheduler, disc_scheduler: lr_scheduler, 
    trainloader: torch.utils.data.DataLoader, 
) -> float:
    
    enc.train()
    gen.train()
    disc.train()

    pbar = tqdm(enumerate(trainloader), total=len(trainloader), desc=f'Train ')
    for step, batch in pbar:
        X = batch['x'].to(cfg.device), batch['y'].to(cfg.device)
        latent, mu, std = enc.forward(X)
        reconstruction = gen.forward(latent)

        # encoder loss
        encoder_loss = encoder_loss(mu=mu, std=std, gen_out=reconstruction, X=X)

    return 0